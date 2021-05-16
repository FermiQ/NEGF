// NEMO5, The Nanoelectronics simulation package.
// This package is a free software.
// It is distributed under the NEMO5 Non-Commercial License (NNCL).
// Purdue Research Foundation, 1281 Win Hentschel Blvd., West Lafayette, IN 47906, USA
//$Id: PropagationUtilities.cpp 24104 $

#include "PropagationUtilities.h"
#include "GreensfunctionInterface.h"
#include "PropagationOptionsInterface.h"
#include "PropagatorInterface.h"
#include "NemoPhys.h"
#include "NemoMICUtils.h"
#include "QuantumNumberUtils.h"
#include "ComplexContourEnergyGrid.h"
#include "Greensolver.h"
#include "Domain.h"
#include "LinearSolverPetscComplex.h"
#include "LinearSolverPetscDouble.h"
#include "LinearSolverPetsc.h"
#include "EigensolverPetscDouble.h"
#include "EigensolverSlepcDouble.h"
#include "EigensolverSlepc.h"
#include "BlockSchroedingerModule.h"
#include "PetscMatrixParallelComplexContainer.h"
#include "Self_energy.h"
#include "NemoMatrixInterface.h"
#include "TransformationUtilities.h"
#include "Schroedinger.h"

/*
 * testing
 */
#include "NemoFactory.h"
#include "NemoMatrixComplex.h"
#include <string.h> //memcpy
#include <complex>


void PropagationUtilities::direct_iterative_leads(Simulation* this_simulation, Simulation* Hamilton_Constructor, std::map<std::string, NemoPhys::Momentum_type>& momentum_mesh_types,
                                                  std::map<std::string, const Propagator*>& Propagators, std::map<std::string, NemoPhys::Propagator_type>& Propagator_types,
                                                  const Domain* neighbor_domain, const std::vector<NemoMeshPoint>& momentum,
                                                  PetscMatrixParallelComplex*& result)
{
  const InputOptions& options=this_simulation->get_options();
  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::direct_iterative_leads ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::direct_iterative_leads() ";
  Propagation* this_propagation=dynamic_cast<Propagation*>(this_simulation);
  NEMO_ASSERT(this_propagation!=NULL,prefix+"should be called by a Propagation only\n");
 
  std::string name_of_writeable_Propagator=this_propagation->get_writeable_Propagator_name();
  std::map<std::string,Simulation*>* pointer_to_Propagator_Constructors=this_propagation->list_Propagator_constructors();

  PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;
  PetscMatrixParallelComplex* coupling_Overlap    =NULL;
  std::vector<NemoMeshPoint> sorted_momentum;


  //DOFmap "D3"="D1"+"D2" (for later usage)
  DOFmapInterface* coupling_DOFmap = NULL;
  NemoUtils::tic(tic_toc_prefix + "obtain Hamiltonian");
  const DOFmapInterface& device_DOFmap = Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain());

  if(!options.get_option("RF_with_kmesh",false))
  {
    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,momentum_mesh_types,Hamilton_Constructor);
  }
  else
  {
    for(unsigned int ii=0; ii<momentum.size(); ii++)
      if(momentum[ii].get_dim()==3)
        sorted_momentum.push_back(momentum[ii]);

    vector<string> Hamilton_momenta;
    options.get_option("Hamilton_momenta",Hamilton_momenta);
    InputOptions& writeable_solver_options = Hamilton_Constructor->get_reference_to_options();
    writeable_solver_options.set_option("quantum_number_order",Hamilton_momenta);
  }
  
  // get the Hamilton constructor of the neighboring domain by asking the Hamilton constructor of this domain
  Simulation* neighbor_HamiltonConstructor =  NULL;
  if (this_propagation->get_start_lead_Hamilton_Constructor() != NULL)
    neighbor_HamiltonConstructor = this_propagation->get_start_lead_Hamilton_Constructor();
  else if (this_propagation->get_end_lead_Hamilton_Constructor() != NULL)
    neighbor_HamiltonConstructor = this_propagation->get_end_lead_Hamilton_Constructor();
  else
  {
    std::string Hamilton_constructor_name;
    std::string variable_name="Hamilton_constructor_"+neighbor_domain->get_name();
    if (Hamilton_Constructor->get_options().check_option(variable_name))
      Hamilton_constructor_name=Hamilton_Constructor->get_options().get_option(variable_name,std::string(""));
    else
      Hamilton_constructor_name=Hamilton_Constructor->get_name(); //this is for the BlockSchroedingerModule
    //throw std::invalid_argument("Self_energy("+this->get_name()+")::direct_iterative_leads: define \""+variable_name+"\" in simulation (\""
    //                            +Hamilton_Constructor->get_name()+"\")\n");
    neighbor_HamiltonConstructor = this_simulation->find_simulation(Hamilton_constructor_name);
    if(neighbor_HamiltonConstructor==NULL)
      neighbor_HamiltonConstructor = Hamilton_Constructor->find_simulation(Hamilton_constructor_name);
    if (neighbor_HamiltonConstructor == NULL)
    {
      Simulation* ret_solver = this_propagation->get_forward_gR_solver();
      if (ret_solver == NULL)
      {
        throw std::runtime_error(prefix + "run without the forward_gR_solver defined\n");
        //std::string solver_name = options.get_option(ret_Green_cit->first+"_solver",std::string(""));
        //ret_solver = find_simulation(solver_name);
      }

      //NEMO_ASSERT(ret_solver!=NULL,prefix+"have not found solver \""+solver_name+"\"\n");
      neighbor_HamiltonConstructor = ret_solver->find_simulation(Hamilton_constructor_name);
    }
    NEMO_ASSERT(neighbor_HamiltonConstructor != NULL,
      "PropagationUtilities(" + this_simulation->get_name() + ")::direct_iterative_leads: Simulation \"" + Hamilton_constructor_name + "\" has not been found!\n");
  }

  

  //DOFmap "D2"
  DOFmapInterface& neighbor_domain_DOFmap = neighbor_HamiltonConstructor->get_dof_map(neighbor_domain);

  coupling_DOFmap=&neighbor_domain_DOFmap;
  DOFmapInterface* temp_pointer=coupling_DOFmap;
  Hamilton_Constructor->get_data(std::string("Hamiltonian"), sorted_momentum, neighbor_domain, coupling_Hamiltonian, coupling_DOFmap,
                                 this_simulation->get_const_simulation_domain());
  if(coupling_DOFmap!=temp_pointer)
    delete coupling_DOFmap;
  
  //coupling_Hamiltonian->assemble();
  //coupling_Hamiltonian->save_to_matlab_file("coupling_H.m");

  coupling_DOFmap=NULL;
  Hamilton_Constructor->get_data(string("overlap_matrix_coupling"), sorted_momentum, neighbor_domain, coupling_Overlap, coupling_DOFmap,
                                 this_simulation->get_const_simulation_domain());


  NemoUtils::toc(tic_toc_prefix + "obtain Hamiltonian");


  NemoUtils::tic(tic_toc_prefix + "get surface greens function");
  //find the retarded Greensfunction if writeable_Propagator.begin is a retarded self-energy (or lesser respectively)
  std::map<std::string, const Propagator*>::const_iterator ret_Green_cit=Propagators.begin();
  for(; ret_Green_cit!=Propagators.end(); ++ret_Green_cit)
  {
    std::map<std::string, NemoPhys::Propagator_type>::const_iterator c_prop_type_it=Propagator_types.find(ret_Green_cit->first);
    NEMO_ASSERT(c_prop_type_it!=Propagator_types.end(),
                prefix+"have not found \""+ret_Green_cit->first+"\" in the Propagator map\n");
    //make sure that the readable Propagator is the Green's function that corresponds to the writeable Propagator self-energy
    std::map<std::string, NemoPhys::Propagator_type>::const_iterator c_prop_type_it2=Propagator_types.find(name_of_writeable_Propagator);
    NEMO_ASSERT(c_prop_type_it2!=Propagator_types.end(),
                prefix+"have not found \""+name_of_writeable_Propagator+"\" in the Propagator map\n");
    //if(c_prop_type_it->second==Fermion_retarded_Green || c_prop_type_it->second==Boson_retarded_Green ||
    //   c_prop_type_it->second==Fermion_lesser_Green ||c_prop_type_it->second==Boson_lesser_Green) break;
    if((c_prop_type_it->second==NemoPhys::Fermion_retarded_Green && c_prop_type_it2->second==NemoPhys::Fermion_retarded_self) ||
        (c_prop_type_it->second==NemoPhys::Boson_retarded_Green && c_prop_type_it2->second==NemoPhys::Boson_retarded_self) ||
        (c_prop_type_it->second==NemoPhys::Fermion_lesser_Green && c_prop_type_it2->second==NemoPhys::Fermion_lesser_self) ||
        (c_prop_type_it->second==NemoPhys::Boson_lesser_Green && c_prop_type_it2->second==NemoPhys::Boson_lesser_self) ||
        (c_prop_type_it->second==NemoPhys::Fermion_greater_Green && c_prop_type_it2->second==NemoPhys::Fermion_greater_self) ||
        (c_prop_type_it->second==NemoPhys::Boson_greater_Green && c_prop_type_it2->second==NemoPhys::Boson_greater_self)) break;
  }


  NEMO_ASSERT(ret_Green_cit!=Propagators.end(),
              prefix+"have not found the Green's function\n");


  // -----------------------------------

  NemoPhys::Propagator_type temp_type=NemoPhys::Fermion_retarded_Green;
  NemoPhys::Propagator_type type_of_writeable_Propagator=this_propagation->get_Propagator_type(name_of_writeable_Propagator);
  if(type_of_writeable_Propagator==NemoPhys::Fermion_lesser_self)
    temp_type=NemoPhys::Fermion_lesser_Green;
  else if(type_of_writeable_Propagator==NemoPhys::Boson_retarded_self)
    temp_type=NemoPhys::Boson_retarded_Green;
  else if(type_of_writeable_Propagator==NemoPhys::Boson_lesser_self)
    temp_type=NemoPhys::Boson_lesser_Green;

  PetscMatrixParallelComplex* GR_matrix=NULL; // = temp_cit->second;
  if(!options.check_option(ret_Green_cit->first+"_solver"))
  {
    //std::map<std::string, std::set<Simulation*> >* pointer_to_solver_list = NULL;
    //get_Solver_list(ret_Green_cit->first, pointer_to_solver_list);
    Simulation* retGreen_solver=pointer_to_Propagator_Constructors->find(ret_Green_cit->first)->second;
    NEMO_ASSERT(retGreen_solver!=NULL,prefix+"constructor of \""+ret_Green_cit->first+"\" is NULL\n");
    //std::set<Simulation*> retGreen_solvers = pointer_to_solver_list->find(ret_Green_cit->first)->second;
    //(*(retGreen_solvers.begin()))->get_data(ret_Green_cit->first,&momentum,GR_matrix,&neighbor_domain_DOFmap);
    GreensfunctionInterface* temp_interface=dynamic_cast<GreensfunctionInterface*> (retGreen_solver);
    NEMO_ASSERT(temp_interface!=NULL,prefix+retGreen_solver->get_name()+" is not a GreensfunctionInterface\n");
    temp_interface->get_Greensfunction(momentum,GR_matrix,&neighbor_domain_DOFmap,NULL, temp_type);    
    //retGreen_solver->get_data(ret_Green_cit->first,&momentum,GR_matrix,&neighbor_domain_DOFmap);
  }
  else
  {
    std::string solver_name = options.get_option(ret_Green_cit->first+"_solver",std::string(""));
    Simulation* ret_solver = this_simulation->find_simulation(solver_name);
    NEMO_ASSERT(ret_solver!=NULL,prefix+"have not found solver \""+solver_name+"\"\n");
    //const DOFmap& neighbor_domain_DOFmap = ret_solver->get_const_dof_map(get_const_simulation_domain());
    
    GreensfunctionInterface* temp_interface=dynamic_cast<GreensfunctionInterface*> (ret_solver);
    NEMO_ASSERT(temp_interface!=NULL,prefix+ret_solver->get_name()+" is not a GreensfunctionInterface\n");
    temp_interface->get_Greensfunction(momentum,GR_matrix,&neighbor_domain_DOFmap,NULL, temp_type);
    //ret_solver->get_data(ret_Green_cit->first,&momentum,GR_matrix,&neighbor_domain_DOFmap);

    //msg<<prefix<<"received the retarded Greens function from "<<ret_solver->get_name()<<"\n";
  }
  NEMO_ASSERT(GR_matrix!=NULL, prefix+"received NULL matrix for GR\n");
  //GR_matrix->save_to_matlab_file(get_name()+std::string("test_GR.m"));
  NemoUtils::toc(tic_toc_prefix + "get surface greens function");

  NemoUtils::tic(tic_toc_prefix + "coupling_Hamiltonian_get_ownership_range");

  bool LRA_RGF = Hamilton_Constructor->get_type().find("Transformation") != std::string::npos;
  //also coupling Hamiltonian is different

  unsigned int number_of_super_rows;
  unsigned int number_of_super_cols;
  unsigned int number_of_own_super_rows;
  int start_own_super_rows;
  int end_own_super_rows_p1;
  unsigned int number_of_rows2;
  unsigned int number_of_cols2;
  unsigned int local_number_of_rows2;
  number_of_super_rows = coupling_Hamiltonian->get_num_rows();
  number_of_super_cols = coupling_Hamiltonian->get_num_cols();
  number_of_own_super_rows = coupling_Hamiltonian->get_num_owned_rows();
  coupling_Hamiltonian->get_ownership_range(start_own_super_rows, end_own_super_rows_p1);
  NemoUtils::toc(tic_toc_prefix + "coupling_Hamiltonian_get_ownership_range");
  //NEMO_ASSERT(number_of_super_rows == number_of_super_cols,
  //            prefix+"rectangular matrix received for the super-domain\n");

  //these are the dimensions of the device matrices:
  number_of_rows2 = device_DOFmap.get_global_dof_number();

  number_of_cols2 = number_of_rows2;
  local_number_of_rows2 = device_DOFmap.get_number_of_dofs();

  //NEMO_ASSERT(number_of_rows2==number_of_cols2,
  //            prefix+"rectangular matrix received for this domain\n");
  //these are the dimensions of the lead matrices:
  unsigned int number_of_rows1      = number_of_super_rows-number_of_rows2;
  unsigned int number_of_cols1      = number_of_super_cols-number_of_cols2;
  //J.C. the idea here is that from the LRAModule the get_data will return the matrix to be used in the multiplication
  //not just the large matrix. so if this happens this conditional will be true and the elements should be set as such.
  //in this case we should also have a dense sigma.
  //still needs to be tested
  bool use_dense = false;
  if(coupling_Hamiltonian->get_num_rows() <= number_of_rows2)
  {
    number_of_rows1 = number_of_super_rows;
    number_of_cols1 = number_of_super_cols;
    number_of_rows2 = number_of_super_rows;
    number_of_cols2  = 0;//number_of_super_cols;
    use_dense = true;
  }
  //NEMO_ASSERT(number_of_rows1==number_of_cols1,
  //            prefix+"rectangular matrix received for neighbor domain\n");
  bool oldversion=options.get_option("old_mult_version",bool(false));//for debug purpose
  if(oldversion)
  {
    //===============================================
    //Yu: this is old version -- super matrix concept
    //-----------------------------------------------
    PetscMatrixParallelComplex* super_GR_matrix = new PetscMatrixParallelComplex(number_of_super_rows,number_of_super_cols,
        this_simulation->get_simulation_domain()->get_communicator());

    // set the sparsity pattern (using the known pattern of GR in its domain - the rest of super_GR is zero
    //super_GR_matrix->set_num_owned_rows(number_of_super_rows);
    NemoUtils::tic(tic_toc_prefix + "super_GR_matrx->set_num_owned_rows");
    super_GR_matrix->set_num_owned_rows(number_of_own_super_rows);
    NemoUtils::toc(tic_toc_prefix + "super_GR_matrx->set_num_owned_rows");
    //we assume that GR is the lower right block of super_GR
    //since GR_matrix is dense, we distribute the local nonzeros equally
    NemoUtils::tic(tic_toc_prefix + "super_GR_matrx set num nonzero");
    int size = -1;
    MPI_Comm_size(this_simulation->get_simulation_domain()->get_communicator(), &size);
    //int rank = -1;
    //MPI_Comm_rank(NEMO::geometry_communicator, &rank);
    unsigned int constant_local_nonzeros = std::min(number_of_rows1/size+1,number_of_rows1);
    unsigned int local_nonzeros = std::max(int(number_of_rows1-(end_own_super_rows_p1-start_own_super_rows)),0);
    //for (unsigned int i = 0; i < number_of_rows1; i++)
    //  super_GR_matrix->set_num_nonzeros(i,constant_local_nonzeros,number_of_rows1-constant_local_nonzeros);
    //for (unsigned int i = number_of_rows1; i < number_of_super_rows; i++)
    //  super_GR_matrix->set_num_nonzeros(i,0,0);
    //for (unsigned int i = start_own_super_rows; i < end_own_super_rows_p1 && i < number_of_rows1; i++)
    for (unsigned int i = 0; i < number_of_rows2+start_own_super_rows; i++)
      super_GR_matrix->set_num_nonzeros(i,0,0);
    for (unsigned int i = start_own_super_rows+number_of_rows2; i < end_own_super_rows_p1+number_of_rows2 && i < number_of_super_rows; i++)
    {
      //super_GR_matrix->set_num_nonzeros(i,constant_local_nonzeros,number_of_rows1-constant_local_nonzeros);
      super_GR_matrix->set_num_nonzeros(i,end_own_super_rows_p1-start_own_super_rows,local_nonzeros);
    }
    for (unsigned int i = end_own_super_rows_p1+number_of_rows2; i < number_of_super_rows; i++)
      super_GR_matrix->set_num_nonzeros(i,0,0);
    NemoUtils::toc(tic_toc_prefix + "super_GR_matrx set num nonzero");
    //msg << "Self_energy("<<this->get_name()<<")::direct_iterative_leads: allocating memory\n";
    NemoUtils::tic(tic_toc_prefix + "super_GR_matrx allocate memory");
    super_GR_matrix->allocate_memory();
    NemoUtils::toc(tic_toc_prefix + "super_GR_matrx allocate memory");
    //msg << "Self_energy("<<this->get_name()<<")::direct_iterative_leads: memory allocated\n";
    NemoUtils::tic(tic_toc_prefix + "super_GR_matrx set matrix elements");
    //fill the lower right corner of super_GR_matrix with GR_matrix (lead rows come after the device rows)
    //NOTE: might want to replace with array-filling scheme later (using e.g. MatGetArray)
    //NOTE: this works for two cases: 1)GR_matrix is defined on the neighbor domain; 2) GR_matrix is defined on this domain which is an equivalent domain to the neighbor domain
    //for(unsigned int i=number_of_rows2;i<number_of_super_rows;i++)
    //  for(unsigned int j=number_of_cols2;j<number_of_super_cols;j++)
    //    super_GR_matrix->set(i,j,GR_matrix->get(i-number_of_rows2,j-number_of_cols2)); //NOTE: the commented for loops are covered by the transfer_matrix_set_matrix_elements call
    //transfer_matrix_set_matrix_elements(number_of_rows2,number_of_super_rows,number_of_cols2,number_of_super_cols,-number_of_rows2,-number_of_cols2,GR_matrix,super_GR_matrix);
    if(GR_matrix->if_container())
      GR_matrix->assemble();
    const std::complex<double>* pointer_to_data= NULL;
    //  vector<const std::complex<double> > pointer_to_data;
    vector<cplx> data_vector;
    vector<int> col_index;
    int n_nonzeros=0;
    const int* n_col_nums=NULL;
    //vector<const int > n_col_nums;
    for(unsigned int i=number_of_rows2; i<number_of_super_rows; i++)
    {
      GR_matrix->get_row(i-number_of_rows2,&n_nonzeros,n_col_nums,pointer_to_data);
      //GR_matrix->get_row(i-number_of_rows2,&n_nonzeros,&(n_col_nums[0]),&(pointer_to_data[0]));
      col_index.resize(n_nonzeros,0);
      data_vector.resize(n_nonzeros,cplx(0.0,0.0));
      for(int j=0; j<n_nonzeros; j++)
      {
        col_index[j]=n_col_nums[j]+number_of_cols2;
        std::complex<double> temp_val=pointer_to_data[j];
        data_vector[j]=temp_val;
      }
      super_GR_matrix->set(i,col_index,data_vector);
      GR_matrix->store_row(i-number_of_rows2,&n_nonzeros,n_col_nums,pointer_to_data);
      //GR_matrix->store_row(i-number_of_rows2,&n_nonzeros,&(n_col_nums[0]),&(pointer_to_data[0]));
    }
    NemoUtils::toc(tic_toc_prefix + "super_GR_matrx set matrix elements");

    NemoUtils::tic(tic_toc_prefix + "super_GR_matrix assemble");
    super_GR_matrix->assemble();
    NemoUtils::toc(tic_toc_prefix + "super_GR_matrix assemble");
    //super_GR_matrix->save_to_matlab_file(get_name()+"super_testGR.m");
    //super_GR_matrix->save_to_matlab_file(this->get_name()+"super_GR"+temp_name2+".m");
    //result = GR_lead * H_lead_device
    PetscMatrixParallelComplex* super_result = NULL;
    //coupling_Hamiltonian->save_to_matlab_file(get_name()+"coupling_Ham.m")
    NemoUtils::tic(tic_toc_prefix + "H_lead_device*GRlead mult");
    PetscMatrixParallelComplex::mult(*coupling_Hamiltonian,*super_GR_matrix,&super_result);
    NemoUtils::toc(tic_toc_prefix + "H_lead_device*GRlead mult");
    delete super_GR_matrix;
    super_GR_matrix=NULL;
    //super_GR_matrix = H_device_lead * GR_lead * H_lead_device
    NemoUtils::tic(tic_toc_prefix + "hermitian transpose");
    coupling_Hamiltonian->hermitian_transpose_matrix(*coupling_Hamiltonian,MAT_REUSE_MATRIX);
    NemoUtils::toc(tic_toc_prefix + "hermitian transpose");
    //coupling_Hamiltonian->save_to_matlab_file(get_name()+"Herm_couling_Ham.m");
    NemoUtils::tic(tic_toc_prefix + "H_lead_device*GRlead*H_lead_device mult");
    PetscMatrixParallelComplex::mult(*super_result,*coupling_Hamiltonian,&super_GR_matrix);
    NemoUtils::toc(tic_toc_prefix + "H_lead_device*GRlead*H_lead_device mult");
    //coupling_Hamiltonian->hermitian_transpose_matrix(*coupling_Hamiltonian,MAT_REUSE_MATRIX);
    //super_GR_matrix->save_to_matlab_file("super_contact_self.m");

    std::vector<int> temp_rows(number_of_rows2);
    std::vector<int> temp_cols(number_of_cols2);
    for(unsigned int i=0; i<number_of_rows2; i++)
    {
      //temp_rows[i]=number_of_rows1+i;
      temp_rows[i]=i;
    }
    for(unsigned int i=0; i<number_of_cols2; i++)
    {
      //temp_cols[i]=number_of_cols1+i;
      temp_cols[i]=i;
    }

    NemoUtils::tic(tic_toc_prefix + "set result matrix");
    //set the sparsity pattern of the result matrix
    result= new PetscMatrixParallelComplex(number_of_rows2,number_of_cols2,this_simulation->get_simulation_domain()->get_communicator());
    result->set_num_owned_rows(local_number_of_rows2);
    constant_local_nonzeros = std::min(number_of_rows2/size+1,number_of_rows2);
    for(unsigned int i=0; i<number_of_cols2; i++)
      result->set_num_nonzeros(i,constant_local_nonzeros,number_of_rows2-constant_local_nonzeros);
    //copy the calculated Hamiltonian elements to result
    super_GR_matrix->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,result);
    result->assemble();
    NemoUtils::toc(tic_toc_prefix + "set result matrix");
    //bool Gr_method=options.get_option("Gr_by_transfer_matrix", bool(false));
    //if(Gr_method)
    //  result->hermitian_transpose_matrix(*result,MAT_REUSE_MATRIX);

    //std::string temp_name=get_name()+std::string("contact_self");
    //std::ostringstream strs;
    //for (unsigned int i=0;i<momentum.size();i++)
    //{
    //  std::vector<double> temp_vector=momentum[i].get_coords();
    //  for (unsigned int j=0;j<temp_vector.size();j++)
    //  {
    //    if (temp_vector[j]>=0.0)
    //      strs << fabs(temp_vector[j]);
    //    else
    //      strs << "m"<<fabs(temp_vector[j]);
    //  }
    //}
    //std::string str = strs.str();
    //temp_name+=str;
    //result->save_to_matlab_file(temp_name+std::string(".m"));

    //std::string temp_name2;
    //const std::vector<NemoMeshPoint>* temp_vector_pointer=&(momentum);
    //translate_momentum_vector(temp_vector_pointer,temp_name2);
    //Hamiltonian->save_to_matlab_file(this->get_name()+"_Ham_"+".m");
    //result->save_to_matlab_file(get_name()+"_oldsigma.m");

    delete super_GR_matrix;
    super_GR_matrix=NULL;
    delete super_result;
    super_result=NULL;

    delete coupling_DOFmap;
    coupling_DOFmap=NULL;
    if(coupling_Hamiltonian!=NULL)
    {
      delete coupling_Hamiltonian;
      coupling_Hamiltonian=NULL;
    }
    //-------------------------------------
    //Yu: end of old version
    //=====================================
  }
  else
  {
    bool include_sigma_scattering = false;
    if(this_propagation->get_scattering_sigmaR_solver()!=NULL || this_propagation->get_scattering_sigmaL_solver()!=NULL)
      include_sigma_scattering = true;

    PetscMatrixParallelComplex* coupling = NULL;
    PetscMatrixParallelComplex* S_matrix = NULL;



    //=====================================
    //Yu: new version -- avoid super matrix
    //-------------------------------------
    //1. get the submatrix (upper right block) from coupling Hamiltonian and (optionally) the S-matrix
    //1.1 figure out the start rows and end rows of nonzero block
    NemoUtils::tic(tic_toc_prefix + "get submatrix from coupling Hamiltonian");
    unsigned int start_row = 0;
    unsigned int end_row = 0;
    unsigned int nonzero_index = 0;

    for(unsigned int i=0; i<number_of_rows2; i++)
    {
      nonzero_index = coupling_Hamiltonian->get_nz_diagonal(i);
      if(nonzero_index>0)
      {
        start_row = i;
        break;
      }
    }
    nonzero_index = 0;
    for(unsigned int i=number_of_rows2; i>0; i--)
    {
      nonzero_index = coupling_Hamiltonian->get_nz_diagonal(i-1);
      if(nonzero_index>0)
      {
        end_row = i-1;
        break;
      }
    }
    unsigned int number_of_couple_rows = end_row + 1 - start_row;
    //1.2 get sub block
    std::vector<int> temp_rows(number_of_couple_rows);
    std::vector<int> temp_cols(number_of_cols1);
    for(unsigned int i=0; i<number_of_couple_rows; i++)
      temp_rows[i]=i+start_row;
    for(unsigned int i=0; i<number_of_cols1; i++)
      temp_cols[i]=i+number_of_cols2;


    coupling = new PetscMatrixParallelComplex(number_of_couple_rows, number_of_cols1, this_simulation->get_simulation_domain()->get_communicator());
    coupling->set_num_owned_rows(number_of_couple_rows);
    vector<int> rows_diagonal(number_of_couple_rows,0);
    vector<int> rows_offdiagonal(number_of_couple_rows,0);
    unsigned int number_of_nonzero_cols_local=0;
    unsigned int number_of_nonzero_cols_nonlocal=0;
    PetscMatrixParallelComplex* temp_result=NULL;
    if(!include_sigma_scattering)
    {
      for (unsigned int i = 0; i<number_of_couple_rows; i++)
      {
        rows_diagonal[i] = coupling_Hamiltonian->get_nz_diagonal(i + start_row);
        rows_offdiagonal[i] = coupling_Hamiltonian->get_nz_offdiagonal(i + start_row);
        if (rows_diagonal[i]>0)
          number_of_nonzero_cols_local++;
        if (rows_offdiagonal[i] > 0)
          number_of_nonzero_cols_nonlocal++;
      }
      //setup sparsity pattern of the coupling Hamiltonian
      for(unsigned int i = 0; i < number_of_couple_rows; i++)
        coupling->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);
      coupling_Hamiltonian->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, coupling);
      coupling->assemble();
      //optional: set up sparsity pattern of S-matrix
      if(coupling_Overlap!=NULL)
      {
        S_matrix = new PetscMatrixParallelComplex(number_of_couple_rows, number_of_cols1, this_simulation->get_simulation_domain()->get_communicator());
        S_matrix->set_num_owned_rows(number_of_couple_rows);
        for(unsigned int i = 0; i < number_of_couple_rows; i++)
          S_matrix->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);
        coupling_Overlap->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, S_matrix);
        S_matrix->assemble();
        //S_matrix->save_to_matlab_file("test_S_matrix.m");
      }

      NemoUtils::toc(tic_toc_prefix + "get submatrix from coupling Hamiltonian");
      
      if(GR_matrix->if_container())
        GR_matrix->assemble();
      //coupling->save_to_matlab_file("coupling.m");
      //GR_matrix->save_to_matlab_file(get_name()+"_GR.m");

      //2. perform the multiplication

      PetscMatrixParallelComplex* temp_matrix1=NULL;

      NemoUtils::tic(tic_toc_prefix + "H_lead_device*GRlead mult");
      if(S_matrix!=NULL)
      {
        NEMO_ASSERT(Propagators.size()>0,prefix+"have found empty Propagators map\n");
        const Propagator* temp_propagator=Propagators.begin()->second; // *(known_Propagators.begin());
        const double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum, temp_propagator);
        *S_matrix *= std::complex<double> (energy,0.0); //ES
        coupling->add_matrix(*S_matrix,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));//H-ES
        *coupling *= std::complex<double>(-1.0,0.0); //ES-H
        delete S_matrix;
        S_matrix=NULL;
      }

      PropagationUtilities::supress_noise(this_simulation,coupling);
      PropagationUtilities::supress_noise(this_simulation,GR_matrix);
      if(this_propagation->get_debug_output())
      {
        coupling->save_to_matlab_file("DLH_RGF.m");
        GR_matrix->save_to_matlab_file("GR_RGF.m");
      }

      PetscMatrixParallelComplex::mult(*coupling,*GR_matrix,&temp_matrix1);

      PropagationUtilities::supress_noise(this_simulation,temp_matrix1);
      //temp_matrix1->save_to_matlab_file("HxGR.m");
      NemoUtils::toc(tic_toc_prefix + "H_lead_device*GRlead mult");

      NemoUtils::tic(tic_toc_prefix + "hermitian transpose");
      PetscMatrixParallelComplex transpose_coupling(coupling->get_num_cols(), coupling->get_num_rows(),
          coupling->get_communicator());
      coupling->hermitian_transpose_matrix(transpose_coupling, MAT_INITIAL_MATRIX);
      delete coupling;
      coupling=NULL;
      NemoUtils::toc(tic_toc_prefix + "hermitian transpose");

      NemoUtils::tic(tic_toc_prefix + "H_lead_device*GRlead*H_device_lead mult");
      PropagationUtilities::supress_noise(this_simulation,&transpose_coupling);
      PetscMatrixParallelComplex::mult(*temp_matrix1,transpose_coupling,&temp_result);
      PropagationUtilities::supress_noise(this_simulation,temp_result);
      if(this_propagation->get_debug_output())
        temp_result->save_to_matlab_file("sigma_RGF.m");
      NemoUtils::toc(tic_toc_prefix + "H_lead_device*GRlead*H_device_lead mult");

      delete temp_matrix1;
      temp_matrix1=NULL;
    }
    else // include scattering sigma
    {
      PetscMatrixParallelComplex *off_sigma = NULL;
      if(this_propagation->get_scattering_sigmaR_solver()!=NULL)
      {
        std::string temp_name;
        this_propagation->get_scattering_sigmaR_solver()->get_data("writeable_Propagator",temp_name);
        this_propagation->get_scattering_sigmaR_solver()->get_data(temp_name,&momentum, off_sigma,
            &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),&neighbor_domain_DOFmap);
      }
      else //scattering_sigmaL_solver!=NULL
      {
        std::string temp_name;
        this_propagation->get_scattering_sigmaL_solver()->get_data("writeable_Propagator",temp_name);
        this_propagation->get_scattering_sigmaL_solver()->get_data(temp_name,&momentum, off_sigma,
            &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),&neighbor_domain_DOFmap);
      }

      std::vector<int> temp_rows(local_number_of_rows2);
      std::vector<int> temp_cols(number_of_cols1);
      for (unsigned int i = 0; i<local_number_of_rows2 && i<number_of_rows2; i++)
        temp_rows[i] = i;
      for (unsigned int i = 0; i<number_of_cols1; i++)
        temp_cols[i] = i + number_of_cols2;
      delete coupling;
      //PetscMatrixParallelComplex* coupling = NULL;
      coupling = new PetscMatrixParallelComplex(number_of_rows2, number_of_cols1, this_simulation->get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
      coupling->set_num_owned_rows(local_number_of_rows2);
      rows_diagonal.resize(local_number_of_rows2);
      rows_offdiagonal.resize(local_number_of_rows2);

      for (unsigned int i = 0; i<local_number_of_rows2; i++)
      {
        rows_diagonal[i] = std::max(coupling_Hamiltonian->get_nz_diagonal(i),off_sigma->get_nz_diagonal(i));
        rows_offdiagonal[i] = std::max(coupling_Hamiltonian->get_nz_offdiagonal(i),off_sigma->get_nz_offdiagonal(i));

        if (rows_diagonal[i]>0)
          number_of_nonzero_cols_local++;
        if (rows_offdiagonal[i] > 0)
          number_of_nonzero_cols_nonlocal++;
      }
      for (unsigned int i = 0; i<local_number_of_rows2; i++)
        coupling->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);
      coupling_Hamiltonian->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, coupling);

      coupling->assemble();
      start_row = 0;

      PetscMatrixParallelComplex transpose_coupling(coupling->get_num_cols(), coupling->get_num_rows(),
          coupling->get_communicator());

      if(temp_type == NemoPhys::Fermion_lesser_Green || temp_type == NemoPhys::Boson_lesser_Green)
      {
        coupling->add_matrix(*off_sigma,DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
        coupling->hermitian_transpose_matrix(transpose_coupling, MAT_INITIAL_MATRIX);
      }
      else
      {
        coupling->hermitian_transpose_matrix(transpose_coupling, MAT_INITIAL_MATRIX);
        coupling->add_matrix(*off_sigma,DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
        transpose_coupling.add_matrix(*off_sigma,DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
      }

      //cerr << get_name() << "SE coupling_with_sigma" << get_const_simulation_domain()->get_name() << " "  << coupling->get(0,0) << " \n";

      if(GR_matrix->if_container())
        GR_matrix->assemble();
      
      PetscMatrixParallelComplex* temp_matrix1=NULL;
      PetscMatrixParallelComplex::mult(*coupling,*GR_matrix,&temp_matrix1);
   
      PropagationUtilities::supress_noise(this_simulation,temp_matrix1);
      //temp_matrix1->save_to_matlab_file("HxGR.m");
      delete coupling;
      coupling=NULL;
      NemoUtils::toc(tic_toc_prefix + "hermitian transpose");

      NemoUtils::tic(tic_toc_prefix + "H_lead_device*GRlead*H_device_lead mult");
      PropagationUtilities::supress_noise(this_simulation,&transpose_coupling);
      PetscMatrixParallelComplex::mult(*temp_matrix1,transpose_coupling,&temp_result);
     
      PropagationUtilities::supress_noise(this_simulation,temp_result);

      delete temp_matrix1;
      temp_matrix1=NULL;
    }

    // Jun Huang: if LRA_RGF, the self energy is dense, no need to copy to a sparse matrix
    if (LRA_RGF || use_dense)
    {
      result=temp_result;
    }
    else
    {
      //3. copy result into a sparse matrix
      NemoUtils::tic(tic_toc_prefix + "set result matrix");
      vector<int> result_diagonal(number_of_rows2,0);
      vector<int> result_offdiagonal(number_of_rows2,0);
      for(unsigned int i=start_row; i<=end_row; i++)
      {
        if(rows_diagonal[i-start_row]>0)
          result_diagonal[i]=number_of_nonzero_cols_local;
        if(rows_offdiagonal[i-start_row]>0)
          result_offdiagonal[i]=number_of_nonzero_cols_nonlocal;
      }
      PropagationUtilities::transfer_matrix_initialize_temp_matrix(this_simulation,number_of_rows2,number_of_rows2,result_diagonal,result_offdiagonal,result);
      const std::complex<double>* pointer_to_data= NULL;
      vector<cplx> data_vector;
      vector<int> col_index;
      int n_nonzeros=0;
      const int* n_col_nums=NULL;
      for(unsigned int i=0; i<temp_result->get_num_rows(); i++)
      {
        if(rows_diagonal[i]>0)
        {
          temp_result->get_row(i,&n_nonzeros,n_col_nums,pointer_to_data);
          //Yu He: petsc tells us that temp_result is always full
          //n_nonzeros must be the same as it number of columns
          col_index.resize(number_of_nonzero_cols_local,0);
          data_vector.resize(number_of_nonzero_cols_local,cplx(0.0,0.0));
          unsigned int result_i=0;
          for(int j=0; j<n_nonzeros; j++)
          {
            if(rows_diagonal[j]>0)
            {
              col_index[result_i]=n_col_nums[j]+start_row;
              data_vector[result_i]=pointer_to_data[j];
              result_i++;
            }
          }

          result->set(i+start_row,col_index,data_vector);
          temp_result->store_row(i,&n_nonzeros,n_col_nums,pointer_to_data);
        }
      }
      result->assemble();
      NemoUtils::toc(tic_toc_prefix + "set result matrix");
      delete temp_result;
      temp_result=NULL;
      PropagationUtilities::supress_noise(this_propagation,result);
    }

    delete coupling_DOFmap;
    coupling_DOFmap=NULL;
    if(coupling_Hamiltonian!=NULL)
    {
      delete coupling_Hamiltonian;
      coupling_Hamiltonian=NULL;
    }
    //-------------------------------------
    //Yu: end of new version
    //=====================================
  }
  
  //throw std::invalid_argument("Self_energy::direct_iterative_leads not implemented, yet\n");
  //result->save_to_matlab_file("self_energy_" + get_name() + get_const_simulation_domain()->get_name() + ".m");
  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::transfer_matrix_leads(Simulation* this_simulation, Propagator*& output_Propagator,const Domain* neighbor_domain, const std::vector<NemoMeshPoint>& momentum,
  PetscMatrixParallelComplex*& result)
{
  //main function for transfer matrix method
  //i.prepare matrix for eigenvalue problem
  //ii.solve eigenvalue problem for eigenvalues and modes
  //iii.solve self-energy
  //ref: PRB 74, 205323 (2006)
  std::string tic_toc_name = this_simulation->get_name();
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::transfer_matrix_leads ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_leads()";
  msg<<prefix<<std::endl;

  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  
  //NemoPhys::transfer_matrix_type temp_transfer_type=NemoPhys::into_device_propagating_modes;
  //PetscMatrixParallelComplex* into_device_propagating_modes=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_propagating_modes,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::into_device_decaying_modes;
  //PetscMatrixParallelComplex* into_device_decaying_modes=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_decaying_modes,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_propagating_modes;
  //PetscMatrixParallelComplex* out_of_device_propagating_modes=NULL;
  //PropInterface->get_transfer_matrix_element(out_of_device_propagating_modes,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_decaying_modes;
  //PetscMatrixParallelComplex* out_of_device_decaying_modes=NULL; 
  //PropInterface->get_transfer_matrix_element(out_of_device_decaying_modes,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::into_device_propagating_phase;
  //PetscMatrixParallelComplex* into_device_propagating_phase=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_propagating_phase,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::into_device_decaying_phase;
  //PetscMatrixParallelComplex* into_device_decaying_phase=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_decaying_phase,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_propagating_phase;
  //PetscMatrixParallelComplex* out_of_device_propagating_phase=NULL; 
  //PropInterface->get_transfer_matrix_element(out_of_device_propagating_phase,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_decaying_phase;
  //PetscMatrixParallelComplex* out_of_device_decaying_phase=NULL; 
  //PropInterface->get_transfer_matrix_element(out_of_device_decaying_phase,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::into_device_modes;
  //PetscMatrixParallelComplex* into_device_modes=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_modes,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::into_device_phase;
  //PetscMatrixParallelComplex* into_device_phase=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_phase,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_modes;
  //PetscMatrixParallelComplex* out_of_device_modes=NULL; 
  //PropInterface->get_transfer_matrix_element(out_of_device_modes,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_phase;
  //PetscMatrixParallelComplex* out_of_device_phase=NULL; 
  //PropInterface->get_transfer_matrix_element(out_of_device_phase,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::into_device_velocity;
  //PetscMatrixParallelComplex* into_device_velocity=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_velocity,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::into_device_propagating_velocity;
  //PetscMatrixParallelComplex* into_device_propagating_velocity=NULL; 
  //PropInterface->get_transfer_matrix_element(into_device_propagating_velocity,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_velocity;
  //PetscMatrixParallelComplex* out_of_device_velocity=NULL; 
  //PropInterface->get_transfer_matrix_element(out_of_device_velocity,&temp_transfer_type);

  //temp_transfer_type=NemoPhys::out_of_device_propagating_velocity;
  //PetscMatrixParallelComplex* out_of_device_propagating_velocity=NULL; 
  //PropInterface->get_transfer_matrix_element(out_of_device_propagating_velocity,&temp_transfer_type);

  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();
  InputOptions options = this_simulation->get_reference_to_options();
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);


  std::string time_for_transfer_matrix_leads = this_simulation->get_name()+"_transfer_matrix_leads()";
  NemoUtils::tic(time_for_transfer_matrix_leads);

  std::string temp_name2;
  const std::vector<NemoMeshPoint>* temp_vector_pointer=&(momentum);
  translate_momentum_vector(this_simulation,temp_vector_pointer,temp_name2);

  //set_valley(writeable_Propagators.begin()->second, momentum, Hamilton_Constructor); //Bozidar

  //process flow: e.g. 4 atomic layers described in PRB 74, 205323 (2006)
  //1. Hamiltonian Hs and coupling_Hamiltonian Ts for atomic layer
  //H=|H00 H01 0     0|; T=|0 0 0 H43|; T'=|0   0 0 0|
  //  |H10 H11 H12   0|    |0 0 0 0  |     |0   0 0 0|
  //  |0   H21 H22 H23|    |0 0 0 0  |     |0   0 0 0|
  //  |0   0   H32 H33|    |0 0 0 0  |     |H34 0 0 0|
  Domain* neighbor_domain2=NULL;
  bool new_self_energy = false;
  std::string retarded_lead_method=options.get_option("retarded_lead_method", std::string("direct_iterations"));
  if(retarded_lead_method.find("avoid_surface_greens_function")!=std::string::npos) new_self_energy = true; //for OMEN-QTBM this is used

  //1a. get Hamiltonian and coupling term (and DOF) for each subdomain 0~3, 3 is always labelled as the one attached to device
  PetscMatrixParallelComplex* D00 = NULL;
  PetscMatrixParallelComplex* H01 = NULL;
  PetscMatrixParallelComplex* S00 = NULL;
  PetscMatrixParallelComplex* S01 = NULL;

  //DOFmap subdomain_DOF0;
  const DOFmapInterface* pointer_to_subdomain_DOF0=NULL;
  DOFmapInterface* subdomain_coupling_DOF0=NULL;
  PropagationUtilities::transfer_matrix_get_Hamiltonian(this_simulation,std::string("0"),output_Propagator,neighbor_domain2, momentum, D00, H01,S00, S01, pointer_to_subdomain_DOF0, subdomain_coupling_DOF0);
  //delete subdomain_coupling_DOF0;

  PetscMatrixParallelComplex* D11 = NULL;
  PetscMatrixParallelComplex* H12 = NULL;
  PetscMatrixParallelComplex* S11 = NULL;
  PetscMatrixParallelComplex* S12 = NULL;

  //DOFmap subdomain_DOF1;
  const DOFmapInterface* pointer_to_subdomain_DOF1=NULL;
  DOFmapInterface* subdomain_coupling_DOF1=NULL;
  PropagationUtilities::transfer_matrix_get_Hamiltonian(this_simulation,std::string("1"),output_Propagator,neighbor_domain2, momentum, D11, H12, S11, S12, pointer_to_subdomain_DOF1, subdomain_coupling_DOF1);
  delete subdomain_coupling_DOF1;
  //delete pointer_to_subdomain_DOF1;
  PetscMatrixParallelComplex* D22 = NULL;
  PetscMatrixParallelComplex* H23 = NULL;
  PetscMatrixParallelComplex* S22 = NULL;
  PetscMatrixParallelComplex* S23 = NULL;

  //DOFmap subdomain_DOF2;
  const DOFmapInterface* pointer_to_subdomain_DOF2=NULL;
  DOFmapInterface* subdomain_coupling_DOF2=NULL;
  PropagationUtilities::transfer_matrix_get_Hamiltonian(this_simulation,std::string("2"),output_Propagator,neighbor_domain2, momentum, D22, H23,S22, S23, pointer_to_subdomain_DOF2, subdomain_coupling_DOF2);
  delete subdomain_coupling_DOF2;

  PetscMatrixParallelComplex* D33 = NULL;
  PetscMatrixParallelComplex* H34 = NULL;
  PetscMatrixParallelComplex* S33 = NULL;
  PetscMatrixParallelComplex* S34 = NULL;

  //DOFmap subdomain_DOF3;
  const DOFmapInterface* pointer_to_subdomain_DOF3=NULL;
  DOFmapInterface* subdomain_coupling_DOF3=NULL;

  PetscMatrixParallelComplex* Dtmp = NULL;
  PetscMatrixParallelComplex* Htmp = NULL;
  PetscMatrixParallelComplex* Stmp = NULL;
  PetscMatrixParallelComplex* SStmp = NULL;

  //DOFmap subdomain_DOF4;
  const DOFmapInterface* pointer_to_subdomain_DOF4=NULL;
  DOFmapInterface* subdomain_coupling_DOF4=NULL;

  if(new_self_energy) //for OMEN type QTBM
  {
    PropagationUtilities::transfer_matrix_get_Hamiltonian(this_simulation,std::string("3"),output_Propagator,neighbor_domain2, momentum, D33, Htmp,S33, Stmp, pointer_to_subdomain_DOF3, subdomain_coupling_DOF3);
    delete subdomain_coupling_DOF3;
    PropagationUtilities::transfer_matrix_get_Hamiltonian(this_simulation,std::string("4"),output_Propagator,neighbor_domain2, momentum, Dtmp, H34, SStmp, S34, pointer_to_subdomain_DOF4, subdomain_coupling_DOF4);
    delete subdomain_coupling_DOF4;
    delete Htmp;
  }
  else //for RGF/NEGF
    PropagationUtilities::transfer_matrix_get_Hamiltonian(this_simulation,std::string("3"),output_Propagator,neighbor_domain2, momentum, D33, H34, S33, S34, pointer_to_subdomain_DOF3, subdomain_coupling_DOF3);

  //Hermitian transpose of coupling matrix for subdomains
  PetscMatrixParallelComplex* H10 = new PetscMatrixParallelComplex(*H01);
  H10->hermitian_transpose_matrix(*H10,MAT_REUSE_MATRIX);
  PetscMatrixParallelComplex* H21 = new PetscMatrixParallelComplex(*H12);
  H21->hermitian_transpose_matrix(*H21,MAT_REUSE_MATRIX);
  PetscMatrixParallelComplex* H32 = new PetscMatrixParallelComplex(*H23);
  H32->hermitian_transpose_matrix(*H32,MAT_REUSE_MATRIX);
  PetscMatrixParallelComplex* H43 = new PetscMatrixParallelComplex(*H34);
  H43->hermitian_transpose_matrix(*H43,MAT_REUSE_MATRIX);

  //1b. obtain matrix information
  //global rows
  unsigned int number_of_00_rows = D00->get_num_rows();
  unsigned int number_of_00_cols = D00->get_num_cols();
  unsigned int number_of_11_rows = D11->get_num_rows();
  unsigned int number_of_11_cols = D11->get_num_cols();
  unsigned int number_of_22_rows = D22->get_num_rows();
  unsigned int number_of_22_cols = D22->get_num_cols();
  unsigned int number_of_33_rows = D33->get_num_rows();
  unsigned int number_of_33_cols = D33->get_num_cols();

  //local rows
  unsigned int number_of_00_local_rows = D00->get_num_owned_rows();
  unsigned int number_of_11_local_rows = D11->get_num_owned_rows();
  unsigned int number_of_22_local_rows = D22->get_num_owned_rows();
  unsigned int number_of_33_local_rows = D33->get_num_owned_rows();
  int start_own_00_rows;
  int end_own_00_rows_p1;
  D00->get_ownership_range(start_own_00_rows,end_own_00_rows_p1);
  int start_own_11_rows;
  int end_own_11_rows_p1;
  D11->get_ownership_range(start_own_11_rows,end_own_11_rows_p1);
  int start_own_22_rows;
  int end_own_22_rows_p1;
  D22->get_ownership_range(start_own_22_rows,end_own_22_rows_p1);
  int start_own_33_rows;
  int end_own_33_rows_p1;
  D33->get_ownership_range(start_own_33_rows,end_own_33_rows_p1);

  //get information of coupling Hamiltonian
  int start_own_01_rows;
  int end_own_01_rows_p1;
  H01->get_ownership_range(start_own_01_rows,end_own_01_rows_p1);
  int start_own_12_rows;
  int end_own_12_rows_p1;
  H12->get_ownership_range(start_own_12_rows,end_own_12_rows_p1);
  int start_own_23_rows;
  int end_own_23_rows_p1;
  H23->get_ownership_range(start_own_23_rows,end_own_23_rows_p1);
  int start_own_34_rows;
  int end_own_34_rows_p1;
  H34->get_ownership_range(start_own_34_rows,end_own_34_rows_p1);

  //1c. set local nonzeros and nonlocal nonzeros
  vector<int> D01_rows_diagonal(number_of_00_local_rows,0);
  vector<int> D01_rows_offdiagonal(number_of_00_local_rows,0);
  for(unsigned int i=0; i<number_of_00_local_rows; i++)
  {
    D01_rows_diagonal[i]=H01->get_nz_diagonal_by_global_index(i+start_own_01_rows);
    D01_rows_offdiagonal[i]=H01->get_nz_offdiagonal_by_global_index(i+start_own_01_rows);
  }
  vector<int> D12_rows_diagonal(number_of_11_local_rows,0);
  vector<int> D12_rows_offdiagonal(number_of_11_local_rows,0);
  for(unsigned int i=0; i<number_of_11_local_rows; i++)
  {
    D12_rows_diagonal[i]=H12->get_nz_diagonal_by_global_index(i+start_own_12_rows);
    D12_rows_offdiagonal[i]=H12->get_nz_offdiagonal_by_global_index(i+start_own_12_rows);
  }
  vector<int> D23_rows_diagonal(number_of_22_local_rows,0);
  vector<int> D23_rows_offdiagonal(number_of_22_local_rows,0);
  for(unsigned int i=0; i<number_of_22_local_rows; i++)
  {
    D23_rows_diagonal[i]=H23->get_nz_diagonal_by_global_index(i+start_own_23_rows);
    D23_rows_offdiagonal[i]=H23->get_nz_offdiagonal_by_global_index(i+start_own_23_rows);
  }
  vector<int> D34_rows_diagonal(number_of_33_local_rows,0);
  vector<int> D34_rows_offdiagonal(number_of_33_local_rows,0);
  for(unsigned int i=0; i<number_of_33_local_rows; i++)
  {
    D34_rows_diagonal[i]=H34->get_nz_diagonal_by_global_index(i+start_own_34_rows);
    D34_rows_offdiagonal[i]=H34->get_nz_offdiagonal_by_global_index(i+start_own_34_rows);
  }
  vector<int> D10_rows_diagonal(number_of_11_local_rows,0);
  vector<int> D10_rows_offdiagonal(number_of_11_local_rows,0);
  for(unsigned int i=0; i<number_of_11_local_rows; i++)
  {
    D10_rows_diagonal[i]=H10->get_nz_diagonal(i+number_of_00_local_rows);
    D10_rows_offdiagonal[i]=H10->get_nz_offdiagonal(i+number_of_00_local_rows);
  }
  vector<int> D21_rows_diagonal(number_of_22_local_rows,0);
  vector<int> D21_rows_offdiagonal(number_of_22_local_rows,0);
  for(unsigned int i=0; i<number_of_22_local_rows; i++)
  {
    D21_rows_diagonal[i]=H21->get_nz_diagonal(i+number_of_11_local_rows);
    D21_rows_offdiagonal[i]=H21->get_nz_offdiagonal(i+number_of_11_local_rows);
  }
  vector<int> D32_rows_diagonal(number_of_33_local_rows,0);
  vector<int> D32_rows_offdiagonal(number_of_33_local_rows,0);
  for(unsigned int i=0; i<number_of_33_local_rows; i++)
  {
    D32_rows_diagonal[i]=H32->get_nz_diagonal(i+number_of_22_local_rows);
    D32_rows_offdiagonal[i]=H32->get_nz_offdiagonal(i+number_of_22_local_rows);
  }
  vector<int> D43_rows_diagonal(number_of_00_local_rows,0);
  vector<int> D43_rows_offdiagonal(number_of_00_local_rows,0);
  for(unsigned int i=0; i<number_of_00_local_rows; i++)
  {
    D43_rows_diagonal[i]=H43->get_nz_diagonal(i+number_of_33_local_rows);
    D43_rows_offdiagonal[i]=H43->get_nz_offdiagonal(i+number_of_33_local_rows);
  }

  //1d. obtain the submatrix -- Dij is the upper right block of super-coupling matrix Hij
  PetscMatrixParallelComplex* D01 = NULL;
  PetscMatrixParallelComplex* D12 = NULL;
  PetscMatrixParallelComplex* D23 = NULL;
  PetscMatrixParallelComplex* D34 = NULL;
  transfer_matrix_get_submatrix(this_simulation,number_of_00_rows,number_of_11_cols,0,number_of_00_cols,D01_rows_diagonal,D01_rows_offdiagonal,H01,D01);
  transfer_matrix_get_submatrix(this_simulation,number_of_11_rows,number_of_22_cols,0,number_of_11_cols,D12_rows_diagonal,D12_rows_offdiagonal,H12,D12);
  transfer_matrix_get_submatrix(this_simulation,number_of_22_rows,number_of_33_cols,0,number_of_22_cols,D23_rows_diagonal,D23_rows_offdiagonal,H23,D23);
  transfer_matrix_get_submatrix(this_simulation,number_of_33_rows,number_of_00_cols,0,number_of_33_cols,D34_rows_diagonal,D34_rows_offdiagonal,H34,D34);

  PetscMatrixParallelComplex* D10 = NULL;
  PetscMatrixParallelComplex* D21 = NULL;
  PetscMatrixParallelComplex* D32 = NULL;
  PetscMatrixParallelComplex* D43 = NULL;
  transfer_matrix_get_submatrix(this_simulation,number_of_11_rows,number_of_00_cols,number_of_00_rows,0,D10_rows_diagonal,D10_rows_offdiagonal,H10,D10);
  transfer_matrix_get_submatrix(this_simulation,number_of_22_rows,number_of_11_cols,number_of_11_rows,0,D21_rows_diagonal,D21_rows_offdiagonal,H21,D21);
  transfer_matrix_get_submatrix(this_simulation,number_of_33_rows,number_of_22_cols,number_of_22_rows,0,D32_rows_diagonal,D32_rows_offdiagonal,H32,D32);
  transfer_matrix_get_submatrix(this_simulation,number_of_00_rows,number_of_33_cols,number_of_33_rows,0,D43_rows_diagonal,D43_rows_offdiagonal,H43,D43);
  delete H01;
  delete H12;
  delete H23;
  delete H34;
  delete H10;
  delete H21;
  delete H32;
  delete H43;

  //2. get the full Hamiltonian for the lead domain
  //E-H=|D00 D01 0     0|; T=|0 0 0 D43|
  //    |D10 D11 D12   0|    |0 0 0 0  |
  //    |0   D21 D22 D23|    |0 0 0 0  |
  //    |0   0   D32 D33|    |0 0 0 0  |
  std::complex<double> energy;
  transfer_matrix_get_energy(this_simulation,output_Propagator, momentum, &energy);
  double eta=options.get_option("constant_lead_eta",1e-12);//small i*eta adding to energy for van Hove singularity
  //Kai Miao:differentiate the fermion and boson in the transfer matrix method.
  if((output_Propagator->get_name()).find("Fermion")!=std::string::npos)
  {
    energy+=cplx(0.0,1.0)*eta;
    *D00 *=std::complex<double>(-1.0,0.0); //Dii=E-Hii
    D00->matrix_diagonal_shift(energy);
    *D11 *=std::complex<double>(-1.0,0.0);
    D11->matrix_diagonal_shift(energy);
    *D22 *=std::complex<double>(-1.0,0.0);
    D22->matrix_diagonal_shift(energy);
    *D33 *=std::complex<double>(-1.0,0.0);
    D33->matrix_diagonal_shift(energy);
  }
  else if ((output_Propagator->get_name()).find("Boson")!=std::string::npos)
  {
    energy=energy*energy+cplx(0.0,1.0)*eta;
    *D00 *=std::complex<double>(-1.0,0.0);
    D00->matrix_diagonal_shift(energy);
    *D11 *=std::complex<double>(-1.0,0.0);
    D11->matrix_diagonal_shift(energy);
    *D22 *=std::complex<double>(-1.0,0.0);
    D22->matrix_diagonal_shift(energy);
    *D33 *=std::complex<double>(-1.0,0.0);
    D33->matrix_diagonal_shift(energy);
  }
  *D01 *=std::complex<double>(-1.0,0.0); //Dij=-Hij
  *D10 *=std::complex<double>(-1.0,0.0);
  *D12 *=std::complex<double>(-1.0,0.0);
  *D21 *=std::complex<double>(-1.0,0.0);
  *D23 *=std::complex<double>(-1.0,0.0);
  *D32 *=std::complex<double>(-1.0,0.0);
  *D34 *=std::complex<double>(-1.0,0.0);
  *D43 *=std::complex<double>(-1.0,0.0);

  PetscMatrixParallelComplex* E_minus_H_matrix = NULL; //E-H matrix
  PetscMatrixParallelComplex* T43 = NULL; //T matrix
  PetscMatrixParallelComplex* T34 = NULL; //T' matrix

  //construct the E-H matrix for the contact domain
  unsigned int number_of_EmH_rows = number_of_00_rows+number_of_11_rows+number_of_22_rows+number_of_33_rows;
  unsigned int number_of_EmH_cols = number_of_00_cols+number_of_11_cols+number_of_22_cols+number_of_33_cols;
  unsigned int number_of_EmH_local_rows = number_of_00_local_rows+number_of_11_local_rows+number_of_22_local_rows+number_of_33_local_rows;
  unsigned int offset_rows = start_own_00_rows+start_own_11_rows+start_own_22_rows+start_own_33_rows;

  //-------------------------------------------------------
  vector<int> EmH_rows_diagonal(number_of_EmH_local_rows,0);
  vector<int> EmH_rows_offdiagonal(number_of_EmH_local_rows,0);
  for(unsigned int i=0; i<number_of_00_local_rows; i++)
  {
    EmH_rows_diagonal[i]=D00->get_nz_diagonal(i)+D01->get_nz_diagonal(i);
    EmH_rows_offdiagonal[i]=D00->get_nz_offdiagonal(i)+D01->get_nz_offdiagonal(i);
  }
  for(unsigned int i=0; i<number_of_11_local_rows; i++)
  {
    EmH_rows_diagonal[i+number_of_00_local_rows]=D11->get_nz_diagonal(i)+
        D12->get_nz_diagonal(i)+D10->get_nz_diagonal(i);
    EmH_rows_offdiagonal[i+number_of_00_local_rows]=D11->get_nz_offdiagonal(i)+
        D12->get_nz_offdiagonal(i)+D10->get_nz_offdiagonal(i);
  }
  for(unsigned int i=0; i<number_of_22_local_rows; i++)
  {
    EmH_rows_diagonal[i+number_of_00_local_rows+number_of_11_local_rows]=D22->get_nz_diagonal(i)+
        D23->get_nz_diagonal(i)+D21->get_nz_diagonal(i);
    EmH_rows_offdiagonal[i+number_of_00_local_rows+number_of_11_local_rows]=D22->get_nz_offdiagonal(i)+
        D23->get_nz_offdiagonal(i)+D21->get_nz_offdiagonal(i);
  }
  for(unsigned int i=0; i<number_of_33_local_rows; i++)
  {
    EmH_rows_diagonal[i+number_of_EmH_local_rows-number_of_33_local_rows]=D33->get_nz_diagonal(i)+
        D32->get_nz_diagonal(i);
    EmH_rows_offdiagonal[i+number_of_EmH_local_rows-number_of_33_local_rows]=D33->get_nz_offdiagonal(i)+
        D32->get_nz_offdiagonal(i);
  }

  vector<int> T43_rows_diagonal(number_of_EmH_local_rows,0);
  vector<int> T43_rows_offdiagonal(number_of_EmH_local_rows,0);
  for(unsigned int i=0; i<number_of_00_local_rows; i++)
  {
    T43_rows_diagonal[i]=D43->get_nz_diagonal(i);
    T43_rows_offdiagonal[i]=D43->get_nz_offdiagonal(i);
  }

  //set the sparsity pattern
  transfer_matrix_initialize_temp_matrix(this_simulation,number_of_EmH_rows,number_of_EmH_cols,EmH_rows_diagonal,EmH_rows_offdiagonal,E_minus_H_matrix);

  //fill E-H elements from the sub blocks 0~3
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows,0,0,D00,E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows,number_of_00_cols,0,D01,E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_00_local_rows,0,0,D10,E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_00_local_rows,number_of_00_cols,0,D11,E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_00_local_rows,number_of_00_cols+number_of_11_cols,0,D12,E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_00_local_rows+number_of_11_local_rows,number_of_00_cols,0,D21,E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_00_local_rows+number_of_11_local_rows,number_of_00_cols+number_of_11_cols,0,D22,
                                      E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_00_local_rows+number_of_11_local_rows,number_of_EmH_cols-number_of_33_cols,0,D23,
                                      E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,number_of_00_cols+number_of_11_cols,0,D32,
                                      E_minus_H_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,number_of_EmH_cols-number_of_33_cols,0,D33,
                                      E_minus_H_matrix);
  E_minus_H_matrix->assemble();

  //fill the elements for T matrix
  transfer_matrix_initialize_temp_matrix(this_simulation,number_of_EmH_rows,number_of_EmH_cols,T43_rows_diagonal,T43_rows_offdiagonal,T43);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows,number_of_EmH_cols-number_of_33_cols,0,D43,T43);
  T43->assemble();
  T34 = new PetscMatrixParallelComplex(*T43);
  T34->hermitian_transpose_matrix(*T34,MAT_REUSE_MATRIX);

  bool LRA=options.get_option("LRA", bool(false)); //LRA in contact used or not
  double number_of_set_num_ratio=options.get_option("ratio_of_eigenvalues_to_be_solved",double(1.0)); //LRA ratio
  double shift=-0.5; //real part of eigenvalues =-0.5eV is for the propagating modes

  unsigned int number_of_eigenvalues; //number of eigenvalues that are solved and stored
  unsigned int number_of_vectors_size; //dimension of contact modes
  unsigned int number_of_wave_left; //number of out of device modes
  unsigned int number_of_wave_right; //number of into device modes

  //Yu: from step 3 to step 5 we customize the code for real-type matrix and complex-type matrix
  bool wireNonSO=options.get_option("wire_nonSO_basis", bool(false)); //wire without spin orbit coupling in the tb model, matrix is real
  if(wireNonSO)
  {
    //if matrix is real
    //3. derivation of eigenvalue problem
    //HP=|D00 D01 0   D43|; P=|0 0 0   D43|; M=inv(HP)*P
    //   |D10 D11 D12   0|    |0 0 0   0  |
    //   |0   D21 D22   0|    |0 0 0   0  |
    //   |D34 0   D32 D33|    |0 0 D32 D33|
    PetscMatrixParallel<double>* M1_matrix = NULL; //upper right block of M
    PetscMatrixParallel<double>* M2_matrix = NULL; //lower right block of M

    //obtain M matrix for eigenproblem
    transfer_matrix_get_M_matrix_double(this_simulation,LRA,D00,D11,D22,D33,D32,D43,E_minus_H_matrix,T34,T43,M1_matrix,M2_matrix);
    delete D01;
    delete D10;
    delete D12;
    delete D21;
    delete D23;
    delete D34;
    delete D32;
    delete D43;
    //Hack to avoid memory leaks for Phonon transport"
    //reason: StrainVFF solver creates the "Hamiltonian", i.e. D11, D22 etc under the assumption
    //that the get_data calling solver deletes it afterwards.
    if(Hamilton_Constructor->get_type()=="VFFStrain")
    {
      delete D11;
      D11=NULL;
      delete D22;
      D22=NULL;
      delete D33;
      D33=NULL;
    }

    //4. solve the eigenvalue problem (or part of it if use LRA)
    int number_of_set_num=M2_matrix->get_num_rows();
    number_of_set_num*=number_of_set_num_ratio;
    vector<std::complex<double> > M_values; //eigenvalues
    vector< PetscVectorNemo<double> > M_vectors_real; //real part of eigenvector
    vector< PetscVectorNemo<double> > M_vectors_imag; //imag part of eigenvector

    transfer_matrix_solve_eigenvalues_double(this_simulation,M2_matrix,LRA,shift,number_of_set_num,M_values,
        M_vectors_real,M_vectors_imag,&number_of_eigenvalues,&number_of_vectors_size);

    //5. find out which waves are needed
    transfer_matrix_get_wave_direction_double(this_simulation,M1_matrix,T43,M_values,M_vectors_real,M_vectors_imag,&number_of_eigenvalues,
        &number_of_vectors_size,&number_of_wave_left,&number_of_wave_right);
    delete M1_matrix;
    delete M2_matrix;
  }
  else
  {
    //if matrix is complex
    //3. derivation of eigenvalue problem
    //HP=|D00 D01 0   D43|; P=|0 0 0   D43|; M=inv(HP)*P
    //   |D10 D11 D12   0|    |0 0 0   0  |
    //   |0   D21 D22   0|    |0 0 0   0  |
    //   |D34 0   D32 D33|    |0 0 D32 D33|
    PetscMatrixParallelComplex* M1_matrix = NULL; //upper right block of M
    PetscMatrixParallelComplex* M2_matrix = NULL; //lower right block of M

    //obtain M matrix for eigenproblem
    transfer_matrix_get_M_matrix(this_simulation,LRA,D00,D11,D22,D33,D32,D43,E_minus_H_matrix,T34,T43,M1_matrix,M2_matrix);
    delete D01;
    delete D10;
    delete D12;
    delete D21;
    delete D23;
    delete D34;
    delete D32;
    delete D43;

    //4. solve the eigenvalue problem (or part of it)
    int number_of_set_num=M2_matrix->get_num_rows();
    number_of_set_num*=number_of_set_num_ratio;
    vector<std::complex<double> > M_values; //eigenvalues
    vector< vector< std::complex<double> > > M_vectors_temp; //eigenvectors

    transfer_matrix_solve_eigenvalues(this_simulation,M2_matrix,LRA,shift,number_of_set_num,M_values,
                                      M_vectors_temp,&number_of_eigenvalues,&number_of_vectors_size);

    //5. find out which waves are needed
    transfer_matrix_get_wave_direction(this_simulation,M1_matrix,T43,M_values,M_vectors_temp,&number_of_eigenvalues,
                                       &number_of_vectors_size,&number_of_wave_left,&number_of_wave_right);
    delete M1_matrix;
    delete M2_matrix;
  }

  NemoPhys::transfer_matrix_type temp_transfer_type=NemoPhys::into_device_propagating_modes;
  PetscMatrixParallelComplex* into_device_propagating_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_modes;
  PetscMatrixParallelComplex* into_device_decaying_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_modes;
  PetscMatrixParallelComplex* out_of_device_propagating_modes=NULL;
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_modes;
  PetscMatrixParallelComplex* out_of_device_decaying_modes=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_phase;
  PetscMatrixParallelComplex* into_device_propagating_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_phase;
  PetscMatrixParallelComplex* into_device_decaying_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_phase;
  PetscMatrixParallelComplex* out_of_device_propagating_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_phase;
  PetscMatrixParallelComplex* out_of_device_decaying_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_modes;
  PetscMatrixParallelComplex* into_device_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_phase;
  PetscMatrixParallelComplex* into_device_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_modes;
  PetscMatrixParallelComplex* out_of_device_modes=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_phase;
  PetscMatrixParallelComplex* out_of_device_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_velocity;
  PetscMatrixParallelComplex* into_device_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_velocity;
  PetscMatrixParallelComplex* into_device_propagating_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_velocity;
  PetscMatrixParallelComplex* out_of_device_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_velocity;
  PetscMatrixParallelComplex* out_of_device_propagating_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_velocity,&temp_transfer_type);

  //6. get the self energy
  if(new_self_energy) //OMEN-type QTBM, self-energy with the same size as device domain
  { 
    //std::set<unsigned int> Hamilton_momentum_indices;
    //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
    //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
    //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
    //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=momentum[*(pointer_to_Hamilton_momentum_indices->begin())];

    // get the Hamilton constructor of the neighboring domain
    Simulation* neighbor_HamiltonConstructor =  NULL;
    std::string neighbor_Hamilton_constructor_name;
    std::string variable_name="Hamilton_constructor_"+neighbor_domain->get_name();
    InputOptions& opt = Hamilton_Constructor->get_reference_to_options();
    if(opt.check_option(variable_name))
      neighbor_Hamilton_constructor_name=opt.get_option(variable_name,std::string(""));
    else
      throw std::invalid_argument("Schroedinger("+this_simulation->get_name()+")::assemble_hamiltonian: define \""+variable_name+"\"\n");
    neighbor_HamiltonConstructor = this_simulation->find_simulation(neighbor_Hamilton_constructor_name);
    NEMO_ASSERT(neighbor_HamiltonConstructor!=NULL,
                "Schroedinger("+this_simulation->get_name()+")::assemble_hamiltonian: Simulation \""+neighbor_Hamilton_constructor_name+"\" has not been found!\n");
    PetscMatrixParallelComplex* coupling_Hamiltonian_lead=NULL;
    Domain* neighbor_domain_lead = this_simulation->get_simulation_domain();


    PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;
    DOFmapInterface& neighbor_domain_DOFmap = neighbor_HamiltonConstructor->get_dof_map(neighbor_domain);
    DOFmapInterface* coupling_DOFmap=&neighbor_domain_DOFmap;
    DOFmapInterface* temp_pointer=coupling_DOFmap;

    const DOFmapInterface& device_DOFmap = Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain());
    NemoUtils::tic(tic_toc_prefix + "obtain Hamiltonian");
    std::vector<NemoMeshPoint> sorted_momentum;
    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,PropInterface->get_momentum_mesh_types(),Hamilton_Constructor);
    Hamilton_Constructor->get_data(std::string("Hamiltonian"),sorted_momentum, neighbor_domain, coupling_Hamiltonian, coupling_DOFmap);
    if(coupling_DOFmap!=temp_pointer)
      delete coupling_DOFmap;
    NemoUtils::toc(tic_toc_prefix + "obtain Hamiltonian");







    NemoUtils::tic(tic_toc_prefix + "coupling_Hamiltonian_get_ownership_range");
    //construct the domain3=domain1+domain2 matrices
    unsigned int number_of_super_rows = coupling_Hamiltonian->get_num_rows();
    unsigned int number_of_super_cols = coupling_Hamiltonian->get_num_cols();
    int start_own_super_rows;
    int end_own_super_rows_p1;
    coupling_Hamiltonian->get_ownership_range(start_own_super_rows,end_own_super_rows_p1);
    NemoUtils::toc(tic_toc_prefix + "coupling_Hamiltonian_get_ownership_range");

    //these are the dimensions of the device matrices:
    unsigned int number_of_rows2 = device_DOFmap.get_global_dof_number();
    unsigned int number_of_cols2 = number_of_rows2;
    unsigned int local_number_of_rows2 = device_DOFmap.get_number_of_dofs();

    //these are the dimensions of the lead matrices:
    unsigned int number_of_rows1      = number_of_super_rows-number_of_rows2;
    unsigned int number_of_cols1      = number_of_super_cols-number_of_cols2;

    if(opt.check_option("atom_order_from") || number_of_rows1>number_of_rows2)
    {
      //if smartdof used in device domain or device size < lead size (e.g RGF)
      //----------------------------------------------------------
      //Method I. get self energy that is defined on device domain
      //Yu: this maybe slow since the matrix is of device size
      // get the submatrix (upper right block) from coupling Hamiltonian
      std::vector<int> temp_rows(local_number_of_rows2);
      std::vector<int> temp_cols(number_of_cols1);
      for(unsigned int i=0; i<local_number_of_rows2 && i<number_of_rows2; i++)
        temp_rows[i]=i;
      for(unsigned int i=0; i<number_of_cols1; i++)
        temp_cols[i]=i+number_of_cols2;

      PetscMatrixParallelComplex* coupling = NULL;
      coupling= new PetscMatrixParallelComplex(number_of_rows2,number_of_cols1,this_simulation->get_simulation_domain()->get_communicator());
      coupling->set_num_owned_rows(local_number_of_rows2);
      vector<int> rows_diagonal(local_number_of_rows2,0);
      vector<int> rows_offdiagonal(local_number_of_rows2,0);
      for(unsigned int i=0; i<local_number_of_rows2; i++)
      {
        rows_diagonal[i]=coupling_Hamiltonian->get_nz_diagonal(i);
        rows_offdiagonal[i]=coupling_Hamiltonian->get_nz_offdiagonal(i);
      }
      for(unsigned int i=0; i<local_number_of_rows2; i++)
        coupling->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
      coupling_Hamiltonian->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,coupling);
      coupling->assemble();

      transfer_matrix_get_new_self_energy(this_simulation,E_minus_H_matrix,T43,&number_of_wave_left,&number_of_vectors_size,
                                          out_of_device_phase,out_of_device_modes,result,coupling);
      delete coupling;
      //----------------------------------------------------------
      delete coupling_Hamiltonian;
      delete coupling_Hamiltonian_lead;
    }
    else
    {
      //Bozidar (to set multivalley effective mass Hamiltonian)
      //set_valley(writeable_Propagators.begin()->second, momentum, neighbor_HamiltonConstructor);
      sorted_momentum.clear();
      QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,PropInterface->get_momentum_mesh_types(),neighbor_HamiltonConstructor);
      Simulation* temp_simulation=this_simulation->find_simulation_on_domain(neighbor_domain_lead,Hamilton_Constructor->get_type());
      //NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found Hamilton_Constructor for neighbor_domain_lead\n");
      if (temp_simulation == NULL && (Hamilton_Constructor->get_type().find("module") != std::string::npos || Hamilton_Constructor->get_type().find("Module") != std::string::npos))
        temp_simulation = Hamilton_Constructor;
      NEMO_ASSERT(temp_simulation != NULL, tic_toc_prefix + "have not found Hamilton_Constructor for neighbor_domain\n");
      DOFmapInterface& temp_DOFmap=temp_simulation->get_dof_map(neighbor_domain_lead);
      DOFmapInterface* coupling_DOFmap_lead=&temp_DOFmap;
      DOFmapInterface* temp_pointer=coupling_DOFmap_lead;
      neighbor_HamiltonConstructor->get_data(std::string("Hamiltonian"),sorted_momentum, neighbor_domain_lead, coupling_Hamiltonian_lead,
                                             coupling_DOFmap_lead);
      //----------------------------------------------------------
      //Method II. get self energy that is defined on contact domain, then translate into device
      //Yu: this maybe faster than I, however requires a translation from contact into device
      //1. get self energy on contact -- only upper left block is full
      PetscMatrixParallelComplex* result_temp = NULL;
      transfer_matrix_get_self_energy(this_simulation,E_minus_H_matrix,T43,&number_of_wave_left,&number_of_vectors_size,
                                      out_of_device_phase,out_of_device_modes,result_temp,number_of_00_rows,number_of_00_cols);

      //2. translate self energy into the correct atomic order
      NemoUtils::tic(tic_toc_prefix + " translate self energy into device domain");
      result = new PetscMatrixParallelComplex(number_of_rows2,number_of_cols2,this_simulation->get_simulation_domain()->get_communicator() );
      result->set_num_owned_rows(local_number_of_rows2);
      std::map<unsigned int, unsigned int> self_subindex_map;
      coupling_DOFmap_lead->get_sub_DOF_index_map(pointer_to_subdomain_DOF0,self_subindex_map);
      if(coupling_DOFmap_lead!=temp_pointer)
      delete coupling_DOFmap_lead;
      delete coupling_Hamiltonian_lead;
      std::map<unsigned int, unsigned int>::iterator temp_it1=self_subindex_map.begin();

      //figure out the nonzero rows of coupling
      int row_index = 0;
      for(unsigned int i=0; i<local_number_of_rows2; i++)
      {
        row_index = coupling_Hamiltonian->get_nz_diagonal(i);
        if(row_index>0)
        {
          row_index = i;
          break;
        }
      }
      delete coupling_Hamiltonian;
      //which block is this nonzero rows located?
      int contact_index=0;
      int start_point = 0;
      unsigned int number_of_rows_to_set=0;
      if(row_index!=0)
      {
        contact_index = number_of_rows2/row_index;
      }
      start_point = number_of_rows2-contact_index*number_of_rows1;
      number_of_rows_to_set = number_of_rows1;
      if(start_point<0 || row_index==0)
        start_point = 0;

      //set nonzero pattern
      for(unsigned int i=0; i<number_of_rows2; i++)
        result->set_num_nonzeros(i,0,0);
      for(unsigned int i=0; i<number_of_rows_to_set; i++)
      {
        temp_it1=self_subindex_map.find(i);
        if(temp_it1!=self_subindex_map.end())
        {
          unsigned int j = temp_it1->second;//translated matrix index
          if(j>number_of_00_rows)
            result->set_num_nonzeros(i+start_point,0,0);
          else
            result->set_num_nonzeros(i+start_point,result_temp->get_nz_diagonal(j),result_temp->get_nz_offdiagonal(j));
        }
        else
          result->set_num_nonzeros(i+start_point,0,0);
      }
      result->allocate_memory();
      result->set_to_zero();

      //set matrix elements
      const std::complex<double>* pointer_to_data= NULL;
      vector<cplx> data_vector;
      vector<int> col_index;
      int n_nonzeros=0;
      const int* n_col_nums=NULL;
      temp_it1=self_subindex_map.begin();
      for(; temp_it1!=self_subindex_map.end(); temp_it1++)
      {
        unsigned int i = temp_it1->first;
        unsigned int j = temp_it1->second;//translated row index
        unsigned int nonzero_cols = result_temp->get_nz_diagonal(j);
        if(j<=number_of_00_rows && nonzero_cols>0)
        {
          result_temp->get_row(j,&n_nonzeros,n_col_nums,pointer_to_data);
          col_index.resize(n_nonzeros,0);
          data_vector.resize(n_nonzeros,cplx(0.0,0.0));
          std::map<unsigned int, unsigned int>::iterator temp_it2=self_subindex_map.begin();
          for(int jj=0; jj<n_nonzeros; jj++)
          {
            data_vector[jj]=pointer_to_data[jj];
            col_index[jj]=temp_it2->first+start_point;
            temp_it2++;
          }
          result->set(i+start_point,col_index,data_vector);
          result_temp->store_row(j,&n_nonzeros,n_col_nums,pointer_to_data);
        }
      }
      result->assemble();
      NemoUtils::toc(tic_toc_prefix + " translate self energy into device domain");
      delete result_temp;
      //----------------------------------------------------------
    }
  }
  else
  {
    transfer_matrix_get_self_energy(this_simulation,E_minus_H_matrix,T43,&number_of_wave_left,&number_of_vectors_size,
                                    out_of_device_phase,out_of_device_modes,result,number_of_00_rows,number_of_00_cols);
  }
  bool output_cplxEk=options.get_option("output_complexEk", bool(false));
  if(output_cplxEk&&!PropOptionInterface->get_no_file_output()) //complex band structure output
  {
    vector<cplx> kdelta(out_of_device_phase->get_num_rows(),0.0);
    for(unsigned int i=0; i<out_of_device_phase->get_num_rows(); i++)
    {
      cplx ktmp=cplx(0.0,1.0)*log(out_of_device_phase->get(i,i));
      kdelta[i]=ktmp;
    }
    std::ostringstream strs_e;
    strs_e << energy.real();
    std::string str = strs_e.str();
    std::string Ekname="cplxEk.dat";
    const char* filename = Ekname.c_str();
    ofstream fEk;
    fEk.open(filename,ios_base::app);
    fEk<< out_of_device_phase->get_num_rows() << " ";
    fEk<< energy.real() << " ";
    for(unsigned int i=0; i<out_of_device_phase->get_num_rows(); i++)
      fEk<<kdelta[i].real()<<" "<<kdelta[i].imag()<<" ";
    fEk<<"\n";
    fEk.close();
  }

  bool output_Ek=options.get_option("output_Ek", bool(false));
  if(output_Ek&&!PropOptionInterface->get_no_file_output()) //normal band structure output
  {
    vector<double> kdelta(out_of_device_propagating_phase->get_num_rows(),0.0);
    for(unsigned int i=0; i<out_of_device_propagating_phase->get_num_rows(); i++)
    {
      cplx ktmp=cplx(0.0,1.0)*log(out_of_device_propagating_phase->get(i,i));
      kdelta[i]=ktmp.real();
    }
    std::stable_sort(kdelta.begin(),kdelta.end());
    std::ostringstream strs_e;
    strs_e << energy.real();
    std::string str = strs_e.str();
    std::string Ekname="Ek.dat";
    const char* filename = Ekname.c_str();
    ofstream fEk;
    fEk.open(filename,ios_base::app);
    fEk<< energy.real() << " ";
    for(unsigned int i=0; i<out_of_device_propagating_phase->get_num_rows(); i++)
      fEk<<kdelta[i]<<" ";
    fEk<<"\n";
    fEk.close();

    vector<double> kdeltam(into_device_propagating_phase->get_num_rows(),0.0);
    for(unsigned int i=0; i<into_device_propagating_phase->get_num_rows(); i++)
    {
      cplx ktmp=cplx(0.0,1.0)*log(into_device_propagating_phase->get(i,i));
      kdeltam[i]=ktmp.real();
    }
    std::stable_sort(kdeltam.begin(),kdeltam.end());
    std::ostringstream strs_em;
    strs_em << energy.real();
    std::string strm = strs_em.str();
    std::string Eknamem="Ekm.dat";
    const char* filenamem = Eknamem.c_str();
    ofstream fEkm;
    fEkm.open(filenamem,ios_base::app);
    fEkm<< energy.real() << " ";
    for(unsigned int i=0; i<into_device_propagating_phase->get_num_rows(); i++)
      fEkm<<kdeltam[i]<<" ";
    fEkm<<"\n";
    fEkm.close();

    //std::set<unsigned int> Hamilton_momentum_indices;
    //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
    //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
    //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
    //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=momentum[*(pointer_to_Hamilton_momentum_indices->begin())];
    Simulation* neighbor_HamiltonConstructor =  NULL;
    std::string neighbor_Hamilton_constructor_name;
    std::string variable_name="Hamilton_constructor_"+neighbor_domain->get_name();
    InputOptions& opt = Hamilton_Constructor->get_reference_to_options();
    if(opt.check_option(variable_name))
      neighbor_Hamilton_constructor_name=opt.get_option(variable_name,std::string(""));
    else
      throw std::invalid_argument("Schroedinger("+this_simulation->get_name()+")::assemble_hamiltonian: define \""+variable_name+"\"\n");
    neighbor_HamiltonConstructor = this_simulation->find_simulation(neighbor_Hamilton_constructor_name);
    PetscMatrixParallelComplex* contact_H=NULL;

    //Bozidar (to set multivalley effective mass Hamiltonian)
    //set_valley(writeable_Propagators.begin()->second, momentum, neighbor_HamiltonConstructor);
    std::vector<NemoMeshPoint> sorted_momentum;
    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,PropInterface->get_momentum_mesh_types(),neighbor_HamiltonConstructor);
    neighbor_HamiltonConstructor->get_data(std::string("Hamiltonian"),contact_H,sorted_momentum);

    contact_H->save_to_matlab_file(this_simulation->get_name()+"contact_Hamiltonian.m");
    E_minus_H_matrix->save_to_matlab_file(this_simulation->get_name()+"EmH.m");
  }
  delete T43;
  delete T34;
  delete E_minus_H_matrix;

  if(PropOptionInterface->get_avoid_copying_hamiltonian())
  {
    delete D00;
    delete D11;
    delete D22;
    delete D33;
    if(new_self_energy)
      delete Dtmp;
  }

  temp_transfer_type=NemoPhys::into_device_propagating_modes;
  PropInterface->set_transfer_matrix_element(into_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_modes;
  PropInterface->set_transfer_matrix_element(into_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_phase;
  PropInterface->set_transfer_matrix_element(into_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_phase;
  PropInterface->set_transfer_matrix_element(into_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_modes;
  PropInterface->set_transfer_matrix_element(into_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_phase;
  PropInterface->set_transfer_matrix_element(into_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_velocity;
  PropInterface->set_transfer_matrix_element(into_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_velocity;
  PropInterface->set_transfer_matrix_element(into_device_propagating_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_velocity;
  PropInterface->set_transfer_matrix_element(out_of_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_velocity;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_velocity,&temp_transfer_type);

//  delete subdomain_DOF0;
//  delete subdomain_DOF1;
//  delete subdomain_DOF2;
//  delete subdomain_DOF3;
//  delete subdomain_DOF4;
  
  NemoUtils::toc(time_for_transfer_matrix_leads);

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_batch_reorder_matrix(Simulation* this_simulation,unsigned int rows0, unsigned int rows1, unsigned int rows2, unsigned int rows3,
    unsigned int cols0, unsigned int cols1, unsigned int cols2, unsigned int cols3,
    bool flag, const DOFmapInterface& fullDOF, const DOFmapInterface*& subdomain_DOF0, const DOFmapInterface*& subdomain_DOF1,
    const DOFmapInterface*& subdomain_DOF2, const DOFmapInterface*& subdomain_DOF3,
    PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallelComplex*& result_matrix)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_batch_reorder_matrix ");
  NemoUtils::tic(tic_toc_prefix);

  transfer_matrix_reorder_matrix(this_simulation,rows0,cols0,flag,fullDOF,subdomain_DOF0,subdomain_DOF0,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows0,cols1,flag,fullDOF,subdomain_DOF0,subdomain_DOF1,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows0,cols2,flag,fullDOF,subdomain_DOF0,subdomain_DOF2,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows0,cols3,flag,fullDOF,subdomain_DOF0,subdomain_DOF3,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows1,cols0,flag,fullDOF,subdomain_DOF1,subdomain_DOF0,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows1,cols1,flag,fullDOF,subdomain_DOF1,subdomain_DOF1,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows1,cols2,flag,fullDOF,subdomain_DOF1,subdomain_DOF2,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows1,cols3,flag,fullDOF,subdomain_DOF1,subdomain_DOF3,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows2,cols0,flag,fullDOF,subdomain_DOF2,subdomain_DOF0,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows2,cols1,flag,fullDOF,subdomain_DOF2,subdomain_DOF1,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows2,cols2,flag,fullDOF,subdomain_DOF2,subdomain_DOF2,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows2,cols3,flag,fullDOF,subdomain_DOF2,subdomain_DOF3,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows3,cols0,flag,fullDOF,subdomain_DOF3,subdomain_DOF0,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows3,cols1,flag,fullDOF,subdomain_DOF3,subdomain_DOF1,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows3,cols2,flag,fullDOF,subdomain_DOF3,subdomain_DOF2,source_matrix,result_matrix);
  transfer_matrix_reorder_matrix(this_simulation,rows3,cols3,flag,fullDOF,subdomain_DOF3,subdomain_DOF3,source_matrix,result_matrix);
  result_matrix->assemble();
  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::transfer_matrix_initialize_reorder_matrix(Simulation* this_simulation,unsigned int offset_row, bool flag, const DOFmapInterface& fullDOF, const DOFmapInterface*& subdomain_DOF0,
    PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallelComplex*& result_matrix)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_initialize_reorder_matrix ");
  NemoUtils::tic(tic_toc_prefix);
  int start_own_source_rows;
  int end_own_source_rows_p1;
  source_matrix->get_ownership_range(start_own_source_rows,end_own_source_rows_p1);
  std::map<unsigned int, unsigned int> DOF0_subindex_map;
  fullDOF.get_sub_DOF_index_map(subdomain_DOF0,DOF0_subindex_map,flag);
  std::map<unsigned int, unsigned int>::iterator temp_it1=DOF0_subindex_map.begin();
  for(; temp_it1!=DOF0_subindex_map.end(); temp_it1++)
  {
    int i = temp_it1->first;
    int j = temp_it1->second;//translated row index
    if(j<start_own_source_rows || j>end_own_source_rows_p1)
      result_matrix->set_num_nonzeros(i,0,0);
    else
      result_matrix->set_num_nonzeros(i,source_matrix->get_nz_diagonal(j+offset_row),source_matrix->get_nz_offdiagonal(j+offset_row));
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_reorder_matrix(Simulation* this_simulation,unsigned int offset_row,
    unsigned int offset_col,
    bool flag, const DOFmapInterface& fullDOF, const DOFmapInterface*& subdomain_DOF0,
    const DOFmapInterface*& subdomain_DOF1,PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallelComplex*& result_matrix)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_reorder_matrix ");
  NemoUtils::tic(tic_toc_prefix);
  int start_own_source_rows;
  int end_own_source_rows_p1;
  source_matrix->get_ownership_range(start_own_source_rows,end_own_source_rows_p1);
  std::map<unsigned int, unsigned int> DOF0_subindex_map;
  std::map<unsigned int, unsigned int> DOF1_subindex_map;
  fullDOF.get_sub_DOF_index_map(subdomain_DOF0,DOF0_subindex_map,flag);
  fullDOF.get_sub_DOF_index_map(subdomain_DOF1,DOF1_subindex_map,flag);
  std::map<unsigned int, unsigned int>::iterator temp_it1=DOF0_subindex_map.begin();
  for(; temp_it1!=DOF0_subindex_map.end(); temp_it1++)
  {
    int i = temp_it1->first;
    int j = temp_it1->second;//translated row index
    if(j>=start_own_source_rows && j<end_own_source_rows_p1)
    {
      std::map<unsigned int, unsigned int>::iterator temp_it2=DOF1_subindex_map.begin();
      for(; temp_it2!=DOF1_subindex_map.end(); temp_it2++)
      {
        unsigned int ii = temp_it2->first;
        unsigned int jj = temp_it2->second;//translated column index
        if(jj<source_matrix->get_num_cols())
        {
          cplx temp_val=source_matrix->get(j+offset_row,jj+offset_col);
          if(abs(temp_val)>1e-8)
            result_matrix->set(i,ii,temp_val);
        }
      }
    }
  }
  result_matrix->assemble();
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_get_Green_function(Simulation* this_simulation,PetscMatrixParallelComplex*& E_minus_H_matrix, PetscMatrixParallelComplex*& T_matrix,
    unsigned int* number_of_wave, unsigned int* number_of_vectors_size, PetscMatrixParallelComplex*& phase_factor,
    PetscMatrixParallelComplex*& Wave_matrix, PetscMatrixParallelComplex*& result_matrix)
{
  //calculate surface Green's function gr with given modes
  //Note: to explicitly solve gr is very inefficient
  //only used for non-OMEN QTBM
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_Green_function ");
  NemoUtils::tic(tic_toc_prefix);
  const InputOptions& options=this_simulation->get_options();
  if(*number_of_wave==0)
  {
    result_matrix = new PetscMatrixParallelComplex(*number_of_vectors_size,*number_of_vectors_size,
        this_simulation->get_simulation_domain()->get_communicator());
    result_matrix->set_num_owned_rows(*number_of_vectors_size);
    for(unsigned int i=0; i<*number_of_vectors_size; i++)
      result_matrix->set_num_nonzeros(i,1,0);
    result_matrix->allocate_memory();
    result_matrix->set_to_zero();
    result_matrix->assemble();
  }
  else
  {
    PetscMatrixParallelComplex Wave = (*Wave_matrix);
    PetscMatrixParallelComplex WaveT(Wave.get_num_cols(),Wave.get_num_rows(),this_simulation->get_simulation_domain()->get_communicator());
    PetscMatrixParallelComplex* GR_inv_temp1 = NULL;
    PetscMatrixParallelComplex* GR_inv1 = NULL;
    PetscMatrixParallelComplex* GR_inv_temp2 = NULL;
    PetscMatrixParallelComplex* GR_inv2 = NULL;
    (&Wave)->hermitian_transpose_matrix(WaveT,MAT_INITIAL_MATRIX);//VL'
    std::string tic_toc_mult2 = NEMOUTILS_PREFIX(tic_toc_prefix+": GR_inv1=VL'*(E-H), GR_inv2=VL'*T");
    NemoUtils::tic(tic_toc_mult2);
    PetscMatrixParallelComplex::mult(WaveT,*E_minus_H_matrix,&GR_inv_temp1);//GR_inv1=VL'*(E-H)
    PetscMatrixParallelComplex::mult(WaveT,*T_matrix,&GR_inv2);//GR_inv2=VL'*T
    NemoUtils::toc(tic_toc_mult2);

    std::string tic_toc_mult3 = NEMOUTILS_PREFIX(tic_toc_prefix+": mult VL'*(E-H)*VL = GR_inv1");
    NemoUtils::tic(tic_toc_mult3);
    PetscMatrixParallelComplex::mult(*GR_inv_temp1,*Wave_matrix,&GR_inv1);//GR_inv1=VL'*(E-H)*VL
    NemoUtils::toc(tic_toc_mult3);

    std::string tic_toc_mult4 = NEMOUTILS_PREFIX(tic_toc_prefix+": mult VL'*T*VL");
    NemoUtils::tic(tic_toc_mult4);
    PetscMatrixParallelComplex::mult(*GR_inv2,*Wave_matrix,&GR_inv_temp2);//GR_inv2=VL'*T*VL
    NemoUtils::toc(tic_toc_mult4);

    delete GR_inv2;
    GR_inv2 = NULL;
    std::string tic_toc_mult5 = NEMOUTILS_PREFIX(tic_toc_prefix+": GR_inv2=VL'*T*VL*expmikdelta");
    NemoUtils::tic(tic_toc_mult5);
    PetscMatrixParallelComplex::mult(*GR_inv_temp2,*phase_factor,&GR_inv2);//GR_inv2=VL'*T*VL*expmikdelta
    NemoUtils::toc(tic_toc_mult5);

    std::string tic_toc_mult6 = NEMOUTILS_PREFIX(tic_toc_prefix+": add two matrix GR_inv1+GR_inv2");
    NemoUtils::tic(tic_toc_mult6);
    GR_inv1->add_matrix(*GR_inv2,DIFFERENT_NONZERO_PATTERN);//GR_inv1=VL'*(E-H)*VL+VL'*T*VL*expmikdelta
    NemoUtils::toc(tic_toc_mult6);

    PetscMatrixParallelComplex* GR_matrix_inv = NULL;
    //get the left upper nonzeros of GR_inv1
    vector<int> GR_rows_diagonal(*number_of_wave,0);
    vector<int> GR_rows_offdiagonal(*number_of_wave,0);
    for(unsigned int i=0; i<*number_of_wave; i++)
    {
      GR_rows_diagonal[i]=GR_inv1->get_nz_diagonal(i);
      GR_rows_offdiagonal[i]=GR_inv1->get_nz_offdiagonal(i);
    }
    transfer_matrix_get_full_submatrix(this_simulation,*number_of_wave,*number_of_wave,0,0,GR_rows_diagonal,GR_rows_offdiagonal,GR_inv1,GR_matrix_inv);
    PetscMatrixParallelComplex* GR_matrix = NULL;
    PetscMatrixParallelComplex* temp_G = NULL;
    PetscMatrixParallelComplex* temp_Gr = NULL;

    if(*number_of_wave==1)
    {
      std::complex<double> GR_matrix_1(0.0,0.0);
      GR_matrix_1 = 1.0/(GR_matrix_inv->get(0,0));
      GR_matrix = new PetscMatrixParallelComplex(1,1,this_simulation->get_simulation_domain()->get_communicator());
      GR_matrix->set_num_owned_rows(1);
      GR_matrix->set_num_nonzeros(0,1,0);
      GR_matrix->allocate_memory();
      GR_matrix->set_to_zero();
      GR_matrix->set(0,0,GR_matrix_1);
      GR_matrix->assemble();
      std::string tic_toc_mult9 = NEMOUTILS_PREFIX(tic_toc_prefix+": GR*VL'");
      NemoUtils::tic(tic_toc_mult9);
      PetscMatrixParallelComplex::mult(*GR_matrix,WaveT,&temp_G);//GR*VL'
      NemoUtils::toc(tic_toc_mult9);
      std::string tic_toc_mult10 = NEMOUTILS_PREFIX(tic_toc_prefix+": VL*GR*VL'");
      NemoUtils::tic(tic_toc_mult10);
      PetscMatrixParallelComplex::mult(*Wave_matrix,*temp_G,&temp_Gr);//VL*GR*VL'
      NemoUtils::toc(tic_toc_mult10);
      delete temp_G;
    }
    else
    {
      // ------------------------------------
      // method 1: inverse/mult
      // ------------------------------------
      //std::string tic_toc_mult7 = NEMOUTILS_PREFIX(tic_toc_prefix+": invert(GR_inv1+GR_inv2)");
      //NemoUtils::tic(tic_toc_mult7);
      //PetscMatrixParallelComplex::invert(*GR_matrix_inv,&GR_matrix);//GR=inv(GR_inv1)
      //PetscMatrixParallelComplex::mult(*Wave_matrix,*super_GR_matrix,&temp_G);//VL*GR
      //std::string tic_toc_mult9 = NEMOUTILS_PREFIX(tic_toc_prefix+": GR*VL'");
      //NemoUtils::tic(tic_toc_mult9);
      //PetscMatrixParallelComplex::mult(*GR_matrix,WaveT,&temp_G);//GR*VL'
      //NemoUtils::toc(tic_toc_mult9);
      //std::string tic_toc_mult10 = NEMOUTILS_PREFIX(tic_toc_prefix+": VL*GR*VL'");
      //NemoUtils::tic(tic_toc_mult10);
      //PetscMatrixParallelComplex::mult(*Wave_matrix,*temp_G,&temp_Gr);//VL*GR*VL'
      //NemoUtils::toc(tic_toc_mult10);

      // ------------------------------------
      // method 2: solve a linear equation as OMEN did
      // ------------------------------------
      //set up the Linear solver
      PetscMatrixParallelComplex temp_solution(GR_matrix_inv->get_num_rows(),WaveT.get_num_cols(),this_simulation->get_simulation_domain()->
          get_communicator() /*holder.geometry_communicator*/ );
      temp_solution.consider_as_full();
      temp_solution.allocate_memory();

      LinearSolverPetscComplex solver(*GR_matrix_inv,WaveT,&temp_solution);

      //prepare input options for the linear solver
      InputOptions options_for_linear_solver;
      if (options.check_option("linear_system_solver_for_G"))
      {
        string key("solver");
        string value = options.get_option("linear_system_solver_for_G", string("mumps"));
        options_for_linear_solver[key] = value;
      }
      solver.set_options(options_for_linear_solver);

      std::string tic_toc_prefix_linear= NEMOUTILS_PREFIX(tic_toc_prefix+": solve_linear_equation GR_inv1*X=VL'");
      NemoUtils::tic(tic_toc_prefix_linear);
      solver.solve();
      NemoUtils::toc(tic_toc_prefix_linear);

      temp_G=&temp_solution;
      std::string tic_toc_mult10 = NEMOUTILS_PREFIX(tic_toc_prefix+": VL*GR*VL'");
      NemoUtils::tic(tic_toc_mult10);
      PetscMatrixParallelComplex::mult(*Wave_matrix,*temp_G,&temp_Gr);//VL*GR*VL'
      NemoUtils::toc(tic_toc_mult10);
    }
    delete GR_inv_temp1;
    delete GR_inv1;
    delete GR_inv_temp2;
    delete GR_inv2;
    delete GR_matrix_inv;
    delete GR_matrix;
    delete temp_Gr;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_get_Hamiltonian(Simulation* this_simulation,std::string domain_index, Propagator*& ,const Domain* neighbor_domain,
    const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& Hamiltonian,
    PetscMatrixParallelComplex*& coupling_Hamiltonian, PetscMatrixParallelComplex*& Overlap,
    PetscMatrixParallelComplex*& coupling_Overlap, const DOFmapInterface*& subdomain_DOF,
    DOFmapInterface*& subdomain_coupling_DOF)
{
  //obtain Hamiltonian and coupling term for each subdomain, given by domain_index
  //neighbor_domain is the neighbor subdomain
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+this_simulation->get_name()+"\")::transfer_matrix_get_Hamiltonian ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this_simulation->get_name()+"\")::transfer_matrix_get_Hamiltonian ";
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  //1. find the Hamilton constructor
  std::string Hamilton_variable_name=std::string("Hamilton_constructor")+domain_index;
  std::string Hamilton_constructor_name;
  const InputOptions& options=this_simulation->get_options();
  if (options.check_option(Hamilton_variable_name))
    Hamilton_constructor_name=options.get_option(Hamilton_variable_name,std::string(""));
  else
    throw std::invalid_argument(prefix+" define \""+Hamilton_variable_name+"\"\n");
  Simulation* Hamilton_Constructor_temp;
  Hamilton_Constructor_temp = this_simulation->find_simulation(Hamilton_constructor_name);
  NEMO_ASSERT(Hamilton_Constructor_temp!=NULL,prefix+"Hamilton_constructor has not been found!\n");

  //2. find the lead (sub) domain that is supposed to be created by repartition
  std::string variable_name=std::string("lead_domain")+domain_index;
  std::string neighbor_domain_name;
  if (options.check_option(variable_name))
    neighbor_domain_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(prefix+"define \""+variable_name+"\"\n");
  neighbor_domain=Domain::get_domain(neighbor_domain_name);
  NEMO_ASSERT(neighbor_domain!=NULL,prefix+"Domain \""+neighbor_domain_name+"\" has not been found!\n");

  //For EH support: get the neighbor domain Hamilton constructor (later: will be asked for its DOFmapInterface)
  //find simulation/Schroedinger of a given domain
  Simulation* neighbor_Hamilton_Constructor=this_simulation->find_simulation_on_domain(neighbor_domain,Hamilton_Constructor_temp->get_type());
  //NEMO_ASSERT(neighbor_Hamilton_Constructor!=NULL,prefix+"have not found the neighbor_Hamilton_Constructor\n");
  if (neighbor_Hamilton_Constructor == NULL && (Hamilton_Constructor_temp->get_type().find("module") != std::string::npos || Hamilton_Constructor_temp->get_type().find("Module") != std::string::npos))
    neighbor_Hamilton_Constructor = Hamilton_Constructor_temp;
  NEMO_ASSERT(neighbor_Hamilton_Constructor != NULL, tic_toc_prefix + "have not found Hamilton_Constructor for neighbor_domain\n");
  NEMO_ASSERT(neighbor_Hamilton_Constructor->get_simulation_domain()==neighbor_domain,prefix+"have not found HamiltonConstructor of neighbor domain\n");

  //3. find the names of the momenta that are constructed by the Hamiltonian_Constructor
  //std::set<unsigned int> Hamilton_momentum_indices;
  //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
  //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
  //set_valley(output_Propagator, momentum, Hamilton_Constructor); //Hamilton_Constructor added by Bozidar
  //set_valley(output_Propagator, momentum, Hamilton_Constructor_temp); //additional set_valley added by Bozidar
  //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
  //TODO @Fabio, need to replace set_valley() and deal with 2 NemoMehsPoint in a general case
  //if(pointer_to_Hamilton_momentum_indices!=NULL)
  //if(momentum[*(pointer_to_Hamilton_momentum_indices->begin())].get_dim()==3)
  //temp_NemoMeshPoint=momentum[*(pointer_to_Hamilton_momentum_indices->begin())];

  //DOFmap "D3"="D1"+"D2" (for later usage)
  //PetscMatrixParallelComplex* temp_Hamiltonian = Hamiltonian;
  //PetscMatrixParallelComplex* temp_Overlap = Overlap;
  std::vector<NemoMeshPoint> sorted_momentum;
  
  if(!options.get_option("RF_with_kmesh",false))
  {    
    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,PropInterface->get_momentum_mesh_types(),Hamilton_Constructor_temp);
  }
  else
  {
    for(unsigned int ii=0; ii<momentum.size(); ii++)
      if(momentum[ii].get_dim()==3)
        sorted_momentum.push_back(momentum[ii]);

    vector<string> Hamilton_momenta;
    options.get_option("Hamilton_momenta",Hamilton_momenta);
    InputOptions& writeable_solver_options = Hamilton_Constructor_temp->get_reference_to_options();
    writeable_solver_options.set_option("quantum_number_order",Hamilton_momenta);
  }
 
  //bool avoid_copying_hamiltonian=options.get_option("avoid_copying_hamiltonian",false);
  Hamilton_Constructor_temp->get_data(std::string("Hamiltonian"),Hamiltonian,sorted_momentum,PropOptionInterface->get_avoid_copying_hamiltonian());
  NEMO_ASSERT(Hamiltonian!=NULL,prefix+"received NULL for the Hamiltonian\n");
  Hamilton_Constructor_temp->get_data(string("overlap_matrix"),Overlap,sorted_momentum);
  subdomain_DOF = &(Hamilton_Constructor_temp->get_const_dof_map(this_simulation->get_const_simulation_domain()));
  DOFmapInterface& ref_subdomain_coupling_DOF=neighbor_Hamilton_Constructor->get_dof_map(neighbor_domain);
  subdomain_coupling_DOF=&ref_subdomain_coupling_DOF;
  DOFmapInterface* temp_pointer=subdomain_coupling_DOF;
  Hamilton_Constructor_temp->get_data(std::string("Hamiltonian"),sorted_momentum, neighbor_domain, coupling_Hamiltonian, subdomain_coupling_DOF);
  if(subdomain_coupling_DOF!=temp_pointer)
    delete subdomain_coupling_DOF;
  subdomain_coupling_DOF=temp_pointer;
  NEMO_ASSERT(coupling_Hamiltonian!=NULL,prefix+"received NULL for the coupling_Hamiltonian\n");
  
  //DOFmapInterface* Over_subdomain_coupling_DOFmap = subdomain_coupling_DOF;
  Hamilton_Constructor_temp->get_data(string("overlap_matrix_coupling"), sorted_momentum, neighbor_domain, coupling_Overlap , subdomain_coupling_DOF);
  if(subdomain_coupling_DOF!=temp_pointer)
  delete subdomain_coupling_DOF;
  subdomain_coupling_DOF=NULL;

  NemoUtils::toc(tic_toc_prefix);
}

//12/09/2012 - Ganesh Hegde - Creating this placeholder that can be used if full submatrices need to be created
void PropagationUtilities::transfer_matrix_get_full_submatrix(Simulation* this_simulation,const int number_of_sub_rows, const int number_of_sub_cols, const int offset_rows,
    const int offset_cols, vector<int> rows_diagonal,vector<int> rows_offdiagonal,
    PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallelComplex*& result_matrix)
{
  //obtain dense submatrix from original matrix
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_submatrix ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_submatrix ";
  NEMO_ASSERT(rows_diagonal.size()==rows_offdiagonal.size(),prefix+"number of diagonal rows differs from number of offdiagonal rows\n");
  int local_number_of_sub_rows = rows_diagonal.size();

  std::vector<int> temp_rows(local_number_of_sub_rows) ;
  std::vector<int> temp_cols(number_of_sub_cols); //number_of_sub_cols is the same locally and globally
  for(int i=0; i<local_number_of_sub_rows && i<number_of_sub_rows; i++)
    temp_rows[i]=i+offset_rows;
  for(int i=0; i<number_of_sub_cols; i++)
    temp_cols[i]=i+offset_cols;
  //set the sparsity pattern of the result matrix
  result_matrix= new PetscMatrixParallelComplex(number_of_sub_rows,number_of_sub_cols,
      this_simulation->get_simulation_domain()->get_communicator());
  result_matrix->set_num_owned_rows(local_number_of_sub_rows);
  for(int i=0; i<local_number_of_sub_rows; i++)
    result_matrix->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
  result_matrix->consider_as_full();
  source_matrix->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,result_matrix);
  result_matrix->assemble();
  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::transfer_matrix_get_submatrix(Simulation* this_simulation,const int number_of_sub_rows, const int number_of_sub_cols, const int offset_rows,
    const int offset_cols, vector<int> rows_diagonal,vector<int> rows_offdiagonal,
    PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallelComplex*& result_matrix)
{
  //obtain submatrix from orignal matrix
  //index of matrix elements to extract is given by offset_rows/offset_cols
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_submatrix ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_submatrix ";
  NEMO_ASSERT(rows_diagonal.size()==rows_offdiagonal.size(),prefix+"number of diagonal rows differs from number of offdiagonal rows\n");
  int local_number_of_sub_rows = rows_diagonal.size();

  std::vector<int> temp_rows(local_number_of_sub_rows);
  std::vector<int> temp_cols(number_of_sub_cols); //number_of_sub_cols is the same locally and globally
  for(int i=0; i<local_number_of_sub_rows && i<number_of_sub_rows; i++)
    temp_rows[i]=i+offset_rows;
  for(int i=0; i<number_of_sub_cols; i++)
    temp_cols[i]=i+offset_cols;
  //set the sparsity pattern of the result matrix
  result_matrix= new PetscMatrixParallelComplex(number_of_sub_rows,number_of_sub_cols,
      this_simulation->get_simulation_domain()->get_communicator());
  result_matrix->set_num_owned_rows(local_number_of_sub_rows);

  int total_non_zeros = 0;

  for(int i=0; i<local_number_of_sub_rows; i++)
  {
    result_matrix->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
    total_non_zeros += rows_diagonal[i];
  }

  //NEMO_ASSERT(source_matrix->check_if_sparse()==false,"has sparse\n");
  //NEMO_ASSERT(source_matrix->is_scalar==false,"has scalar\n");


  //NEMO_ASSERT(total_non_zeros > 0, prefix + " no nonzero rows\n");
  source_matrix->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,result_matrix);
  result_matrix->assemble();
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_get_submatrix_double(Simulation* this_simulation,const int number_of_sub_rows, const int number_of_sub_cols, const int offset_rows,
    const int offset_cols, vector<int> rows_diagonal,vector<int> rows_offdiagonal,
    PetscMatrixParallel<double>*& source_matrix, PetscMatrixParallel<double>*& result_matrix)
{
  //obtain submatrix from orignal matrix -- double precision version
  //index of matrix elements to extract is given by offset_rows/offset_cols
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_submatrix ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_submatrix ";
  NEMO_ASSERT(rows_diagonal.size()==rows_offdiagonal.size(),prefix+"number of diagonal rows differs from number of offdiagonal rows\n");
  int local_number_of_sub_rows = rows_diagonal.size();

  std::vector<int> temp_rows(local_number_of_sub_rows);
  std::vector<int> temp_cols(number_of_sub_cols); //number_of_sub_cols is the same locally and globally
  for(int i=0; i<local_number_of_sub_rows && i<number_of_sub_rows; i++)
    temp_rows[i]=i+offset_rows;
  for(int i=0; i<number_of_sub_cols; i++)
    temp_cols[i]=i+offset_cols;
  //set the sparsity pattern of the result matrix
  result_matrix= new PetscMatrixParallel<double>(number_of_sub_rows,number_of_sub_cols,
      this_simulation->get_simulation_domain()->get_communicator() );
  //result_matrix->set_num_owned_rows(number_of_sub_rows);
  result_matrix->set_num_owned_rows(local_number_of_sub_rows);
  for(int i=0; i<local_number_of_sub_rows; i++)
    result_matrix->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
  source_matrix->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,result_matrix);
  result_matrix->assemble();
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_initialize_temp_matrix(Simulation* this_simulation,const int number_of_rows, const int number_of_cols, vector<int> number_of_nonzero_cols,
    vector<int> number_of_nonzero_cols_another_cpu, PetscMatrixParallelComplex*& result_matrix)
{
  //initialize matrix 
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_initialize_temp_matrix ");
  NemoUtils::tic(tic_toc_prefix);
  if(result_matrix!=NULL)
    delete result_matrix;
  result_matrix = new PetscMatrixParallelComplex(number_of_rows,number_of_cols,
      this_simulation->get_simulation_domain()->get_communicator() );
  //set matrix pattern
  int number_of_local_rows = number_of_nonzero_cols.size();
  result_matrix->set_num_owned_rows(number_of_local_rows);
  for(int i=0; i<number_of_local_rows && i<number_of_rows; i++)
    result_matrix->set_num_nonzeros_for_local_row(i,number_of_nonzero_cols[i],number_of_nonzero_cols_another_cpu[i]);
  result_matrix->allocate_memory();
  result_matrix->set_to_zero();
  NemoUtils::toc(tic_toc_prefix);
/*
  if(result_matrix->check_if_sparse())
       	  std::cout<<"has sparse"<<endl;
       if(result_matrix->is_scalar)
       	  std::cout<<"has scalar"<<endl;
*/
}

void PropagationUtilities::transfer_matrix_initialize_temp_matrix_double(Simulation* this_simulation,const int number_of_rows, const int number_of_cols, vector<int> number_of_nonzero_cols,
    vector<int> number_of_nonzero_cols_another_cpu, PetscMatrixParallel<double>*& result_matrix)
{
  //initialize matrix -- double version
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_initialize_temp_matrix ");
  NemoUtils::tic(tic_toc_prefix);
  if(result_matrix!=NULL)
    delete result_matrix;
  result_matrix = new PetscMatrixParallel<double>(number_of_rows,number_of_cols,
      this_simulation->get_simulation_domain()->get_communicator() );
  //set matrix pattern
  int number_of_local_rows = number_of_nonzero_cols.size();
  result_matrix->set_num_owned_rows(number_of_local_rows);
  for(int i=0; i<number_of_local_rows && i<number_of_rows; i++)
    result_matrix->set_num_nonzeros_for_local_row(i,number_of_nonzero_cols[i],number_of_nonzero_cols_another_cpu[i]);
  result_matrix->allocate_memory();
  result_matrix->set_to_zero();
  NemoUtils::toc(tic_toc_prefix);
}

//11/9/2012 - Ganesh Hegde: creating this function to initialize some of the matrices involved, esp Wave_left and expmi_delta_left as full.
//The arguments are a carry-over from the sparse version. Some of the arguments won't be used, but the arguments can be cleaned up later.
//As of Now this is a placeholder because E-H is still sparse and multiplying with this may cause a conflict.
void PropagationUtilities::transfer_matrix_initialize_temp_full_matrix(Simulation* this_simulation,const int number_of_rows, const int number_of_cols, vector<int> number_of_nonzero_cols,
    vector<int> number_of_nonzero_cols_another_cpu, PetscMatrixParallelComplex*& result_matrix)
{
	/*
	NemoMatrixInterface* A_interface=NULL;
	if(result_matrix!=NULL)
	{
		A_interface = NemoFactory::matrix_instance(
				"NemoMatrixComplex", result_matrix->get_num_cols(),
				result_matrix->get_num_rows(),
				result_matrix->get_communicator());
		A_interface->consider_as_full();
		A_interface->allocate_memory();
		int ii = 0, jj = 0;
		for (; ii < A_interface->get_num_rows(); ii++)
		{
			for (jj = 0; jj < A_interface->get_num_cols(); jj++)
			{
				A_interface->set(ii, jj, result_matrix->get(ii, jj));
			}
		}
		A_interface->assemble();
	}//copy A_interface from result_matrix



	//comparing with PETSC
	if (A_interface != NULL)
	{
		delete A_interface;
	}
	A_interface = NemoFactory::matrix_instance("NemoMatrixComplex",
			number_of_rows, number_of_cols,
			this_simulation->get_simulation_domain()->get_communicator());

	int number_of_local_rows_A = number_of_nonzero_cols.size();
	A_interface->set_num_owned_rows(number_of_local_rows_A);
	for (int iA = 0; iA < number_of_local_rows_A && iA < number_of_rows; iA++)
		A_interface->set_num_nonzeros_for_local_row(iA,
				number_of_nonzero_cols[iA],
				number_of_nonzero_cols_another_cpu[iA]);

	A_interface->consider_as_full();
	A_interface->allocate_memory();
	A_interface->set_to_zero();
*/

  //initialize matrix -- dense
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_initialize_temp_full_matrix ");
  NemoUtils::tic(tic_toc_prefix);
  if(result_matrix!=NULL)
    delete result_matrix;
  result_matrix = new PetscMatrixParallelComplex(number_of_rows,number_of_cols,
      this_simulation->get_simulation_domain()->get_communicator());
  //set matrix pattern
  int number_of_local_rows = number_of_nonzero_cols.size();
  result_matrix->set_num_owned_rows(number_of_local_rows);
  for(int i=0; i<number_of_local_rows && i<number_of_rows; i++)
    result_matrix->set_num_nonzeros_for_local_row(i,number_of_nonzero_cols[i],number_of_nonzero_cols_another_cpu[i]);
  result_matrix->consider_as_full();
  result_matrix->allocate_memory();
  result_matrix->set_to_zero();
  NemoUtils::toc(tic_toc_prefix);
/*
  if(result_matrix->check_if_sparse())
     	  std::cout<<"has sparse"<<endl;
     if(result_matrix->is_scalar)
     	  std::cout<<"has scalar"<<endl;
*/
  /*
  if(A_interface!=NULL)
  {
	  delete A_interface;
	  A_interface=NULL;
  }
*/

}

void PropagationUtilities::transfer_matrix_initialize_temp_full_matrix_double(Simulation* this_simulation,const int number_of_rows, const int number_of_cols,
    vector<int> number_of_nonzero_cols,
    vector<int> number_of_nonzero_cols_another_cpu, PetscMatrixParallel<double>*& result_matrix)
{
  //initialize matrix -- dense double version
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_initialize_temp_full_matrix_double ");
  NemoUtils::tic(tic_toc_prefix);
  if(result_matrix!=NULL)
    delete result_matrix;
  result_matrix = new PetscMatrixParallel<double>(number_of_rows,number_of_cols,
      this_simulation->get_simulation_domain()->get_communicator() );
  //set matrix pattern
  int number_of_local_rows = number_of_nonzero_cols.size();
  result_matrix->set_num_owned_rows(number_of_local_rows);
  result_matrix->consider_as_full();
  for(int i=0; i<number_of_local_rows && i<number_of_rows; i++)
    result_matrix->set_num_nonzeros_for_local_row(i,number_of_nonzero_cols[i],number_of_nonzero_cols_another_cpu[i]);
  result_matrix->allocate_memory();
  result_matrix->set_to_zero();
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_set_matrix_elements(Simulation* this_simulation,const int start_rows_result,
    const int start_cols_result,
    const int source_offset_rows,
    PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallelComplex*& result_matrix)
{
/*	if(source_matrix->check_if_sparse())
	     	  std::cout<<"has sparse"<<endl;
	     if(source_matrix->is_scalar)
	     	  std::cout<<"has scalar"<<endl;
*/

	//set matrix elements from source_matrix
  //start_rows_result/start_cols_result give the index of where to put the first element in the result_matrix
  //source_offset_rows/source_offset_cols give the index of where to extract the first element in the source_matrix
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_set_matrix_elements ");
  NemoUtils::tic(tic_toc_prefix);
  //fill elements one-by-one
  cplx temp_val(0.0,0.0);

  int source_start_local_row;
  int source_end_local_row;
  source_matrix->get_ownership_range(source_start_local_row, source_end_local_row);
  source_start_local_row = source_start_local_row+source_offset_rows;

  const std::complex<double>* pointer_to_data= NULL;
  vector<cplx> data_vector;
  vector<int> col_index;
  int n_nonzeros=0;
  const int* n_col_nums=NULL;
  for(int i=(start_rows_result+source_start_local_row); i<(start_rows_result+source_end_local_row); i++)
  {
    //row-wise, this is the most efficient way so far
    source_matrix->get_row(i-start_rows_result,&n_nonzeros,n_col_nums,pointer_to_data);
    col_index.resize(n_nonzeros,0);
    data_vector.resize(n_nonzeros,cplx(0.0,0.0));
    for(int j=0; j<n_nonzeros; j++)
    {
      col_index[j]=n_col_nums[j]+start_cols_result;
      temp_val=pointer_to_data[j];
      data_vector[j]=temp_val;
    }
    if (n_nonzeros > 0)
      result_matrix->set(i-source_start_local_row,col_index,data_vector);
    source_matrix->store_row(i-start_rows_result,&n_nonzeros,n_col_nums,pointer_to_data);
  }
  NemoUtils::toc(tic_toc_prefix);
/*
  if(result_matrix->check_if_sparse())
       	  std::cout<<"has sparse"<<endl;
       if(result_matrix->is_scalar)
       	  std::cout<<"has scalar"<<endl;
*/

}

void PropagationUtilities::transfer_matrix_set_matrix_elements_double(Simulation* this_simulation,const int start_rows_result,
    const int start_cols_result,
    const int source_offset_rows,
    PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallel<double>*& result_matrix)
{
  //set matrix elements from source_matrix --double version
  //start_rows_result/start_cols_result give the index of where to put the first element in the result_matrix
  //source_offset_rows/source_offset_cols give the index of where to extract the first element in the source_matrix
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_set_matrix_elements ");
  NemoUtils::tic(tic_toc_prefix);
  //fill elements one-by-one
  double temp_val=0.0;

  int source_start_local_row;
  int source_end_local_row;
  source_matrix->get_ownership_range(source_start_local_row, source_end_local_row);
  source_start_local_row = source_start_local_row+source_offset_rows;

  const cplx* pointer_to_data= NULL;
  vector<double> data_vector;
  vector<int> col_index;
  int n_nonzeros=0;
  const int* n_col_nums=NULL;
  for(int i=(start_rows_result+source_start_local_row); i<(start_rows_result+source_end_local_row); i++)
  {
    //row-wise -- this is the most efficient way so far
    source_matrix->get_row(i-start_rows_result,&n_nonzeros,n_col_nums,pointer_to_data);
    col_index.resize(n_nonzeros,0);
    data_vector.resize(n_nonzeros,0.0);
    for(int j=0; j<n_nonzeros; j++)
    {
      col_index[j]=n_col_nums[j]+start_cols_result;
      temp_val=pointer_to_data[j].real();
      data_vector[j]=temp_val;
    }
    result_matrix->set(i-source_start_local_row,col_index,data_vector);
    source_matrix->store_row(i-start_rows_result,&n_nonzeros,n_col_nums,pointer_to_data);
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_get_M_matrix(Simulation* this_simulation,bool LRA,PetscMatrixParallelComplex*& D00, PetscMatrixParallelComplex*& D11,
    PetscMatrixParallelComplex*& D22, PetscMatrixParallelComplex*& D33,
    PetscMatrixParallelComplex*& D32, PetscMatrixParallelComplex*& D43,
    PetscMatrixParallelComplex*& E_minus_H_matrix, PetscMatrixParallelComplex*& T34,
    PetscMatrixParallelComplex*& T43, PetscMatrixParallelComplex*& M1_matrix, PetscMatrixParallelComplex*& M2_matrix)
{
  //calculate M matrix
  //HP=|D00 D01 0   D43|; P=|0 0 0   D43|; M=inv(HP)*P
  //   |D10 D11 D12   0|    |0 0 0   0  |
  //   |0   D21 D22   0|    |0 0 0   0  |
  //   |D34 0   D32 D33|    |0 0 D32 D33|
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix");
  NemoUtils::tic(tic_toc_prefix);

  const InputOptions& options=this_simulation->get_options();
  //double time10=NemoUtils::get_time();

  //1. obatin matrix information
  unsigned int number_of_EmH_rows = E_minus_H_matrix->get_num_rows();
  unsigned int number_of_EmH_local_rows = E_minus_H_matrix->get_num_owned_rows();
  int start_own_EmH_rows;
  int end_own_EmH_rows_p1;
  E_minus_H_matrix->get_ownership_range(start_own_EmH_rows,end_own_EmH_rows_p1);

  unsigned int number_of_00_rows = D00->get_num_rows();
  unsigned int number_of_11_rows = D11->get_num_rows();
  unsigned int number_of_22_rows = D22->get_num_rows();
  unsigned int number_of_22_cols = D22->get_num_cols();
  unsigned int number_of_33_rows = D33->get_num_rows();
  unsigned int number_of_33_cols = D33->get_num_cols();

  unsigned int number_of_00_local_rows = D00->get_num_owned_rows();
  unsigned int number_of_11_local_rows = D11->get_num_owned_rows();
  unsigned int number_of_22_local_rows = D22->get_num_owned_rows();
  unsigned int number_of_33_local_rows = D33->get_num_owned_rows();

  int start_own_00_rows;
  int end_own_00_rows_p1;
  D00->get_ownership_range(start_own_00_rows,end_own_00_rows_p1);
  int start_own_11_rows;
  int end_own_11_rows_p1;
  D11->get_ownership_range(start_own_11_rows,end_own_11_rows_p1);
  int start_own_22_rows;
  int end_own_22_rows_p1;
  D22->get_ownership_range(start_own_22_rows,end_own_22_rows_p1);
  int start_own_33_rows;
  int end_own_33_rows_p1;
  D33->get_ownership_range(start_own_33_rows,end_own_33_rows_p1);

  //2a. define sparsity pattern of P matrix
  vector<int> nz_per_owned_P_rows_diagonal(number_of_EmH_local_rows,0);
  vector<int>
  nz_per_owned_P_rows_offdiagonal(number_of_EmH_local_rows,0);

  std::string tic_toc_prefix_local_rows= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_local_rows");
  NemoUtils::tic(tic_toc_prefix_local_rows);
  for(unsigned int i=0; i<number_of_00_local_rows; i++)
  {
    nz_per_owned_P_rows_diagonal[i]=T43->get_nz_diagonal(i);
    nz_per_owned_P_rows_offdiagonal[i]=T43->get_nz_offdiagonal(i);
  }
  NemoUtils::toc(tic_toc_prefix_local_rows);

  std::string tic_toc_prefix_local_EmH_rows= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_EmH_local_rows");
  NemoUtils::tic(tic_toc_prefix_local_EmH_rows);

  for(unsigned int i=number_of_EmH_local_rows-number_of_33_local_rows; i<number_of_EmH_local_rows; i++)
  {
    nz_per_owned_P_rows_diagonal[i]=D32->get_nz_diagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows))
                                    +D33->get_nz_diagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows));
    nz_per_owned_P_rows_offdiagonal[i]=D32->get_nz_offdiagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows))
                                       +D33->get_nz_offdiagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows));
  }
  NemoUtils::toc(tic_toc_prefix_local_EmH_rows);

  //2b. set P matrix elements
  unsigned int offset_rows = start_own_00_rows+start_own_11_rows+start_own_22_rows+start_own_33_rows;
  std::string tic_toc_prefix_P_matrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_P_matrix");
  NemoUtils::tic(tic_toc_prefix_P_matrix);

  PetscMatrixParallelComplex* P_matrix = NULL;
  unsigned int number_of_P_cols = number_of_22_cols+number_of_33_cols;
  //  transfer_matrix_initialize_temp_matrix(number_of_EmH_rows, number_of_P_cols, number_of_EmH_rows, 0,
  //  number_of_EmH_rows, nz_per_owned_P_rows_diagonal, nz_per_owned_P_rows_offdiagonal, P_matrix); //Yu: if solve M=inv(HP)*P
  transfer_matrix_initialize_temp_full_matrix(this_simulation,number_of_EmH_rows, number_of_P_cols,
      nz_per_owned_P_rows_diagonal, nz_per_owned_P_rows_offdiagonal, P_matrix); //Yu: if solve HP*M=P

  transfer_matrix_set_matrix_elements(this_simulation,offset_rows,number_of_P_cols-number_of_33_cols,0,D43,P_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,
                                      number_of_P_cols-number_of_33_cols,0,D33,P_matrix);
  transfer_matrix_set_matrix_elements(this_simulation,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,
                                      0,0,D32,P_matrix);

  P_matrix->assemble();
  *P_matrix *= std::complex<double> (-1.0,0.0);

  NemoUtils::toc(tic_toc_prefix_P_matrix);


  //3. set H-P matrix
  std::string tic_toc_prefix_H_P_matrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_H_P_matrix");
  NemoUtils::tic(tic_toc_prefix_H_P_matrix);
  PetscMatrixParallelComplex* HP_matrix = new PetscMatrixParallelComplex(*E_minus_H_matrix);
  HP_matrix->add_matrix(*T43,DIFFERENT_NONZERO_PATTERN);
  HP_matrix->add_matrix(*T34,DIFFERENT_NONZERO_PATTERN);
  HP_matrix->assemble();
  NemoUtils::toc(tic_toc_prefix_H_P_matrix);

  //4. obtain M matrix
  std::string tic_toc_prefix_M_matrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_M_matrix");
  NemoUtils::tic(tic_toc_prefix_M_matrix);

  {
    //4a. we need to solve eigenvalues and eigenvectors for M=inv(HP)*P
    //  // ================================
    //  // Yu: Method I  solver M=inv(HP)*P
    //  // --------------------------------
    //  std::string tic_toc_prefix_M_matrix_compute_inverse= NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::transfer_matrix_get_M_matrix_step_M_matrix_compute_inverse");
    //  NemoUtils::tic(tic_toc_prefix_M_matrix_compute_inverse);
    //  PetscMatrixParallelComplex * temp_matrix = NULL;
    //  PetscMatrixParallelComplex::invert(*HP_matrix,&temp_matrix);//inv(HP)
    //  NemoUtils::toc(tic_toc_prefix_M_matrix_compute_inverse);
    //
    //  std::string tic_toc_prefix_M_matrix_compute_multiply= NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::transfer_matrix_get_M_matrix_step_M_matrix_compute_multiply");
    //  NemoUtils::tic(tic_toc_prefix_M_matrix_compute_multiply);
    //
    //  PetscMatrixParallelComplex * M_matrix = NULL;
    //  PetscMatrixParallelComplex::mult(*temp_matrix,*P_matrix,&M_matrix);
    //  delete temp_matrix;
    //  NemoUtils::toc(tic_toc_prefix_M_matrix_compute_multiply);
    //  //M_matrix->save_to_matlab_file("M_matrix.m");
    //  // =========end of Method I===================

    // ==================================
    // Yu: Method II  solve M from HP*M=P
    // ----------------------------------
    // to get M, we solve linear equation HP*M=P
    PetscMatrixParallelComplex* M_matrix=NULL;
    //4a.1 set up the Linear solver
    PetscMatrixParallelComplex temp_solution(P_matrix->get_num_rows(),P_matrix->get_num_cols(),this_simulation->get_simulation_domain()->get_communicator());
    temp_solution.consider_as_full();
    temp_solution.allocate_memory();
    {
      LinearSolverPetscComplex solver(*HP_matrix,*P_matrix,&temp_solution);

      //4a.2 prepare input options for the linear solver
      InputOptions options_for_linear_solver;
      string value("mumps");
      if (options.check_option("linear_system_solver"))
        value = options.get_option("linear_system_solver", string("mumps"));
      string key("solver");
      options_for_linear_solver[key] = value;
      solver.set_options(options_for_linear_solver);

      std::string tic_toc_prefix_M_matrix_compute_inverse= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_M_matrix_solve_linear_equation");
      NemoUtils::tic(tic_toc_prefix_M_matrix_compute_inverse);
      solver.solve();
      NemoUtils::toc(tic_toc_prefix_M_matrix_compute_inverse);
    }
    M_matrix=&temp_solution;
    //M_matrix->save_to_matlab_file("M_matrix.m");
    // =====end of Method II======================

    delete P_matrix;
    delete HP_matrix;

    //4b. extract M1 and M2 submatrix from M
    unsigned int number_of_M1_sub_rows = number_of_00_rows+number_of_11_rows;
    unsigned int number_of_M2_sub_rows = number_of_22_rows+number_of_33_rows;
    unsigned int number_of_M1_sub_cols = number_of_P_cols;
    unsigned int number_of_M2_sub_cols = number_of_M1_sub_cols;
    unsigned int number_of_M1_local_rows = number_of_00_local_rows+number_of_11_local_rows;
    unsigned int number_of_M2_local_rows = number_of_22_local_rows+number_of_33_local_rows;
    //unsigned int start_M1_rows = start_own_00_rows+start_own_11_rows;
    unsigned int start_M2_rows = start_own_22_rows+start_own_33_rows;

    //consider as full matrix -- get_submatrix not work in parallel
    vector<int> M1_rows_diagonal(number_of_M1_local_rows,0);
    vector<int> M1_rows_offdiagonal(number_of_M1_local_rows,0);

    std::string tic_toc_prefix_number_of_M1_local_rows = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix:step_number_of_M1_local_rows");
    NemoUtils::tic(tic_toc_prefix_number_of_M1_local_rows);

    for(unsigned int i=0; i<number_of_M1_local_rows; i++)
    {
      M1_rows_diagonal[i]=number_of_M2_sub_rows;
      M1_rows_offdiagonal[i]=0;
    }
    NemoUtils::toc(tic_toc_prefix_number_of_M1_local_rows);

    std::string tic_toc_prefix_number_of_M2_local_rows = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix:step_number_of_M2_local_rows");
    NemoUtils::tic(tic_toc_prefix_number_of_M2_local_rows);

    vector<int> M2_rows_diagonal(number_of_M2_local_rows,0);
    vector<int> M2_rows_offdiagonal(number_of_M2_local_rows,0);
    for(unsigned int i=0; i<number_of_M2_local_rows; i++)
    {
      M2_rows_diagonal[i]=number_of_M2_sub_rows;
      M2_rows_offdiagonal[i]=0;
    }
    NemoUtils::toc(tic_toc_prefix_number_of_M2_local_rows);

    std::string tic_toc_prefix_M_matrix_get_submatrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_M_matrix_get_submatrix");
    NemoUtils::tic(tic_toc_prefix_M_matrix_get_submatrix);

    //M1 is the upper right block of M
    transfer_matrix_get_submatrix(this_simulation,number_of_M1_sub_rows,number_of_M2_sub_rows,0,0,M1_rows_diagonal,M1_rows_offdiagonal,M_matrix, M1_matrix);

    //M2 is the lower right block of M
    if(LRA)
    {
      //set M2
      M2_matrix = new PetscMatrixParallelComplex(number_of_M2_sub_rows,number_of_M2_sub_cols,
          this_simulation->get_simulation_domain()->get_communicator() );
      M2_matrix->set_num_owned_rows(number_of_M2_local_rows);
      for(unsigned i=0; i<number_of_M2_local_rows; i++)
        M2_matrix->set_num_nonzeros(i+start_M2_rows,number_of_M2_sub_cols,0);
      M2_matrix->allocate_memory();
      M2_matrix->set_to_zero();

      const cplx* pointer_to_data= NULL;
      cplx temp_val=0.0;
      vector<cplx> data_vector;
      vector<int> col_index;
      int n_nonzeros = 0;
      const int* n_col_nums;
      for(unsigned int i=start_M2_rows; i<start_M2_rows+number_of_M2_local_rows; i++)
      {
        M_matrix->get_row(i+number_of_M1_local_rows,&n_nonzeros,n_col_nums,pointer_to_data);
        col_index.resize(number_of_M2_sub_cols,0);
        data_vector.resize(number_of_M2_sub_cols,cplx(0.0,0.0));
        for(unsigned int j=0; j<number_of_M2_sub_cols; j++)
        {
          col_index[j]=j;
          temp_val=pointer_to_data[j];
          data_vector[j]=temp_val;
        }
        M2_matrix->set(i,col_index,data_vector);
        M_matrix->store_row(i+number_of_M1_local_rows,&n_nonzeros,n_col_nums,pointer_to_data);
      }
      M2_matrix->assemble();
    }
    else
      transfer_matrix_get_submatrix(this_simulation,number_of_M2_sub_rows,number_of_M2_sub_rows,number_of_M1_sub_rows,0,M2_rows_diagonal,M2_rows_offdiagonal,M_matrix,
                                    M2_matrix);
    NemoUtils::toc(tic_toc_prefix_M_matrix_get_submatrix);
  }
  NemoUtils::toc(tic_toc_prefix_M_matrix);
  NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::transfer_matrix_get_M_matrix_double(Simulation* this_simulation,bool LRA,PetscMatrixParallelComplex*& D00, PetscMatrixParallelComplex*& D11,
    PetscMatrixParallelComplex*& D22, PetscMatrixParallelComplex*& D33,
    PetscMatrixParallelComplex*& D32, PetscMatrixParallelComplex*& D43,
    PetscMatrixParallelComplex*& E_minus_H_matrix, PetscMatrixParallelComplex*& T34,
    PetscMatrixParallelComplex*& T43, PetscMatrixParallel<double>*& M1_matrix, PetscMatrixParallel<double>*& M2_matrix)
{
  //calculate M matrix -- double version
  //HP=|D00 D01 0   D43|; P=|0 0 0   D43|; M=inv(HP)*P
  //   |D10 D11 D12   0|    |0 0 0   0  |
  //   |0   D21 D22   0|    |0 0 0   0  |
  //   |D34 0   D32 D33|    |0 0 D32 D33|
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix");
  NemoUtils::tic(tic_toc_prefix);

  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  const InputOptions& options=this_simulation->get_options();
  //1. obatin matrix information
  unsigned int number_of_EmH_rows = E_minus_H_matrix->get_num_rows();
  unsigned int number_of_EmH_cols = E_minus_H_matrix->get_num_cols();
  unsigned int number_of_EmH_local_rows = E_minus_H_matrix->get_num_owned_rows();
  int start_own_EmH_rows;
  int end_own_EmH_rows_p1;
  E_minus_H_matrix->get_ownership_range(start_own_EmH_rows,end_own_EmH_rows_p1);

  unsigned int number_of_00_rows = D00->get_num_rows();
  unsigned int number_of_11_rows = D11->get_num_rows();
  unsigned int number_of_22_rows = D22->get_num_rows();
  unsigned int number_of_22_cols = D22->get_num_cols();
  unsigned int number_of_33_rows = D33->get_num_rows();
  unsigned int number_of_33_cols = D33->get_num_cols();

  unsigned int number_of_00_local_rows = D00->get_num_owned_rows();
  unsigned int number_of_11_local_rows = D11->get_num_owned_rows();
  unsigned int number_of_22_local_rows = D22->get_num_owned_rows();
  unsigned int number_of_33_local_rows = D33->get_num_owned_rows();

  int start_own_00_rows;
  int end_own_00_rows_p1;
  D00->get_ownership_range(start_own_00_rows,end_own_00_rows_p1);
  int start_own_11_rows;
  int end_own_11_rows_p1;
  D11->get_ownership_range(start_own_11_rows,end_own_11_rows_p1);
  int start_own_22_rows;
  int end_own_22_rows_p1;
  D22->get_ownership_range(start_own_22_rows,end_own_22_rows_p1);
  int start_own_33_rows;
  int end_own_33_rows_p1;
  D33->get_ownership_range(start_own_33_rows,end_own_33_rows_p1);

  //2a. define sparsity pattern of P matrix
  vector<int> nz_per_owned_P_rows_diagonal(number_of_EmH_local_rows,0);
  vector<int> nz_per_owned_P_rows_offdiagonal(number_of_EmH_local_rows,0);

  std::string tic_toc_prefix_local_rows= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_local_rows");
  NemoUtils::tic(tic_toc_prefix_local_rows);
  for(unsigned int i=0; i<number_of_00_local_rows; i++)
  {
    nz_per_owned_P_rows_diagonal[i]=T43->get_nz_diagonal(i);
    nz_per_owned_P_rows_offdiagonal[i]=T43->get_nz_offdiagonal(i);
  }
  NemoUtils::toc(tic_toc_prefix_local_rows);

  std::string tic_toc_prefix_local_EmH_rows= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_EmH_local_rows");
  NemoUtils::tic(tic_toc_prefix_local_EmH_rows);

  for(unsigned int i=number_of_EmH_local_rows-number_of_33_local_rows; i<number_of_EmH_local_rows; i++)
  {
    nz_per_owned_P_rows_diagonal[i]=D32->get_nz_diagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows))
                                    +D33->get_nz_diagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows));
    nz_per_owned_P_rows_offdiagonal[i]=D32->get_nz_offdiagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows))
                                       +D33->get_nz_offdiagonal(i-(number_of_EmH_local_rows-number_of_33_local_rows));
  }
  NemoUtils::toc(tic_toc_prefix_local_EmH_rows);

  //2b. set P matrix elements
  unsigned int offset_rows = start_own_00_rows+start_own_11_rows+start_own_22_rows+start_own_33_rows;
  std::string tic_toc_prefix_P_matrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_P_matrix");
  NemoUtils::tic(tic_toc_prefix_P_matrix);

  PetscMatrixParallel<double>* P_matrix = NULL;
  unsigned int number_of_P_cols = number_of_22_cols+number_of_33_cols;
  // transfer_matrix_initialize_temp_matrix_double(number_of_EmH_rows, number_of_P_cols, number_of_EmH_rows, 0,
  //number_of_EmH_rows, nz_per_owned_P_rows_diagonal, nz_per_owned_P_rows_offdiagonal, P_matrix); //Yu: if solve M=inv(HP)*P
  transfer_matrix_initialize_temp_full_matrix_double(this_simulation,number_of_EmH_rows, number_of_P_cols,
      nz_per_owned_P_rows_diagonal, nz_per_owned_P_rows_offdiagonal, P_matrix); //Yu: if solve HP*M=P

  transfer_matrix_set_matrix_elements_double(this_simulation,offset_rows,number_of_P_cols-number_of_33_cols,0,D43,P_matrix);
  transfer_matrix_set_matrix_elements_double(this_simulation,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,
      number_of_P_cols-number_of_33_cols,0,D33,P_matrix);
  transfer_matrix_set_matrix_elements_double(this_simulation,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,
      0,0,D32,P_matrix);

  P_matrix->assemble();
  *P_matrix *= -1.0;
  NemoUtils::toc(tic_toc_prefix_P_matrix);

  //3. set H-P matrix
  std::string tic_toc_prefix_H_P_matrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_H_P_matrix");
  NemoUtils::tic(tic_toc_prefix_H_P_matrix);

  PetscMatrixParallel<double>* E_minus_H_matrix_double = new PetscMatrixParallel<double>(number_of_EmH_rows,number_of_EmH_cols,
      this_simulation->get_simulation_domain()->get_communicator());
  extract_real_part(E_minus_H_matrix, E_minus_H_matrix_double);

  PetscMatrixParallel<double>* HP_matrix = new PetscMatrixParallel<double>(*E_minus_H_matrix_double); //NOTE - this is waste of time and increases peak memory
  delete E_minus_H_matrix_double;
  PetscMatrixParallel<double>* T34_double = new PetscMatrixParallel<double>(number_of_EmH_rows,number_of_EmH_cols,
      this_simulation->get_simulation_domain()->get_communicator() );
  extract_real_part(T34,T34_double);

  HP_matrix->add_matrix(*T34_double,DIFFERENT_NONZERO_PATTERN);
  delete T34_double;
  PetscMatrixParallel<double>* T43_double = new PetscMatrixParallel<double>(number_of_EmH_rows,number_of_EmH_cols,
      this_simulation->get_simulation_domain()->get_communicator());
  extract_real_part(T43,T43_double);

  HP_matrix->add_matrix(*T43_double,DIFFERENT_NONZERO_PATTERN);
  delete T43_double;
  HP_matrix->assemble();

  NemoUtils::toc(tic_toc_prefix_H_P_matrix);

  //4. obtain M matrix
  std::string tic_toc_prefix_M_matrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_M_matrix");
  NemoUtils::tic(tic_toc_prefix_M_matrix);
  {
    //4a. we need to solve eigenvalues and eigenvectors for M=inv(HP)*P
    //// ================================
    //// Yu: Method I  solver M=inv(HP)*P
    //// --------------------------------
    //std::string tic_toc_prefix_M_matrix_compute_inverse= NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::transfer_matrix_get_M_matrix_step_M_matrix_compute_inverse");
    //NemoUtils::tic(tic_toc_prefix_M_matrix_compute_inverse);
    //PetscMatrixParallel<double> * temp_matrix = NULL;
    //invert(*HP_matrix,&temp_matrix);//inv(HP)
    //NemoUtils::toc(tic_toc_prefix_M_matrix_compute_inverse);
    //std::string tic_toc_prefix_M_matrix_compute_multiply= NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::transfer_matrix_get_M_matrix_step_M_matrix_compute_multiply");
    //NemoUtils::tic(tic_toc_prefix_M_matrix_compute_multiply);

    //PetscMatrixParallel<double> * M_matrix = NULL;
    //mult(*temp_matrix,*P_matrix,&M_matrix);
    //NemoUtils::toc(tic_toc_prefix_M_matrix_compute_multiply);
    //delete temp_matrix;
    //// ======end of Method I============
    // ==================================
    // Yu: Method II  solve M from HP*M=P
    // ----------------------------------
    // to get M, we solve linear equation HP*M=P
    PetscMatrixParallel<double>* M_matrix=NULL;
    //4a.1 set up the Linear solver
    PetscMatrixParallel<double> temp_solution(P_matrix->get_num_rows(),P_matrix->get_num_cols(),this_simulation->get_simulation_domain()->get_communicator());
    temp_solution.consider_as_full();
    temp_solution.allocate_memory();

    {
      if(PropOptionInterface->get_debug_output())
      {
        HP_matrix->save_to_matlab_file("HP_matrix_issue"+this_simulation->get_name()+".m");
        P_matrix->save_to_matlab_file("P_matrix_issue"+this_simulation->get_name()+".m");
      }
      LinearSolverPetscDouble solver(*HP_matrix,*P_matrix,&temp_solution);


      //4a.2 prepare input options for the linear solver
      InputOptions options_for_linear_solver;
      string value("mumps");
      if (options.check_option("linear_system_solver"))
        value = options.get_option("linear_system_solver", string("mumps"));
      string key("solver");
      options_for_linear_solver[key] = value;
      solver.set_options(options_for_linear_solver);

      std::string tic_toc_prefix_M_matrix_compute_inverse= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()
          +"\")::transfer_matrix_get_M_matrix_step_M_matrix_solve_linear_equation");
      NemoUtils::tic(tic_toc_prefix_M_matrix_compute_inverse);
      solver.solve();
      if(PropOptionInterface->get_debug_output())
        temp_solution.save_to_matlab_file("result_issue"+this_simulation->get_name()+".m");
      NemoUtils::toc(tic_toc_prefix_M_matrix_compute_inverse);
    }
    M_matrix=&temp_solution;
    //M_matrix->save_to_matlab_file("M_matrix.m");
    // =====end of Method II======================

    delete P_matrix;
    delete HP_matrix;

    //4b. extract M1 and M2 submatrix from M
    unsigned int number_of_M1_sub_rows = number_of_00_rows+number_of_11_rows;
    unsigned int number_of_M2_sub_rows = number_of_22_rows+number_of_33_rows;
    unsigned int number_of_M1_sub_cols = number_of_P_cols;
    unsigned int number_of_M2_sub_cols = number_of_M1_sub_cols;
    unsigned int number_of_M1_local_rows = number_of_00_local_rows+number_of_11_local_rows;
    unsigned int number_of_M2_local_rows = number_of_22_local_rows+number_of_33_local_rows;
    unsigned int start_M2_rows = start_own_22_rows+start_own_33_rows;

    //consider as full matrix -- get_submatrix not work in parallel
    vector<int> M1_rows_diagonal(number_of_M1_local_rows,0);
    vector<int> M1_rows_offdiagonal(number_of_M1_local_rows,0);

    std::string tic_toc_prefix_number_of_M1_local_rows = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix:step_number_of_M1_local_rows");
    NemoUtils::tic(tic_toc_prefix_number_of_M1_local_rows);

    for(unsigned int i=0; i<number_of_M1_local_rows; i++)
    {
      M1_rows_diagonal[i]=number_of_M2_sub_rows;
      M1_rows_offdiagonal[i]=0;
    }
    NemoUtils::toc(tic_toc_prefix_number_of_M1_local_rows);

    std::string tic_toc_prefix_number_of_M2_local_rows = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix:step_number_of_M2_local_rows");
    NemoUtils::tic(tic_toc_prefix_number_of_M2_local_rows);

    vector<int> M2_rows_diagonal(number_of_M2_local_rows,0);
    vector<int> M2_rows_offdiagonal(number_of_M2_local_rows,0);
    for(unsigned int i=0; i<number_of_M2_local_rows; i++)
    {
      M2_rows_diagonal[i]=number_of_M2_sub_rows;
      M2_rows_offdiagonal[i]=0;
    }
    NemoUtils::toc(tic_toc_prefix_number_of_M2_local_rows);

    std::string tic_toc_prefix_M_matrix_get_submatrix= NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_M_matrix_step_M_matrix_get_submatrix");
    NemoUtils::tic(tic_toc_prefix_M_matrix_get_submatrix);
    //M1 is the upper right block of M
    transfer_matrix_get_submatrix_double(this_simulation,number_of_M1_sub_rows,number_of_M2_sub_rows,0,0,M1_rows_diagonal,M1_rows_offdiagonal,M_matrix, M1_matrix);

    //M2 is the lower right block of M
    if(LRA)
    {
      //set M2
      M2_matrix = new PetscMatrixParallel<double>(number_of_M2_sub_rows,number_of_M2_sub_cols,
          this_simulation->get_simulation_domain()->get_communicator() );
      M2_matrix->set_num_owned_rows(number_of_M2_local_rows);
      for(unsigned i=0; i<number_of_M2_local_rows; i++)
        M2_matrix->set_num_nonzeros(i+start_M2_rows,number_of_M2_sub_cols,0);
      M2_matrix->allocate_memory();
      M2_matrix->set_to_zero();

      const double* pointer_to_data= NULL;
      double temp_val=0.0;
      vector<double> data_vector;
      vector<int> col_index;
      int n_nonzeros = 0;
      const int* n_col_nums;
      for(unsigned int i=start_M2_rows; i<start_M2_rows+number_of_M2_local_rows; i++)
      {
        M_matrix->get_row(i+number_of_M1_local_rows,&n_nonzeros,n_col_nums,pointer_to_data);
        col_index.resize(number_of_M2_sub_cols,0);
        data_vector.resize(number_of_M2_sub_cols,0.0);
        for(unsigned int j=0; j<number_of_M2_sub_cols; j++)
        {
          col_index[j]=j;
          temp_val=pointer_to_data[j];
          data_vector[j]=temp_val;
        }
        M2_matrix->set(i,col_index,data_vector);
        M_matrix->store_row(i+number_of_M1_local_rows,&n_nonzeros,n_col_nums,pointer_to_data);
      }
      M2_matrix->assemble();
    }
    else
      transfer_matrix_get_submatrix_double(this_simulation,number_of_M2_sub_rows,number_of_M2_sub_rows,number_of_M1_sub_rows,0,M2_rows_diagonal,M2_rows_offdiagonal,
                                           M_matrix, M2_matrix);

    NemoUtils::toc(tic_toc_prefix_M_matrix_get_submatrix);
  }
  NemoUtils::toc(tic_toc_prefix_M_matrix);
  NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::transfer_matrix_solve_eigenvalues(Simulation* this_simulation,PetscMatrixParallelComplex*& M_matrix,
    const bool flagLRA,
    const double shift,
    const int number_of_set_num,
    vector<std::complex<double> >& M_values,
    vector< vector< std::complex<double> > >& M_vectors,
    unsigned int* number_of_eigenvalues, unsigned int* number_of_vectors_size)
{
  //solve eigenvalue of M2 matrix
  //if LRA solve Krylovshcur or CISS, else lapack 
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues ");
  NemoUtils::tic(tic_toc_prefix);
  std::string tic_toc_prefix2 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues creating variables ");
  NemoUtils::tic(tic_toc_prefix2);
  const InputOptions& options=this_simulation->get_options();
  EigensolverSlepc solver;

  //setting options for eigensolver
  string transform_type=options.get_option("transform_type",string("none"));
  string solver_type=options.get_option("solver_type",string("NEMOLapack"));
  string precond_type=options.get_option("precond_type",string(""));
  string linsolver=options.get_option("linsolver",string(""));
  int ncv_number=options.get_option("ncv",int(2*number_of_set_num));
  unsigned int max_number_iterations = options.get_option("max_number_iterations",300);
  double max_error = options.get_option("convergence_limit",1e-8);
  bool monitor_convergence = options.get_option("monitor_convergence", bool(false));

  //other options for CISS
  double center = options.get_option("region_center", 0.0);
  double radius = options.get_option("region_radius", 1.01);
  double vscale = options.get_option("region_vertical_scale", 1.0);
  double radius_inner = options.get_option("region_inner_radius", .99);
  int ip = options.get_option("number_of_integration_points", 16);
  int bs = options.get_option("block_size", 16);
  int ms = options.get_option("moment_size", 16);
  int npart = options.get_option("real_space_partitions", 1);
  int bsmax = options.get_option("block_size_max", 16);
  int isreal = 0; //Petsbool is really an int in C
  int savelu = options.get_option("save_lu", bool(true));
  int isdense = options.get_option("isdense", bool(false));
  int isring = options.get_option("isring", bool(false));
  int refine_inner = options.get_option("refine_inner_loop_iterations", 2);
  int refine_outer = options.get_option("refine_outer_loop_iterations", 0);
  int refine_blsize = options.get_option("refine_block_size_number", 0);
  if (solver_type == "ciss")
  {
    solver.set_ciss_regions(center, radius, vscale, radius_inner);
    solver.set_ciss_sizes(ip, bs, ms, npart, bsmax, isreal,
                          savelu, isdense, isring);
    solver.set_ciss_refinements(refine_inner, refine_outer, refine_blsize);
  }

  NemoUtils::toc(tic_toc_prefix2);
  solver.set_shift_sigma(shift);//energy
  solver.set_num_evals(number_of_set_num);

  if(flagLRA==true) //LRA
  {
    std::string tic_toc_prefix3 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues flagLRA=true ");
    NemoUtils::tic(tic_toc_prefix3);
    solver.set_matrix(M_matrix);
    solver.set_problem_type("eps_non_hermitian");
    solver.set_convergence_params(max_error, max_number_iterations);
    solver.monitor_convergence(monitor_convergence);
    solver.set_transformation_type(transform_type);
    solver.set_precond_type(precond_type);
    solver.set_ncv(ncv_number);
    solver.set_solver_type(solver_type);
    solver.set_linsolver_type(linsolver);
    solver.set_shift_sigma(shift);
    solver.set_spectral_region(EigensolverSlepc::target_real);
    NemoUtils::toc(tic_toc_prefix3);
  }
  else //exact solution
  {
    std::string tic_toc_prefix4 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues flagLRA=false ");
    NemoUtils::tic(tic_toc_prefix4);
    solver.set_matrix(M_matrix);
    solver.set_problem_type("eps_non_hermitian");
    solver.set_transformation_type("none");
    solver.set_solver_type(solver_type);
    solver.set_spectral_region(EigensolverSlepc::smallest_real);
    NemoUtils::toc(tic_toc_prefix4);
  }

  std::string tic_toc_prefix5b = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues_solver_(d)(z)geev ");
  NemoUtils::tic(tic_toc_prefix5b);
  solver.solve();
  NemoUtils::toc(tic_toc_prefix5b);

  std::string tic_toc_prefix6 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues getting eigenvalues ");
  NemoUtils::tic(tic_toc_prefix6);

  M_values = *(solver.get_eigenvalues());
  *number_of_eigenvalues = M_values.size();
  msg<< "Number of eigenvalues = " << *number_of_eigenvalues << endl;
  M_vectors.clear();
  M_vectors.resize( *number_of_eigenvalues );
  for (unsigned int i = 0; i <  *number_of_eigenvalues; i++)
  {
    solver.get_eigenvector(i).get_local_part(M_vectors[i]);
  }
  *number_of_vectors_size = M_vectors[0].size();
  NemoUtils::toc(tic_toc_prefix6);


  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_solve_eigenvalues_double(Simulation* this_simulation,PetscMatrixParallel<double>*& M_matrix,
    const bool flagLRA,
    const double shift,
    const int number_of_set_num,
    vector<std::complex<double> >& M_values,
    vector< PetscVectorNemo<double> >& M_vectors_real,
    vector< PetscVectorNemo<double> >& M_vectors_imag,
    unsigned int* number_of_eigenvalues, unsigned int* number_of_vectors_size)
{
  //solve eigenvalue of M2 matrix -- double version
  //if LRA solve Krylovshcur or CISS, else lapack 
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues ");
  NemoUtils::tic(tic_toc_prefix);
  std::string tic_toc_prefix2 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues creating variables ");
  NemoUtils::tic(tic_toc_prefix2);
  const InputOptions& options=this_simulation->get_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  EigensolverSlepcDouble solver;

  //setting options for eigensolver
  string transform_type=options.get_option("transform_type",string("none"));
  string solver_type=options.get_option("solver_type",string("NEMOLapack"));
  string precond_type=options.get_option("precond_type",string(""));
  string linsolver=options.get_option("linsolver",string(""));
  int ncv_number=options.get_option("ncv",int(2*number_of_set_num));
  unsigned int max_number_iterations = options.get_option("max_number_iterations",300);
  double max_error = options.get_option("convergence_limit",1e-8);
  bool monitor_convergence = options.get_option("monitor_convergence", bool(false));

  NemoUtils::toc(tic_toc_prefix2);

  //other options for CISS
  //double center = options.get_option("region_center", 0.0);
  //double radius = options.get_option("region_radius", 1.01);
  //double vscale = options.get_option("region_vertical_scale", 1.0);
  //double radius_inner = options.get_option("region_inner_radius", .99);
  //int ip = options.get_option("number_of_integration_points", 16);
  //int bs = options.get_option("block_size", 16);
  //int ms = options.get_option("moment_size", 16);
  //int npart = options.get_option("real_space_partitions", 1);
  //int bsmax = options.get_option("block_size_max", 16);
  //int isreal = 1; //Petsbool is really an int in C
  //int savelu = options.get_option("save_lu", bool(true));
  //int isdense = options.get_option("isdense", bool(false));
  //int isring = options.get_option("isring", bool(false));
  //int refine_inner = options.get_option("refine_inner_loop_iterations", 2);
  //int refine_outer = options.get_option("refine_outer_loop_iterations", 0);
  //int refine_blsize = options.get_option("refine_block_size_number", 0);
  if (solver_type == "ciss")
  {
    //solver.set_ciss_regions(center, radius, vscale, radius_inner);
    //solver.set_ciss_sizes(ip, bs, ms, npart, bsmax, isreal,
    //savelu, isdense, isring);
    //solver.set_ciss_refinements(refine_inner, refine_outer, refine_blsize);
  }

  solver.set_shift_sigma(shift);//energy
  solver.set_num_evals(number_of_set_num);

  if(flagLRA==true) //LRA
  {
    std::string tic_toc_prefix3 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues flagLRA=true ");
    NemoUtils::tic(tic_toc_prefix3);
    solver.set_matrix(M_matrix);
    solver.set_problem_type("eps_non_hermitian");
    solver.set_convergence_params(max_error, max_number_iterations);
    solver.monitor_convergence(monitor_convergence);
    solver.set_transformation_type(transform_type);
    solver.set_precond_type(precond_type);
    solver.set_ncv(ncv_number);
    solver.set_solver_type(solver_type);
    solver.set_linsolver_type(linsolver);
    solver.set_shift_sigma(shift);
    solver.set_spectral_region(EigensolverSlepcDouble::target_real);
    NemoUtils::toc(tic_toc_prefix3);
  }
  else //exact solution
  {
    std::string tic_toc_prefix4 = NEMOUTILS_PREFIX("Propagation(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues flagLRA=false ");
    NemoUtils::tic(tic_toc_prefix4);
    if(PropOptionInterface->get_debug_output())
      M_matrix->save_to_matlab_file("M_matrix_issue.m");
    solver.set_matrix(M_matrix);
    solver.set_problem_type("eps_non_hermitian");
    solver.set_transformation_type("none");
    solver.set_solver_type(solver_type);
    solver.set_spectral_region(EigensolverSlepcDouble::smallest_real);
    NemoUtils::toc(tic_toc_prefix4);
  }

  std::string tic_toc_prefix5b = NEMOUTILS_PREFIX("Propagation(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues_solver_(d)(z)geev ");
  NemoUtils::tic(tic_toc_prefix5b);
  if(PropOptionInterface->get_debug_output())
    M_matrix->save_to_matlab_file("NEMO_M_matrix.m");
  solver.solve();
  NemoUtils::toc(tic_toc_prefix5b);

  std::string tic_toc_prefix6 = NEMOUTILS_PREFIX("Propagation(\""+this_simulation->get_name()+"\")::transfer_matrix_solve_eigenvalues getting eigenvalues ");
  NemoUtils::tic(tic_toc_prefix6);

  M_values.clear();
  M_values = *(solver.get_eigenvalues());
  *number_of_eigenvalues = M_values.size();
  msg<< "Number of eigenvalues = " << *number_of_eigenvalues << endl;
  M_vectors_real.clear();
  M_vectors_imag.clear();
  for (unsigned int i = 0; i <  *number_of_eigenvalues; i++)
  {
    PetscVectorNemo<double> v_r(solver.get_eigenvector_from_slepc_solver_real_part(i));
    PetscVectorNemo<double> v_i(solver.get_eigenvector_from_slepc_solver_imag_part(i));
    M_vectors_real.push_back(v_r);
    M_vectors_imag.push_back(v_i);
  }
  *number_of_vectors_size = M_vectors_real[0].get_size();
  NemoUtils::toc(tic_toc_prefix6);


  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_get_wave_direction(Simulation* this_simulation,PetscMatrixParallelComplex*& M1_matrix, PetscMatrixParallelComplex*& T_matrix,
    vector<std::complex<double> >& M_values, vector< vector< std::complex<double> > >& M_vectors,
    unsigned int* number_of_eigenvalues, unsigned int* number_of_vectors_size, unsigned int* num_left, unsigned int* num_right)
{
  //calculate modes
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction ";

  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  const InputOptions& options=this_simulation->get_options();

  NemoPhys::transfer_matrix_type temp_transfer_type=NemoPhys::into_device_propagating_modes;
  PetscMatrixParallelComplex* into_device_propagating_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_modes;
  PetscMatrixParallelComplex* into_device_decaying_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_modes;
  PetscMatrixParallelComplex* out_of_device_propagating_modes=NULL;
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_modes;
  PetscMatrixParallelComplex* out_of_device_decaying_modes=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_phase;
  PetscMatrixParallelComplex* into_device_propagating_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_phase;
  PetscMatrixParallelComplex* into_device_decaying_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_phase;
  PetscMatrixParallelComplex* out_of_device_propagating_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_phase;
  PetscMatrixParallelComplex* out_of_device_decaying_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_modes;
  PetscMatrixParallelComplex* into_device_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_phase;
  PetscMatrixParallelComplex* into_device_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_modes;
  PetscMatrixParallelComplex* out_of_device_modes=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_phase;
  PetscMatrixParallelComplex* out_of_device_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_velocity;
  PetscMatrixParallelComplex* into_device_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_velocity;
  PetscMatrixParallelComplex* into_device_propagating_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_velocity;
  PetscMatrixParallelComplex* out_of_device_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_velocity;
  PetscMatrixParallelComplex* out_of_device_propagating_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_velocity,&temp_transfer_type);

  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);


  std::string tic_toc_prefix_intialisation = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_intialisation");
  NemoUtils::tic(tic_toc_prefix_intialisation);
  vector<unsigned int> eigflag(*number_of_eigenvalues,0);
  //left: out of device modes
  //right: into device modes
  *num_left = 0;
  *num_right = 0;
  unsigned int n_left_propagating = 0;
  unsigned int n_left_decaying = 0;
  unsigned int n_right_propagating = 0;
  unsigned int n_right_decaying = 0;
  unsigned int number_of_long_vector=0; //size of full vector
  unsigned int M1_own_rows = M1_matrix->get_num_owned_rows();

  number_of_long_vector=*number_of_vectors_size+M1_own_rows;

  PetscMatrixParallelComplex  coupling_matrix = (*T_matrix);
  (&coupling_matrix)->hermitian_transpose_matrix(coupling_matrix,MAT_REUSE_MATRIX);//T'

  double eps_limit = options.get_option("eps_limit",1e-6); //eliminate osolete solution phase factor !=0 or 1
  double imag_limit = options.get_option("imag_limit",2e-6); //controls the threshold of decaying modes

  bool cond, cond1, cond2a, cond2b, cond2c;

  NemoUtils::toc(tic_toc_prefix_intialisation);

  //===========================================================24
  //Yu: copy eigenvectors to petscmatrix, do matrix-matrix mult
  //-----------------------------------------------------------
  std::string tic_toc_prefix_for_copy_vector = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction copy vector to matrix");
  NemoUtils::tic(tic_toc_prefix_for_copy_vector);
  //1. determine which eigenvalues to take
  unsigned int num_eig_take = 0;
  for(unsigned int i=0; i<*number_of_eigenvalues; i++)
  {
    cond1 = abs(M_values[i])>eps_limit;
    cond2a = M_values[i].real()<-1.0-eps_limit;
    cond2b = M_values[i].real()>-1.0+eps_limit;
    cond2c = abs(M_values[i].imag())>eps_limit;
    cond = (cond1&&(cond2a||cond2b||cond2c));  //OMEN--Injection::get_condition()
    if(cond) //if not the osolete solution phase factor !=0 or 1
    {
      eigflag[i]=1;
      num_eig_take++;
    }
  }
  //2. copy eigenvalues and eigenvectors
  vector<std::complex<double> > eig_take(num_eig_take,cplx(0.0,0.0));
  PetscMatrixParallelComplex* eig_vector = new PetscMatrixParallelComplex(*number_of_vectors_size,num_eig_take,
      this_simulation->get_simulation_domain()->get_communicator());
  eig_vector->set_num_owned_rows(*number_of_vectors_size);
  eig_vector->consider_as_full();
  for(unsigned int i=0; i<*number_of_vectors_size; i++)
    eig_vector->set_num_nonzeros_for_local_row(i,num_eig_take,0);
  eig_vector->allocate_memory();

  unsigned int eig_index=0;
  vector<int> store_eig_index(num_eig_take,0);
  for(unsigned int i=0; i<*number_of_eigenvalues; i++)
  {
    if(eigflag[i]==1)
    {
      eig_take[eig_index]=M_values[i];
      store_eig_index[eig_index]=i;
      eig_index++;
    }
  }

  cplx* pointer_to_data= NULL;
  eig_vector->get_array(pointer_to_data);
  for(unsigned int i=0; i<num_eig_take; i++)
  {
    for(unsigned int j=0; j<*number_of_vectors_size; j++)
    {
      pointer_to_data[i*(*number_of_vectors_size)+j]=M_vectors[store_eig_index[i]][j];
    }
  }
  eig_vector->store_array(pointer_to_data);
  eig_vector->assemble();
  NemoUtils::toc(tic_toc_prefix_for_copy_vector);

  //3. matrix-matrix mult
  PetscMatrixParallelComplex* eig_vector1=NULL;
  std::string tic_toc_prefix_for_mult = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction for matrix-matrix(eigenvector) mult");
  NemoUtils::tic(tic_toc_prefix_for_mult);
  if(num_eig_take!=0)
    PetscMatrixParallelComplex::mult(*M1_matrix,*eig_vector,&eig_vector1);
  NemoUtils::toc(tic_toc_prefix_for_mult);

  //4. comine double-vector matrix into cplx-vector, and normalize
  std::string tic_toc_prefix_for_cplx = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction copy double-vector into cplx-vector");
  NemoUtils::tic(tic_toc_prefix_for_cplx);
  vector<std::complex<double> > M_vectors0(number_of_long_vector,cplx(0.0,0.0));
  vector< vector< std::complex<double> > > M_vectors_full(num_eig_take,M_vectors0);
  vector<double> vector_norm(number_of_long_vector,0.0);
  cplx* pointer_to_data1= NULL;
  if(num_eig_take!=0)
  {
    eig_vector->get_array(pointer_to_data);
    eig_vector1->get_array(pointer_to_data1);
    for (unsigned int i = 0; i < num_eig_take; i++)
    {
      for (unsigned int j= 0; j < M1_own_rows; j++)
      {
        M_vectors_full[i][j] = pointer_to_data1[i*M1_own_rows+j]/eig_take[i];
        vector_norm[i]+=std::abs(M_vectors_full[i][j]*M_vectors_full[i][j]);
      }
      for (unsigned int j= 0; j < *number_of_vectors_size; j++)
      {
        M_vectors_full[i][j+M1_own_rows] = pointer_to_data[i*(*number_of_vectors_size)+j];
        vector_norm[i]+=std::abs(M_vectors_full[i][j+M1_own_rows]*M_vectors_full[i][j+M1_own_rows]);
      }
    }
    eig_vector1->store_array(pointer_to_data1);
    eig_vector->store_array(pointer_to_data);
  }
  for(unsigned int i=0; i<num_eig_take; i++)
    for(unsigned int j=0; j<number_of_long_vector; j++)
      M_vectors_full[i][j]=M_vectors_full[i][j]/std::sqrt(vector_norm[i]);
  NemoUtils::toc(tic_toc_prefix_for_cplx);
  delete eig_vector;
  if(eig_vector1!=NULL)
    delete eig_vector1;
  //===========================================================


  //5. loop over eigenvalues, determine mode direction
  vector<unsigned int> flag(num_eig_take,0);
  vector<std::complex<double> > expmikdelta(num_eig_take,cplx(0.0,0.0));
  vector<std::complex<double> > kdelta(num_eig_take,cplx(0.0,0.0));
  vector<double> velocity_to_store(num_eig_take,0.0);

  std::string tic_toc_prefix_for_loop = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_for_loop");
  NemoUtils::tic(tic_toc_prefix_for_loop);
  for(unsigned int i=0; i<num_eig_take; i++)
  {
    flag[i]=1;
    std::complex<double>  inv_M_values(0.0,0.0);

    cond1 = abs(eig_take[i])>eps_limit;
    cond2a = eig_take[i].real()<-1.0-eps_limit;
    cond2b = eig_take[i].real()>-1.0+eps_limit;
    cond2c = abs(eig_take[i].imag())>eps_limit;
    cond = (cond1&&(cond2a||cond2b||cond2c));  //OMEN--Injection::get_condition()
    if(cond) //if not the osolete solution phase factor !=0 or 1
    {
      inv_M_values=cplx(1.0,0.0)/eig_take[i];
      expmikdelta[i]=cplx(1.0,0.0)+inv_M_values;
      kdelta[i]=cplx(0.0,1.0)*log(expmikdelta[i]); //Yu: correct version

      vector<std::complex<double> > M_vectors1(M_vectors_full[i]);

      std::complex<double> velocity_temp(0.0,0.0);

      if((kdelta[i].imag()<imag_limit)&&(kdelta[i].imag()>-imag_limit))  //OMEN--Injection::get_type()
      {
        //propagating modes
        double velocity=0.0;
        vector<std::complex<double> > vectors_temp(number_of_long_vector,cplx(0.0,0.0));
        mult(coupling_matrix,M_vectors1,vectors_temp);
        for(unsigned int j=0; j<number_of_long_vector; j++)
          velocity_temp+=conj(M_vectors1[j])*vectors_temp[j];
        velocity_temp/=expmikdelta[i];
        velocity=2*velocity_temp.imag();//velocity of waves -- Tillmann: change - to + in front
        velocity_to_store[i]=velocity;
        if(velocity<0.0)
        {
          flag[i]=5; //reflected propagating wave (from device to lead)
          (*num_left)++;
          n_left_propagating++;
        }
        else
        {
          flag[i]=6; //transmitted propagating wave (from lead to device)
          (*num_right)++;
          n_right_propagating++;
        }
      }
      else
      {
        //decaying modes
        if(kdelta[i].imag()<-imag_limit)
        {
          flag[i]=3;//reflected decaying wave (from device to lead)
          (*num_left)++;
          n_left_decaying++;
        }
        else
        {
          flag[i]=4;//transmitted decaying wave (from lead to device)
          (*num_right)++;
          n_right_decaying++;
        }
      }
    }
    else { } //end of if(cond)...else
  } //end of for loop
  NemoUtils::toc(tic_toc_prefix_for_loop);

  //test flag
  std::string tic_toc_prefix_init_temp_matrix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_init_temp_matrix");
  NemoUtils::tic(tic_toc_prefix_init_temp_matrix);

  bool output_eigen=options.get_option("output_eigenvalues", bool(false));
  if(output_eigen&&!PropOptionInterface->get_no_file_output())
  {
    ofstream outfile;
    std::string filename=this_simulation->get_name()+"complex_bandstructure.dat";
    outfile.open(filename.c_str(),std::ofstream::out | std::ofstream::app);
    for(unsigned int i=0; i<num_eig_take; i++)
    {
      msg<<eig_take[i].real()<<"\t"<<eig_take[i].imag()<<"\t"<<expmikdelta[i].real()<<"\t"<<expmikdelta[i].imag()<<"\t"<<flag[i]<<"\t"<<velocity_to_store[i]<<"\n";
      outfile<<eig_take[i].real()<<"\t"<<eig_take[i].imag()<<"\n";
    }
    outfile<<"---------------------------\n";
    outfile.close();
  }

  //if(!LRA)
  *number_of_vectors_size=number_of_long_vector;
  vector<int> nz_per_owned_wave_left_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_per_owned_wave_left_rows_offdiagonal(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_left_rows_diagonal_velocity(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_left_rows_offdiagonal_velocity(*number_of_vectors_size,0); //YH only out of device modes
  vector<int> nz_per_owned_exp_left_rows_diagonal_velocity(*num_left,0); //Bozidar commented above two lines and inserted these
  vector<int> nz_per_owned_exp_left_rows_offdiagonal_velocity(*num_left,0);
  vector<int> nz_per_owned_exp_left_rows_diagonal(*num_left,0);
  vector<int> nz_per_owned_exp_left_rows_offdiagonal(*num_left,0); //YH only out of device modes

  for(unsigned int i=0; i<*number_of_vectors_size; i++)
  {
    nz_per_owned_wave_left_rows_diagonal[i]=*num_left;
    nz_per_owned_wave_left_rows_offdiagonal[i]=0;
  }
  for(unsigned int i=0; i<*num_left; i++)
  {
    nz_per_owned_exp_left_rows_diagonal_velocity[i]=1;
    nz_per_owned_exp_left_rows_offdiagonal_velocity[i]=0;
    nz_per_owned_exp_left_rows_diagonal[i]=1;
    nz_per_owned_exp_left_rows_offdiagonal[i]=0;
  }
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,*num_left,nz_per_owned_wave_left_rows_diagonal,nz_per_owned_wave_left_rows_offdiagonal,
      out_of_device_modes); //modes are dense
  transfer_matrix_initialize_temp_matrix(this_simulation,*num_left,*num_left,
                                         nz_per_owned_exp_left_rows_diagonal,nz_per_owned_exp_left_rows_offdiagonal,out_of_device_phase);//Yu: this is diagonal matrix, should not be full

  int n_right = n_right_propagating+n_right_decaying;
  vector<int> nz_per_owned_wave_right_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_per_owned_wave_right_rows_offdiagonal(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_right_rows_diagonal_velocity(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_right_rows_offdiagonal_velocity(*number_of_vectors_size,0);//YH only into device modes
  vector<int> nz_per_owned_exp_right_rows_diagonal_velocity(n_right,0); //Bozidar commented above two lines and inserted these
  vector<int> nz_per_owned_exp_right_rows_offdiagonal_velocity(n_right,0);
  vector<int> nz_per_owned_exp_right_rows_diagonal(n_right,0);
  vector<int> nz_per_owned_exp_right_rows_offdiagonal(n_right,0);//YH only into device modes

  for(unsigned int i=0; i<*number_of_vectors_size; i++)
  {
    nz_per_owned_wave_right_rows_diagonal[i]=n_right;
    nz_per_owned_wave_right_rows_offdiagonal[i]=0;
  }
  for(int i=0; i<n_right; i++)
  {
    nz_per_owned_exp_right_rows_diagonal_velocity[i]=1;
    nz_per_owned_exp_right_rows_offdiagonal_velocity[i]=0;
    nz_per_owned_exp_right_rows_diagonal[i]=1;
    nz_per_owned_exp_right_rows_offdiagonal[i]=0;
  }
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,n_right,nz_per_owned_wave_right_rows_diagonal,nz_per_owned_wave_right_rows_offdiagonal,
      into_device_modes); // modes are dense
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right,n_right,
                                         nz_per_owned_exp_right_rows_diagonal,nz_per_owned_exp_right_rows_offdiagonal,into_device_phase); //Yu: this is diagonal matrix, should not be full

  //collect waves and phase factor for QTBM
  vector<int> nz_left_propagating_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_left_propagating_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_left_decaying_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_left_decaying_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_right_propagating_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_right_propagating_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_right_decaying_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_right_decaying_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_left_propagating_phase_rows_diagonal(n_left_propagating,0);
  vector<int> nz_left_propagating_phase_rows_offdiagonal(n_left_propagating,0);
  vector<int> nz_left_decaying_phase_rows_diagonal(n_left_decaying,0);
  vector<int> nz_left_decaying_phase_rows_offdiagonal(n_left_decaying,0);
  vector<int> nz_right_propagating_phase_rows_diagonal(n_right_propagating,0);
  vector<int> nz_right_propagating_phase_rows_offdiagonal(n_right_propagating,0);
  vector<int> nz_right_decaying_phase_rows_diagonal(n_right_decaying,0);
  vector<int> nz_right_decaying_phase_rows_offdiagonal(n_right_decaying,0);

  for(unsigned int i=0; i<*number_of_vectors_size; i++)
  {
    nz_left_propagating_wave_rows_diagonal[i]=n_left_propagating;
    nz_left_propagating_wave_rows_offdiagonal[i]=0;
    nz_left_decaying_wave_rows_diagonal[i]=n_left_decaying;
    nz_left_decaying_wave_rows_offdiagonal[i]=0;
    nz_right_propagating_wave_rows_diagonal[i]=n_right_propagating;
    nz_right_propagating_wave_rows_offdiagonal[i]=0;
    nz_right_decaying_wave_rows_diagonal[i]=n_right_decaying;
    nz_right_decaying_wave_rows_offdiagonal[i]=0;
  }
  for(unsigned int i=0; i<n_left_propagating; i++)
  {
    nz_left_propagating_phase_rows_diagonal[i]=1;
    nz_left_propagating_phase_rows_offdiagonal[i]=0;
  }
  for(unsigned int i=0; i<n_right_propagating; i++)
  {
    nz_right_propagating_phase_rows_diagonal[i]=1;
    nz_right_propagating_phase_rows_offdiagonal[i]=0;
  }
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,n_left_propagating,nz_left_propagating_wave_rows_diagonal,
      nz_left_propagating_wave_rows_offdiagonal,
      out_of_device_propagating_modes); //Yu: make it dense, seems necessary for avoiding reordering
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,n_right_propagating,nz_right_propagating_wave_rows_diagonal,
      nz_right_propagating_wave_rows_offdiagonal,
      into_device_propagating_modes); //Yu: make it dense, seems necessary for avoiding reordering
  transfer_matrix_initialize_temp_matrix(this_simulation,n_left_propagating,n_left_propagating,
                                         nz_left_propagating_phase_rows_diagonal,nz_left_propagating_phase_rows_offdiagonal,out_of_device_propagating_phase);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_left_propagating,n_left_propagating,
                                         nz_left_propagating_phase_rows_diagonal,nz_left_propagating_phase_rows_offdiagonal,out_of_device_propagating_velocity);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right_propagating,n_right_propagating,
                                         nz_right_propagating_phase_rows_diagonal,nz_right_propagating_phase_rows_offdiagonal,into_device_propagating_velocity);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right_propagating,n_right_propagating,
                                         nz_right_propagating_phase_rows_diagonal,nz_right_propagating_phase_rows_offdiagonal,into_device_propagating_phase);
  //transfer_matrix_initialize_temp_matrix(*number_of_vectors_size,*number_of_vectors_size,
  //                                       nz_per_owned_exp_left_rows_diagonal_velocity,nz_per_owned_exp_left_rows_offdiagonal_velocity,out_of_device_velocity);
  //transfer_matrix_initialize_temp_matrix(*number_of_vectors_size,*number_of_vectors_size,
  //                                       nz_per_owned_exp_right_rows_diagonal_velocity,nz_per_owned_exp_right_rows_offdiagonal_velocity,into_device_velocity);
  //Bozidar: the two lines above have incorrect total matrix sizes (first two arguments).
  transfer_matrix_initialize_temp_matrix(this_simulation,*num_left,*num_left,
    nz_per_owned_exp_left_rows_diagonal_velocity,nz_per_owned_exp_left_rows_offdiagonal_velocity,out_of_device_velocity);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right,n_right,
    nz_per_owned_exp_right_rows_diagonal_velocity,nz_per_owned_exp_right_rows_offdiagonal_velocity,into_device_velocity);

  NemoUtils::toc(tic_toc_prefix_init_temp_matrix);

  std::string tic_toc_prefix_collect_waves_phase = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_collect_waves_phase");
  NemoUtils::tic(tic_toc_prefix_collect_waves_phase);

  //set elements
  unsigned int j=0;
  unsigned int k=0;
  unsigned int m=0;
  unsigned int n=0;
  cplx* pointer_modes_left=NULL;
  cplx* pointer_modes_right=NULL;
  cplx* pointer_modes_left_p=NULL;
  cplx* pointer_modes_right_p=NULL;
  out_of_device_modes->get_array(pointer_modes_left);
  into_device_modes->get_array(pointer_modes_right);
  out_of_device_propagating_modes->get_array(pointer_modes_left_p);
  into_device_propagating_modes->get_array(pointer_modes_right_p);

  if(options.get_option("consistent_modes",false))
  {//Bozidar: sort propagating modes by velocities and decaying by phase factor's magnitude. If phase factor's magnitude is equal then by phase factor's phase.

    std::vector<int> into_device_modes_input_index;
    std::vector<int> into_device_modes_input_index_p;
    std::vector<int> into_device_modes_index;
    std::vector<int> into_device_modes_index_p;
    std::vector<int> out_of_device_modes_input_index;
    std::vector<int> out_of_device_modes_input_index_p;
    std::vector<int> out_of_device_modes_index;
    std::vector<int> out_of_device_modes_index_p;

    //Set input vector sizes for vectors necesary for sorting utility function.
    for(unsigned int i = 0; i < num_eig_take; i++)
    {
      if(flag[i] == 3) //reflected decaying waves
        j++;
      if(flag[i] == 5) //reflected propagating waves
        m++;
      if(flag[i] == 4) //transmitted decaying waves
        k++;
      if(flag[i] == 6) //transmitted propagating waves
        n++;
    }
    out_of_device_modes_input_index_p.resize(m);
    out_of_device_modes_input_index.resize(j);
    into_device_modes_input_index_p.resize(n);
    into_device_modes_input_index.resize(k);

    j = 0;
    k = 0;
    m = 0;
    n = 0;
    //Find input indices necessary for sorting utility function.
    for(unsigned int i = 0; i < num_eig_take; i++)
    {
      if(flag[i] == 3)
        out_of_device_modes_input_index[j++] = i;
      if(flag[i] == 5)
        out_of_device_modes_input_index_p[m++] = i;
      if(flag[i] == 4)
        into_device_modes_input_index[k++] = i;
      if(flag[i] == 6)
        into_device_modes_input_index_p[n++] = i;
    }

    //Make complex velocity vector to comply with sorting function API.
    std::vector<std::complex<double> > complex_velocity_to_store(velocity_to_store.size(),std::complex<double>(0.0,0.0));
    for(unsigned int i = 0; i < velocity_to_store.size(); i++)
      complex_velocity_to_store[i].real() = velocity_to_store[i];

    //Run sorting function and obtain necesssary reordered indices.
    NemoUtils::sort_abs_then_phase(complex_velocity_to_store,out_of_device_modes_input_index_p,out_of_device_modes_index_p);
    NemoUtils::sort_abs_then_phase(expmikdelta,out_of_device_modes_input_index,out_of_device_modes_index);
    NemoUtils::sort_abs_then_phase(complex_velocity_to_store,into_device_modes_input_index_p,into_device_modes_index_p);
    NemoUtils::sort_abs_then_phase(expmikdelta,into_device_modes_input_index,into_device_modes_index);

    //Create mapping between original and ordered indices (key is new ordering, value is original ordering).
    std::map<int,int> map_out_of_device_modes_index_p;
    for(unsigned int i = 0; i < out_of_device_modes_input_index_p.size(); i++)
      map_out_of_device_modes_index_p[out_of_device_modes_index_p[i]] = out_of_device_modes_input_index_p[i];
    std::map<int,int> map_out_of_device_modes_index;
    for(unsigned int i = 0; i < out_of_device_modes_input_index.size(); i++)
      map_out_of_device_modes_index[out_of_device_modes_index[i]] = out_of_device_modes_input_index[i];
    std::map<int,int> map_into_device_modes_index_p;
    for(unsigned int i = 0; i < into_device_modes_input_index_p.size(); i++)
      map_into_device_modes_index_p[into_device_modes_index_p[i]] = into_device_modes_input_index_p[i];
    std::map<int,int> map_into_device_modes_index;
    for(unsigned int i = 0; i < into_device_modes_input_index.size(); i++)
      map_into_device_modes_index[into_device_modes_index[i]] = into_device_modes_input_index[i];

    //Copy complex velocity vector back to real vector.
    for(unsigned int i = 0; i < velocity_to_store.size(); i++)
      velocity_to_store[i] = complex_velocity_to_store[i].real();

    //Fill into and out of device petsc matrices with ordered and normalized modes.
    j = 0;
    k = 0;
    m = 0;
    n = 0;
    //WARNING: The following relies on the fact that mode i corresponds to a mode determined by j,k,m,n in one of separate into and out of matrices.
    for(unsigned int i = 0; i < num_eig_take; i++)
    {
      if(flag[i] == 3) //reflected decaying waves
      {
        out_of_device_velocity->set(j,j,velocity_to_store[i]); //no need to sort since zero.
        out_of_device_phase->set(j,j,expmikdelta[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left[j*(*number_of_vectors_size) + l] = M_vectors_full[map_out_of_device_modes_index[j - m]][l];
        j++;
      }

      if(flag[i] == 5) //reflected propagating waves
      {
        out_of_device_velocity->set(j,j,velocity_to_store[i]); //already sorted.
        out_of_device_phase->set(j,j,expmikdelta[map_out_of_device_modes_index_p[m]]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left[j*(*number_of_vectors_size) + l] = M_vectors_full[map_out_of_device_modes_index_p[m]][l];
        j++;

        out_of_device_propagating_phase->set(m,m,expmikdelta[map_out_of_device_modes_index_p[m]]);
        out_of_device_propagating_velocity->set(m,m,velocity_to_store[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left_p[m*(*number_of_vectors_size)+l] = M_vectors_full[map_out_of_device_modes_index_p[m]][l];
        m++;
      }

      if(flag[i] == 4) //transmitted decaying waves
      {
        into_device_velocity->set(k,k,velocity_to_store[i]); //no need to sort since zero.
        into_device_phase->set(k,k,expmikdelta[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right[k*(*number_of_vectors_size) + l] = M_vectors_full[map_into_device_modes_index[k-n]][l];
        k++;
      }
      if(flag[i] == 6) //transmitted propagating waves
      {
        into_device_velocity->set(k,k,velocity_to_store[i]); //already sorted.
        into_device_phase->set(k,k,expmikdelta[map_into_device_modes_index_p[n]]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right[k*(*number_of_vectors_size) + l] = M_vectors_full[map_into_device_modes_index_p[n]][l];
        k++;

        into_device_propagating_phase->set(n,n,expmikdelta[map_into_device_modes_index_p[n]]);
        into_device_propagating_velocity->set(n,n,velocity_to_store[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right_p[n*(*number_of_vectors_size)+l] = M_vectors_full[map_into_device_modes_index_p[n]][l];
        n++;
      }
    }
  }
  else
  {
    for(unsigned int i = 0; i<num_eig_take; i++)
    {
      if(flag[i] == 3 || flag[i] == 5) //reflected waves
      {
        out_of_device_velocity->set(j,j,velocity_to_store[i]);
        out_of_device_phase->set(j,j,expmikdelta[i]);
        for(unsigned int l = 0; l<*number_of_vectors_size; l++)
          pointer_modes_left[j*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        j++;
      }
      if(flag[i] == 4 || flag[i] == 6) //transmitted waves
      {
        into_device_velocity->set(k,k,velocity_to_store[i]);
        into_device_phase->set(k,k,expmikdelta[i]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right[k*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        k++;
      }
      if(flag[i] == 5) //reflected propagating waves
      {
        out_of_device_propagating_phase->set(m,m,expmikdelta[i]);
        out_of_device_propagating_velocity->set(m,m,velocity_to_store[i]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left_p[m*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        m++;
      }
      if(flag[i] == 6) //transmitted propagating waves
      {
        into_device_propagating_phase->set(n,n,expmikdelta[i]);
        into_device_propagating_velocity->set(n,n,velocity_to_store[i]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right_p[n*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        n++;
      }
    }
  }

  into_device_propagating_modes->store_array(pointer_modes_right_p);
  out_of_device_propagating_modes->store_array(pointer_modes_left_p);
  into_device_modes->store_array(pointer_modes_right);
  out_of_device_modes->store_array(pointer_modes_left);
  out_of_device_modes->assemble();
  into_device_modes->assemble();
  out_of_device_phase->assemble();
  into_device_phase->assemble();
  out_of_device_propagating_modes->assemble();
  out_of_device_propagating_phase->assemble();
  into_device_propagating_modes->assemble();
  into_device_propagating_phase->assemble();
  out_of_device_velocity->assemble();
  into_device_velocity->assemble();
  out_of_device_propagating_velocity->assemble();
  into_device_propagating_velocity->assemble();

  
  temp_transfer_type=NemoPhys::into_device_propagating_modes;
  PropInterface->set_transfer_matrix_element(into_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_modes;
  PropInterface->set_transfer_matrix_element(into_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_decaying_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_phase;
  PropInterface->set_transfer_matrix_element(into_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_decaying_phase;
  PropInterface->set_transfer_matrix_element(into_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_decaying_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_decaying_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_modes;
  PropInterface->set_transfer_matrix_element(into_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_phase;
  PropInterface->set_transfer_matrix_element(into_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_velocity;
  PropInterface->set_transfer_matrix_element(into_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_velocity;
  PropInterface->set_transfer_matrix_element(into_device_propagating_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_velocity;
  PropInterface->set_transfer_matrix_element(out_of_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_velocity;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_velocity,&temp_transfer_type);

  NemoUtils::toc(tic_toc_prefix_collect_waves_phase);
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_get_wave_direction_double(Simulation* this_simulation,PetscMatrixParallel<double>*& M1_matrix, PetscMatrixParallelComplex*& T_matrix,
    vector<std::complex<double> >& M_values, vector< PetscVectorNemo<double> >& M_vectors_real,
    vector< PetscVectorNemo<double> >& M_vectors_imag, unsigned int* number_of_eigenvalues,
    unsigned int* number_of_vectors_size, unsigned int* num_left, unsigned int* num_right)
{
  //calculate modes -- double version
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction ";

  const InputOptions& options=this_simulation->get_options();
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);

  NemoPhys::transfer_matrix_type temp_transfer_type=NemoPhys::into_device_modes;
  PetscMatrixParallelComplex* into_device_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_phase;
  PetscMatrixParallelComplex* into_device_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_modes;
  PetscMatrixParallelComplex* out_of_device_modes=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_phase;
  PetscMatrixParallelComplex* out_of_device_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_velocity;
  PetscMatrixParallelComplex* into_device_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_velocity;
  PetscMatrixParallelComplex* into_device_propagating_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_modes;
  PetscMatrixParallelComplex* out_of_device_propagating_modes=NULL;
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_phase;
  PetscMatrixParallelComplex* out_of_device_propagating_phase=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_velocity;
  PetscMatrixParallelComplex* out_of_device_propagating_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_propagating_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_phase;
  PetscMatrixParallelComplex* into_device_propagating_phase=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_modes;
  PetscMatrixParallelComplex* into_device_propagating_modes=NULL; 
  PropInterface->get_transfer_matrix_element(into_device_propagating_modes,&temp_transfer_type);
  
  temp_transfer_type=NemoPhys::out_of_device_velocity;
  PetscMatrixParallelComplex* out_of_device_velocity=NULL; 
  PropInterface->get_transfer_matrix_element(out_of_device_velocity,&temp_transfer_type);
  
  std::string tic_toc_prefix_intialisation = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_intialisation");
  NemoUtils::tic(tic_toc_prefix_intialisation);
  vector<unsigned int> eigflag(*number_of_eigenvalues,0);
  //left out of device modes
  //right into device modes
  *num_left = 0;
  *num_right = 0;
  unsigned int n_left_propagating = 0;
  unsigned int n_left_decaying = 0;
  unsigned int n_right_propagating = 0;
  unsigned int n_right_decaying = 0;
  unsigned int number_of_long_vector=0;
  unsigned int M1_own_rows =M1_matrix->get_num_owned_rows();
  number_of_long_vector=*number_of_vectors_size+M1_own_rows;

  PetscMatrixParallelComplex  coupling_matrix = (*T_matrix);
  (&coupling_matrix)->hermitian_transpose_matrix(coupling_matrix,MAT_REUSE_MATRIX);//T'

  double eps_limit = options.get_option("eps_limit",1e-6); //eliminate osolete solution phase factor !=0 or 1 
  double imag_limit = options.get_option("imag_limit",2e-6); //controls the threshold of decaying modes

  bool cond, cond1, cond2a, cond2b, cond2c;

  NemoUtils::toc(tic_toc_prefix_intialisation);


  //===========================================================
  //Yu: copy eigenvectors to petscmatrix, do matrix-matrix mult
  //-----------------------------------------------------------
  std::string tic_toc_prefix_for_copy_vector = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction copy vector to matrix");
  NemoUtils::tic(tic_toc_prefix_for_copy_vector);
  //1. determine which eigenvalues to take
  unsigned int num_eig_take = 0;
  for(unsigned int i=0; i<*number_of_eigenvalues; i++)
  {
    cond1 = abs(M_values[i])>eps_limit;
    cond2a = M_values[i].real()<-1.0-eps_limit;
    cond2b = M_values[i].real()>-1.0+eps_limit;
    cond2c = abs(M_values[i].imag())>eps_limit;
    cond = (cond1&&(cond2a||cond2b||cond2c));  //OMEN--Injection::get_condition()
    if(cond) //if not the osolete solution phase factor !=0 or 1
    {
      eigflag[i]=1;
      num_eig_take++;
    }
  }

  //2. copy eigenvalues and eigenvectors
  vector<std::complex<double> > eig_take(num_eig_take,cplx(0.0,0.0));
  PetscMatrixParallel<double>* eig_vector = new PetscMatrixParallel<double>(*number_of_vectors_size,num_eig_take,
      this_simulation->get_simulation_domain()->get_communicator());
  eig_vector->set_num_owned_rows(*number_of_vectors_size);
  eig_vector->consider_as_full();
  for(unsigned int i=0; i<*number_of_vectors_size; i++)
    eig_vector->set_num_nonzeros_for_local_row(i,num_eig_take,0);
  eig_vector->allocate_memory();

  unsigned int eig_index=0;
  vector<int> store_eig_index(num_eig_take,0);
  for(unsigned int i=0; i<*number_of_eigenvalues; i++)
  {
    if(eigflag[i]==1)
    {
      eig_take[eig_index]=M_values[i];
      store_eig_index[eig_index]=i;
      eig_index++;
    }
  }

  double* pointer_to_data= NULL;
  eig_vector->get_array(pointer_to_data);
  for(unsigned int i=0; i<num_eig_take;)
  {
    if(abs(eig_take[i].imag())<1e-14)
    {
      for(unsigned int j=0; j<*number_of_vectors_size; j++)
      {
        pointer_to_data[i*(*number_of_vectors_size)+j]=M_vectors_real[store_eig_index[i]].get(j);
      }
      i++;
    }
    else
    {
      for(unsigned int j=0; j<*number_of_vectors_size; j++)
      {
        pointer_to_data[i*(*number_of_vectors_size)+j]=M_vectors_real[store_eig_index[i]].get(j);
        if((i+1)<num_eig_take)
          pointer_to_data[(i+1)*(*number_of_vectors_size)+j]=M_vectors_imag[store_eig_index[i]].get(j);
      }
      i+=2;
    }
  }
  eig_vector->store_array(pointer_to_data);
  eig_vector->assemble();
  NemoUtils::toc(tic_toc_prefix_for_copy_vector);

  //3. matrix-matrix mult
  int n1 = M1_matrix->get_num_rows();
  int m1 = eig_vector->get_num_cols();
  PetscMatrixParallel<double> eig_vector1(n1,m1,M1_matrix->get_communicator());
  eig_vector1.consider_as_full();
  eig_vector1.allocate_memory();
  PetscMatrixParallel<double>* p = &eig_vector1;
  std::string tic_toc_prefix_for_mult = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction for matrix-matrix(eigenvector) mult");
  NemoUtils::tic(tic_toc_prefix_for_mult);
  if(num_eig_take!=0)
    PetscMatrixParallel<double>::mult(*M1_matrix,*eig_vector,&p);


  NemoUtils::toc(tic_toc_prefix_for_mult);

  //4. comine double-vector matrix into cplx-vector, and normalize
  //follow the degeev() rules
  //if eigenvalue[i] is real, eigenvector[i] = eigenvector[i]
  //if eigenvalue[i] and eigenvalue[i+1] form complex conjugate pair, eigenvector[i]=eigenvector[i]+j*eigenvector[i+1] 
  std::string tic_toc_prefix_for_cplx = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction copy double-vector into cplx-vector");
  NemoUtils::tic(tic_toc_prefix_for_cplx);
  vector<std::complex<double> > M_vectors0(number_of_long_vector,cplx(0.0,0.0));
  vector< vector< std::complex<double> > > M_vectors_full(num_eig_take,M_vectors0);
  vector<double> vector_norm(number_of_long_vector,0.0);

  if(num_eig_take!=0)
  {
    double* pointer_to_data1= NULL;
    eig_vector1.get_array(pointer_to_data1);
    eig_vector->get_array(pointer_to_data);
    for (unsigned int i = 0; i < num_eig_take;)
    {
      if(abs(eig_take[i].imag())<1e-14)
      {
        for (unsigned int j= 0; j < M1_own_rows; j++)
        {
          M_vectors_full[i][j] = cplx(pointer_to_data1[i*M1_own_rows+j],0.0)/eig_take[i];
          vector_norm[i]+=std::abs(M_vectors_full[i][j]*M_vectors_full[i][j]);
        }
        for (unsigned int j= 0; j < *number_of_vectors_size; j++)
        {
          M_vectors_full[i][j+M1_own_rows] = cplx(pointer_to_data[i*(*number_of_vectors_size)+j],0.0);
          vector_norm[i]+=std::abs(M_vectors_full[i][j+M1_own_rows]*M_vectors_full[i][j+M1_own_rows]);
        }
        i++;
      }
      else
      {
        for (unsigned int j= 0; j < M1_own_rows; j++)
        {
          if((i+1)<num_eig_take)
          {
            M_vectors_full[i][j] = cplx(pointer_to_data1[i*M1_own_rows+j],pointer_to_data1[(i+1)*M1_own_rows+j])/eig_take[i];
            vector_norm[i]+=std::abs(M_vectors_full[i][j]*M_vectors_full[i][j]);
            M_vectors_full[i+1][j] = cplx(pointer_to_data1[i*M1_own_rows+j],-pointer_to_data1[(i+1)*M1_own_rows+j])/eig_take[i+1];
            vector_norm[i+1]+=std::abs(M_vectors_full[i+1][j]*M_vectors_full[i+1][j]);
          }
          else
          {
            M_vectors_full[i][j] = cplx(pointer_to_data1[i*M1_own_rows+j],0.0)/eig_take[i];
            vector_norm[i]+=std::abs(M_vectors_full[i][j]*M_vectors_full[i][j]);
          }
        }
        for (unsigned int j= 0; j < *number_of_vectors_size; j++)
        {
          if((i+1)<num_eig_take)
          {
            M_vectors_full[i][j+M1_own_rows] = cplx(pointer_to_data[i*(*number_of_vectors_size)+j],pointer_to_data[(i+1)*(*number_of_vectors_size)+j]);
            vector_norm[i]+=std::abs(M_vectors_full[i][j+M1_own_rows]*M_vectors_full[i][j+M1_own_rows]);
            M_vectors_full[i+1][j+M1_own_rows] = cplx(pointer_to_data[i*(*number_of_vectors_size)+j],-pointer_to_data[(i+1)*(*number_of_vectors_size)+j]);
            vector_norm[i+1]+=std::abs(M_vectors_full[i+1][j+M1_own_rows]*M_vectors_full[i+1][j+M1_own_rows]);
          }
          else
          {
            M_vectors_full[i][j+M1_own_rows] = cplx(pointer_to_data[i*(*number_of_vectors_size)+j],0.0);
            vector_norm[i]+=std::abs(M_vectors_full[i][j+M1_own_rows]*M_vectors_full[i][j+M1_own_rows]);
          }
        }
        i+=2;
      }
    }
    eig_vector->store_array(pointer_to_data);
    eig_vector1.store_array(pointer_to_data1);
  }
  for(unsigned int i=0; i<num_eig_take; i++)
    for(unsigned int j=0; j<number_of_long_vector; j++)
      M_vectors_full[i][j]=M_vectors_full[i][j]/std::sqrt(vector_norm[i]);//this is the final mode
  NemoUtils::toc(tic_toc_prefix_for_cplx);
  delete eig_vector;
  //===========================================================

  //5. loop over eigenvalues, determine mode direction
  vector<unsigned int> flag(num_eig_take,0);
  vector<std::complex<double> > expmikdelta(num_eig_take,cplx(0.0,0.0));
  vector<std::complex<double> > kdelta(num_eig_take,cplx(0.0,0.0));
  vector<double> velocity_to_store(num_eig_take,0.0);

  std::string tic_toc_prefix_for_loop = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_for_loop");
  NemoUtils::tic(tic_toc_prefix_for_loop);
  for(unsigned int i=0; i<num_eig_take; i++)
  {
    flag[i]=1;
    std::complex<double>  inv_M_values(0.0,0.0);

    cond1 = abs(eig_take[i])>eps_limit;
    cond2a = eig_take[i].real()<-1.0-eps_limit;
    cond2b = eig_take[i].real()>-1.0+eps_limit;
    cond2c = abs(eig_take[i].imag())>eps_limit;
    cond = (cond1&&(cond2a||cond2b||cond2c));  //OMEN--Injection::get_condition()
    if(cond) //if not the osolete solution phase factor !=0 or 1
    {
      std::string tic_toc_prefix_loop1 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction within loop: get full vector ");
      NemoUtils::tic(tic_toc_prefix_loop1);

      inv_M_values=cplx(1.0,0.0)/eig_take[i];
      expmikdelta[i]=cplx(1.0,0.0)+inv_M_values;
      kdelta[i]=cplx(0.0,1.0)*log(expmikdelta[i]); //Yu: correct version

      vector<std::complex<double> > M_vectors1(M_vectors_full[i]);

      NemoUtils::toc(tic_toc_prefix_loop1);

      std::complex<double> velocity_temp(0.0,0.0);

      std::string tic_toc_prefix_loop3 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction within loop: calculate velocity ");
      NemoUtils::tic(tic_toc_prefix_loop3);

      if((kdelta[i].imag()<imag_limit)&&(kdelta[i].imag()>-imag_limit))  //OMEN--Injection::get_type()
      {
        //propagating modes
        double velocity=0.0;
        vector<std::complex<double> > vectors_temp(number_of_long_vector,cplx(0.0,0.0));
        mult(coupling_matrix,M_vectors1,vectors_temp);
        for(unsigned int j=0; j<number_of_long_vector; j++)
          velocity_temp+=conj(M_vectors1[j])*vectors_temp[j];
        velocity_temp/=expmikdelta[i];
        velocity=2*velocity_temp.imag();//velocity of waves -- Tillmann: change - to + in front
        velocity_to_store[i]=velocity;

        if(velocity<0.0)
        {
          flag[i]=5; //reflected propagating wave (from device to lead)
          (*num_left)++;
          n_left_propagating++;
        }
        else
        {
          flag[i]=6; //transmitted propagating wave (from lead to device)
          (*num_right)++;
          n_right_propagating++;
        }
      }
      else
      {
        //decaying modes
        if(kdelta[i].imag()<-imag_limit)
        {
          flag[i]=3;//reflected decaying wave (from device to lead)
          (*num_left)++;
          n_left_decaying++;
        }
        else
        {
          flag[i]=4;//transmitted decaying wave (from lead to device)
          (*num_right)++;
          n_right_decaying++;
        }
      }
      NemoUtils::toc(tic_toc_prefix_loop3);
    }
    else { } //end of if(cond)...else
  } //end of for loop
  NemoUtils::toc(tic_toc_prefix_for_loop);

  //test flag
  std::string tic_toc_prefix_init_temp_matrix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_init_temp_matrix");
  NemoUtils::tic(tic_toc_prefix_init_temp_matrix);

  bool output_eigen=options.get_option("output_eigenvalues", bool(false));
  if(output_eigen)
  {
    for(unsigned int i=0; i<num_eig_take; i++)
    {
      std::cout<<NemoUtils::nemo_to_string(eig_take[i].real())<<"\t"<<NemoUtils::nemo_to_string(eig_take[i].imag())<<"\t";
      std::cout<<NemoUtils::nemo_to_string(expmikdelta[i].real());
      std::cout<<"\t"<<NemoUtils::nemo_to_string(expmikdelta[i].imag())<<"\t"<<flag[i]<<"\t"<<velocity_to_store[i]<<"\n";
    }
  }

  //if(!LRA)
  *number_of_vectors_size=number_of_long_vector;
  vector<int> nz_per_owned_wave_left_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_per_owned_wave_left_rows_offdiagonal(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_left_rows_diagonal_velocity(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_left_rows_offdiagonal_velocity(*number_of_vectors_size,0); //YH out of device modes
  vector<int> nz_per_owned_exp_left_rows_diagonal_velocity(*num_left,0);//Bozidar commented above two lines and inserted these
  vector<int> nz_per_owned_exp_left_rows_offdiagonal_velocity(*num_left,0);
  vector<int> nz_per_owned_exp_left_rows_diagonal(*num_left,0); 
  vector<int> nz_per_owned_exp_left_rows_offdiagonal(*num_left,0); //YH out of device modes

  for(unsigned int i=0; i<*number_of_vectors_size; i++)
  {
    nz_per_owned_wave_left_rows_diagonal[i]=*num_left;
    nz_per_owned_wave_left_rows_offdiagonal[i]=0;
  }
  for(unsigned int i=0; i<*num_left; i++)
  {
    nz_per_owned_exp_left_rows_diagonal_velocity[i]=1;
    nz_per_owned_exp_left_rows_offdiagonal_velocity[i]=0;
    nz_per_owned_exp_left_rows_diagonal[i]=1;
    nz_per_owned_exp_left_rows_offdiagonal[i]=0;
  }
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,*num_left,nz_per_owned_wave_left_rows_diagonal,nz_per_owned_wave_left_rows_offdiagonal,
      out_of_device_modes); //modes are dense
  transfer_matrix_initialize_temp_matrix(this_simulation,*num_left,*num_left,
                                         nz_per_owned_exp_left_rows_diagonal,nz_per_owned_exp_left_rows_offdiagonal,out_of_device_phase);//Yu: this is diagonal matrix, should not be full


  int n_right = n_right_propagating+n_right_decaying;
  vector<int> nz_per_owned_wave_right_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_per_owned_wave_right_rows_offdiagonal(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_right_rows_diagonal_velocity(*number_of_vectors_size,0);
  //vector<int> nz_per_owned_exp_right_rows_offdiagonal_velocity(*number_of_vectors_size,0);//YH into device modes
  vector<int> nz_per_owned_exp_right_rows_diagonal_velocity(n_right,0);//Bozidar commented above two lines and inserted these
  vector<int> nz_per_owned_exp_right_rows_offdiagonal_velocity(n_right,0);
  vector<int> nz_per_owned_exp_right_rows_diagonal(n_right,0);
  vector<int> nz_per_owned_exp_right_rows_offdiagonal(n_right,0);//YH into device modes

  for(unsigned int i=0; i<*number_of_vectors_size; i++)
  {
    nz_per_owned_wave_right_rows_diagonal[i]=n_right;
    nz_per_owned_wave_right_rows_offdiagonal[i]=0;
  }
  for(int i=0; i<n_right; i++)
  {
    nz_per_owned_exp_right_rows_diagonal_velocity[i]=1;
    nz_per_owned_exp_right_rows_offdiagonal_velocity[i]=0;
    nz_per_owned_exp_right_rows_diagonal[i]=1;
    nz_per_owned_exp_right_rows_offdiagonal[i]=0;
  }
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,n_right,nz_per_owned_wave_right_rows_diagonal,nz_per_owned_wave_right_rows_offdiagonal,
      into_device_modes); //modes are dense
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right,n_right,
                                         nz_per_owned_exp_right_rows_diagonal,nz_per_owned_exp_right_rows_offdiagonal,into_device_phase); //Yu: this is diagonal matrix, should not be full

  //collect waves and phase factor for QTBM
  vector<int> nz_left_propagating_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_left_propagating_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_left_decaying_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_left_decaying_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_right_propagating_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_right_propagating_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_right_decaying_wave_rows_diagonal(*number_of_vectors_size,0);
  vector<int> nz_right_decaying_wave_rows_offdiagonal(*number_of_vectors_size,0);
  vector<int> nz_left_propagating_phase_rows_diagonal(n_left_propagating,0);
  vector<int> nz_left_propagating_phase_rows_offdiagonal(n_left_propagating,0);
  vector<int> nz_left_decaying_phase_rows_diagonal(n_left_decaying,0);
  vector<int> nz_left_decaying_phase_rows_offdiagonal(n_left_decaying,0);
  vector<int> nz_right_propagating_phase_rows_diagonal(n_right_propagating,0);
  vector<int> nz_right_propagating_phase_rows_offdiagonal(n_right_propagating,0);
  vector<int> nz_right_decaying_phase_rows_diagonal(n_right_decaying,0);
  vector<int> nz_right_decaying_phase_rows_offdiagonal(n_right_decaying,0);

  for(unsigned int i=0; i<*number_of_vectors_size; i++)
  {
    nz_left_propagating_wave_rows_diagonal[i]=n_left_propagating;
    nz_left_propagating_wave_rows_offdiagonal[i]=0;
    nz_left_decaying_wave_rows_diagonal[i]=n_left_decaying;
    nz_left_decaying_wave_rows_offdiagonal[i]=0;
    nz_right_propagating_wave_rows_diagonal[i]=n_right_propagating;
    nz_right_propagating_wave_rows_offdiagonal[i]=0;
    nz_right_decaying_wave_rows_diagonal[i]=n_right_decaying;
    nz_right_decaying_wave_rows_offdiagonal[i]=0;
  }
  for(unsigned int i=0; i<n_left_propagating; i++)
  {
    nz_left_propagating_phase_rows_diagonal[i]=1;
    nz_left_propagating_phase_rows_offdiagonal[i]=0;
  }
  for(unsigned int i=0; i<n_right_propagating; i++)
  {
    nz_right_propagating_phase_rows_diagonal[i]=1;
    nz_right_propagating_phase_rows_offdiagonal[i]=0;
  }
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,n_left_propagating,nz_left_propagating_wave_rows_diagonal,
      nz_left_propagating_wave_rows_offdiagonal,
      out_of_device_propagating_modes); //Yu: make it dense, seems necessary for avoiding reordering
  transfer_matrix_initialize_temp_full_matrix(this_simulation,*number_of_vectors_size,n_right_propagating,nz_right_propagating_wave_rows_diagonal,
      nz_right_propagating_wave_rows_offdiagonal,
      into_device_propagating_modes); //Yu: make it dense, seems necessary for avoiding reordering
  transfer_matrix_initialize_temp_matrix(this_simulation,n_left_propagating,n_left_propagating,
                                         nz_left_propagating_phase_rows_diagonal,nz_left_propagating_phase_rows_offdiagonal,out_of_device_propagating_phase);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_left_propagating,n_left_propagating,
                                         nz_left_propagating_phase_rows_diagonal,nz_left_propagating_phase_rows_offdiagonal,out_of_device_propagating_velocity);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right_propagating,n_right_propagating,
                                         nz_right_propagating_phase_rows_diagonal,nz_right_propagating_phase_rows_offdiagonal,into_device_propagating_velocity);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right_propagating,n_right_propagating,
                                         nz_right_propagating_phase_rows_diagonal,nz_right_propagating_phase_rows_offdiagonal,into_device_propagating_phase);
  //transfer_matrix_initialize_temp_matrix(*number_of_vectors_size,*number_of_vectors_size,
  //                                       nz_per_owned_exp_left_rows_diagonal_velocity,nz_per_owned_exp_left_rows_offdiagonal_velocity,out_of_device_velocity);
  //transfer_matrix_initialize_temp_matrix(*number_of_vectors_size,*number_of_vectors_size,
  //                                       nz_per_owned_exp_right_rows_diagonal_velocity,nz_per_owned_exp_right_rows_offdiagonal_velocity,into_device_velocity);
  //Bozidar: the two lines above have incorrect total matrix sizes (first two arguments).
  transfer_matrix_initialize_temp_matrix(this_simulation,*num_left,*num_left,
    nz_per_owned_exp_left_rows_diagonal_velocity,nz_per_owned_exp_left_rows_offdiagonal_velocity,out_of_device_velocity);
  transfer_matrix_initialize_temp_matrix(this_simulation,n_right,n_right,
    nz_per_owned_exp_right_rows_diagonal_velocity,nz_per_owned_exp_right_rows_offdiagonal_velocity,into_device_velocity);

  NemoUtils::toc(tic_toc_prefix_init_temp_matrix);

  std::string tic_toc_prefix_collect_waves_phase = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_wave_direction_step_collect_waves_phase");
  NemoUtils::tic(tic_toc_prefix_collect_waves_phase);

  //set elements
  unsigned int j=0;
  unsigned int k=0;
  unsigned int m=0;
  unsigned int n=0;
  cplx* pointer_modes_left=NULL;
  cplx* pointer_modes_right=NULL;
  cplx* pointer_modes_left_p=NULL;
  cplx* pointer_modes_right_p=NULL;
  out_of_device_modes->get_array(pointer_modes_left);
  into_device_modes->get_array(pointer_modes_right);
  out_of_device_propagating_modes->get_array(pointer_modes_left_p);
  into_device_propagating_modes->get_array(pointer_modes_right_p);

  if(options.get_option("consistent_modes", false))
  {//Bozidar: sort propagating modes by velocities and decaying by phase factor's magnitude. If phase factor's magnitude is equal then by phase factor's phase.

    std::vector<int> into_device_modes_input_index;
    std::vector<int> into_device_modes_input_index_p;
    std::vector<int> into_device_modes_index;
    std::vector<int> into_device_modes_index_p;
    std::vector<int> out_of_device_modes_input_index;
    std::vector<int> out_of_device_modes_input_index_p;
    std::vector<int> out_of_device_modes_index;
    std::vector<int> out_of_device_modes_index_p;

    //Set input vector sizes for vectors necesary for sorting utility function.
    for(unsigned int i = 0; i < num_eig_take; i++)
    {
      if(flag[i] == 3) //reflected decaying waves
        j++;
      if(flag[i] == 5) //reflected propagating waves
        m++;
      if(flag[i] == 4) //transmitted decaying waves
        k++;
      if(flag[i] == 6) //transmitted propagating waves
        n++;
    }
    out_of_device_modes_input_index_p.resize(m);
    out_of_device_modes_input_index.resize(j);
    into_device_modes_input_index_p.resize(n);
    into_device_modes_input_index.resize(k);

    j = 0;
    k = 0;
    m = 0;
    n = 0;
    //Find input indices necessary for sorting utility function.
    for(unsigned int i = 0; i < num_eig_take; i++)
    {
      if(flag[i] == 3)
        out_of_device_modes_input_index[j++] = i;
      if(flag[i] == 5)
        out_of_device_modes_input_index_p[m++] = i;
      if(flag[i] == 4)
        into_device_modes_input_index[k++] = i;
      if(flag[i] == 6)
        into_device_modes_input_index_p[n++] = i;
    }

    //Make complex velocity vector to comply with sorting function API.
    std::vector<std::complex<double> > complex_velocity_to_store(velocity_to_store.size(), std::complex<double>(0.0, 0.0));
    for(unsigned int i = 0; i < velocity_to_store.size(); i++)
      complex_velocity_to_store[i].real() = velocity_to_store[i];

    //Run sorting function and obtain necesssary reordered indices.
    NemoUtils::sort_abs_then_phase(complex_velocity_to_store, out_of_device_modes_input_index_p, out_of_device_modes_index_p);
    NemoUtils::sort_abs_then_phase(expmikdelta, out_of_device_modes_input_index, out_of_device_modes_index);
    NemoUtils::sort_abs_then_phase(complex_velocity_to_store, into_device_modes_input_index_p, into_device_modes_index_p);
    NemoUtils::sort_abs_then_phase(expmikdelta, into_device_modes_input_index, into_device_modes_index);

    //Some test code for run without sorting.
    //out_of_device_modes_index_p.resize(out_of_device_modes_input_index_p.size());
    //for(unsigned int i = 0; i < out_of_device_modes_input_index_p.size(); i++)
    //  out_of_device_modes_index_p[i] = i;
    //out_of_device_modes_index.resize(out_of_device_modes_input_index.size());
    //for(unsigned int i = 0; i < out_of_device_modes_input_index.size(); i++)
    //  out_of_device_modes_index[i] = i;
    //into_device_modes_index_p.resize(into_device_modes_input_index_p.size());
    //for(unsigned int i = 0; i < into_device_modes_input_index_p.size(); i++)
    //  into_device_modes_index_p[i] = i;
    //into_device_modes_index.resize(into_device_modes_input_index.size());
    //for(unsigned int i = 0; i < into_device_modes_input_index.size(); i++)
    //  into_device_modes_index[i] = i;


    //Create mapping between original and ordered indices (key is new ordering, value is original ordering).
    std::map<int,int> map_out_of_device_modes_index_p;
    for(unsigned int i = 0; i < out_of_device_modes_input_index_p.size(); i++)
      map_out_of_device_modes_index_p[out_of_device_modes_index_p[i]] = out_of_device_modes_input_index_p[i];
    std::map<int,int> map_out_of_device_modes_index;
    for(unsigned int i = 0; i < out_of_device_modes_input_index.size(); i++)
      map_out_of_device_modes_index[out_of_device_modes_index[i]] = out_of_device_modes_input_index[i];
    std::map<int,int> map_into_device_modes_index_p;
    for(unsigned int i = 0; i < into_device_modes_input_index_p.size(); i++)
      map_into_device_modes_index_p[into_device_modes_index_p[i]] = into_device_modes_input_index_p[i];
    std::map<int,int> map_into_device_modes_index;
    for(unsigned int i = 0; i < into_device_modes_input_index.size(); i++)
      map_into_device_modes_index[into_device_modes_index[i]] = into_device_modes_input_index[i];

    //Copy complex velocity vector back to real vector.
    for(unsigned int i = 0; i < velocity_to_store.size(); i++)
      velocity_to_store[i] = complex_velocity_to_store[i].real();

    //Fill into and out of device petsc matrices with ordered and normalized modes.
    j = 0;
    k = 0;
    m = 0;
    n = 0;

    //WARNING: The following relies on the fact that mode i corresponds to a mode determined by j,k,m,n in one of separate into and out of matrices.
    for(unsigned int i = 0; i < num_eig_take; i++)
    {
      if(flag[i] == 3) //reflected decaying waves
      {
        out_of_device_velocity->set(j, j, velocity_to_store[i]); //no need to sort since zero.
        out_of_device_phase->set(j, j, expmikdelta[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left[j*(*number_of_vectors_size) + l] = M_vectors_full[map_out_of_device_modes_index[j - m]][l];
        j++;
      }

      if(flag[i] == 5) //reflected propagating waves
      {
        out_of_device_velocity->set(j, j, velocity_to_store[i]); //already sorted.
        out_of_device_phase->set(j, j, expmikdelta[map_out_of_device_modes_index_p[m]]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left[j*(*number_of_vectors_size) + l] = M_vectors_full[map_out_of_device_modes_index_p[m]][l];
        j++;

        out_of_device_propagating_phase->set(m, m, expmikdelta[map_out_of_device_modes_index_p[m]]);
        out_of_device_propagating_velocity->set(m, m, velocity_to_store[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left_p[m*(*number_of_vectors_size)+l] = M_vectors_full[map_out_of_device_modes_index_p[m]][l];
        m++;
      }

      if(flag[i] == 4) //transmitted decaying waves
      {
        into_device_velocity->set(k, k, velocity_to_store[i]); //no need to sort since zero.
        into_device_phase->set(k, k, expmikdelta[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right[k*(*number_of_vectors_size) + l] = M_vectors_full[map_into_device_modes_index[k - n]][l];
        k++;
      }
      if(flag[i] == 6) //transmitted propagating waves
      {
        into_device_velocity->set(k, k, velocity_to_store[i]); //already sorted.
        into_device_phase->set(k, k, expmikdelta[map_into_device_modes_index_p[n]]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right[k*(*number_of_vectors_size) + l] = M_vectors_full[map_into_device_modes_index_p[n]][l];
        k++;

        into_device_propagating_phase->set(n, n, expmikdelta[map_into_device_modes_index_p[n]]);
        into_device_propagating_velocity->set(n, n, velocity_to_store[i]); //already sorted.
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right_p[n*(*number_of_vectors_size)+l] = M_vectors_full[map_into_device_modes_index_p[n]][l];
        n++;
      }
    }
  }
  else
  {
    for(unsigned int i = 0; i<num_eig_take; i++)
    {
      if(flag[i] == 3 || flag[i] == 5) //reflected waves
      {
        out_of_device_velocity->set(j, j, velocity_to_store[i]);
        out_of_device_phase->set(j, j, expmikdelta[i]);
        for(unsigned int l = 0; l<*number_of_vectors_size; l++)
          pointer_modes_left[j*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        j++;
      }
      if(flag[i] == 4 || flag[i] == 6) //transmitted waves
      {
        into_device_velocity->set(k, k, velocity_to_store[i]);
        into_device_phase->set(k, k, expmikdelta[i]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right[k*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        k++;
      }
      if(flag[i] == 5) //reflected propagating waves
      {
        out_of_device_propagating_phase->set(m, m, expmikdelta[i]);
        out_of_device_propagating_velocity->set(m, m, velocity_to_store[i]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_left_p[m*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        m++;
      }
      if(flag[i] == 6) //transmitted propagating waves
      {
        into_device_propagating_phase->set(n, n, expmikdelta[i]);
        into_device_propagating_velocity->set(n, n, velocity_to_store[i]);
        for(unsigned int l = 0; l < *number_of_vectors_size; l++)
          pointer_modes_right_p[n*(*number_of_vectors_size) + l] = M_vectors_full[i][l];
        n++;
      }
    }
  }

  into_device_propagating_modes->store_array(pointer_modes_right_p);
  out_of_device_propagating_modes->store_array(pointer_modes_left_p);
  into_device_modes->store_array(pointer_modes_right);
  out_of_device_modes->store_array(pointer_modes_left);
  out_of_device_modes->assemble();
  into_device_modes->assemble();
  out_of_device_phase->assemble();
  into_device_phase->assemble();
  out_of_device_propagating_modes->assemble();
  out_of_device_propagating_phase->assemble();
  into_device_propagating_modes->assemble();
  into_device_propagating_phase->assemble();
  out_of_device_velocity->assemble();
  into_device_velocity->assemble();
  out_of_device_propagating_velocity->assemble();
  into_device_propagating_velocity->assemble();

    
  temp_transfer_type=NemoPhys::into_device_propagating_modes;
  PropInterface->set_transfer_matrix_element(into_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_phase;
  PropInterface->set_transfer_matrix_element(into_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_modes;
  PropInterface->set_transfer_matrix_element(into_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_phase;
  PropInterface->set_transfer_matrix_element(into_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_modes;
  PropInterface->set_transfer_matrix_element(out_of_device_modes,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_phase;
  PropInterface->set_transfer_matrix_element(out_of_device_phase,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_velocity;
  PropInterface->set_transfer_matrix_element(into_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::into_device_propagating_velocity;
  PropInterface->set_transfer_matrix_element(into_device_propagating_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_velocity;
  PropInterface->set_transfer_matrix_element(out_of_device_velocity,&temp_transfer_type);

  temp_transfer_type=NemoPhys::out_of_device_propagating_velocity;
  PropInterface->set_transfer_matrix_element(out_of_device_propagating_velocity,&temp_transfer_type);

  NemoUtils::toc(tic_toc_prefix_collect_waves_phase);
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::solve_retarded_Green_TransferMatrix(Simulation* this_simulation, Propagator*& , const std::vector<NemoMeshPoint>& ,
        PetscMatrixParallelComplex*& )
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_retarded_Green_TransferMatrix ";
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_retarded_Green_TransferMatrix ";
  //2. perform the transfer matrix method on the new domains 0,1,2,3 (getting the wave functions and phase factors)
  //transfer_matrix_leads(output_Propagator, momentum_point, result);
  //3. translate the results on 0,1,2,3 to the original domain (this->get_const_simulation_domain())

  //4. create the retarded Green's function using the results on this domain and store in result
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::Sancho_Rubio_retarded_green(Simulation* input_simulation,Propagator*& output_Propagator, const Domain* neighbor_domain,
    const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\"" + input_simulation->get_name()+ "\")::Sancho_Rubio_retarded_green ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\"" + input_simulation->get_name()+ "\")::Sancho_Rubio_retarded_green ";
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(input_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(input_simulation);
  const InputOptions& input_options=input_simulation->get_options();

  std::string temp_file_string;
  {
    const std::vector<NemoMeshPoint>* temp_pointer = &(momentum);
    translate_momentum_vector(input_simulation,temp_pointer, temp_file_string);
  }

  PetscMatrixParallelComplex* H00 = NULL;
  PetscMatrixParallelComplex* coupling = NULL;
  PetscMatrixParallelComplex* S00      = NULL;            //!overlap matrix of the lead
  PetscMatrixParallelComplex* coupling_overlap = NULL;    //!overlap matricx coupling with the lead
  DOFmapInterface* coupling_DOFmap = NULL;
  DOFmapInterface* coupling_Ov_DOFmap = NULL;                      //!The overlap matrix always return a NULL DOFmap

  std::set<unsigned int> Hamilton_momentum_indices;
  std::set<unsigned int>* pointer_to_Hamilton_momentum_indices =&Hamilton_momentum_indices;
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  find_Hamiltonian_momenta(input_simulation,writeable_Propagator, pointer_to_Hamilton_momentum_indices);
  NemoMeshPoint temp_NemoMeshPoint(0, std::vector<double>(3, 0.0));
  if (pointer_to_Hamilton_momentum_indices != NULL)
    temp_NemoMeshPoint = momentum[*(pointer_to_Hamilton_momentum_indices->begin())];

  bool LRA_RGF = input_options.get_option("LRA_RGF", bool(false));

  //1. find the Hamilton constructor of lead (H00 in the Sancho-Rubio algorithm)
  std::string lead_Hamilton_variable_name = neighbor_domain->get_name() + std::string("_Hamilton_constructor");
  std::string lead_Hamilton_constructor_name;
  if (input_options.check_option(lead_Hamilton_variable_name))
    lead_Hamilton_constructor_name = input_options.get_option(lead_Hamilton_variable_name, std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix + " define \"" + lead_Hamilton_variable_name + "\"\n");
  Simulation* lead_Hamilton_Constructor;
  lead_Hamilton_Constructor = input_simulation->find_simulation(lead_Hamilton_constructor_name);
  NEMO_ASSERT(lead_Hamilton_Constructor!=NULL,
              tic_toc_prefix+"Hamilton_constructor has not been found!\n");

  //2. find the neighbor_domain of lead and its coupling Hamiltonian
  std::string neighbor_lead_name = std::string("lead_of_")+ neighbor_domain->get_name();
  //cout<<neighbor_domain->get_name()<<"\n";
  std::string neighbor_of_lead;
  if (input_options.check_option(neighbor_lead_name))
    neighbor_of_lead = input_options.get_option(neighbor_lead_name, std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix + " define \"" + neighbor_lead_name + "\"\n");
  const Domain* neighbor_of_lead_domain = Domain::get_domain(neighbor_of_lead);

  PetscMatrixParallelComplex* tmp_H00 = NULL;
  PetscMatrixParallelComplex* tmp_S00 = NULL;

  if (!LRA_RGF)
  {
    std::vector<NemoMeshPoint> sorted_momentum;
    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,input_options,PropInterface->get_momentum_mesh_types(),lead_Hamilton_Constructor);
    lead_Hamilton_Constructor->get_data(std::string("Hamiltonian"), tmp_H00, sorted_momentum,false,neighbor_domain);
    lead_Hamilton_Constructor->get_data(string("overlap_matrix")  , tmp_S00, sorted_momentum,false,neighbor_domain);
  }
  else
  {
    //prepare vector<NemoMeshPoints> 1st entry is Hamilton momenta rest is the other momenta for Propagation
    std::vector<NemoMeshPoint> temp_vector_nemomeshpoint(1,temp_NemoMeshPoint);
    //temp_vector_nemomeshpoint.resize(1);
    //(*temp_vector_nemomeshpoint)[0] = temp_NemoMeshPoint;
    //temp_vector_nemomeshpoint.push_back(temp_NemoMeshPoint);
    int momentum_size = momentum.size();
    for (int i = 0; i < momentum_size; ++i)
      temp_vector_nemomeshpoint.push_back(momentum[i]);
    lead_Hamilton_Constructor->get_data(std::string("Hamiltonian"), tmp_H00,&temp_vector_nemomeshpoint);
  }

  std::vector<NemoMeshPoint> sorted_momentum;
  QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,input_options,PropInterface->get_momentum_mesh_types(),lead_Hamilton_Constructor);
  Simulation* temp_simulation=input_simulation->find_simulation_on_domain(neighbor_of_lead_domain,lead_Hamilton_Constructor->get_type());
  //NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found HamiltonConstructor for neighbor_of_lead_domain\n");
  if (temp_simulation == NULL && (lead_Hamilton_Constructor->get_type().find("module") != std::string::npos || lead_Hamilton_Constructor->get_type().find("Module") != std::string::npos))
    temp_simulation = lead_Hamilton_Constructor;
  NEMO_ASSERT(temp_simulation != NULL, tic_toc_prefix + "have not found Hamilton_Constructor for neighbor_domain\n");
  DOFmapInterface& temp_dofmap=temp_simulation->get_dof_map(neighbor_of_lead_domain);
  coupling_DOFmap=&temp_dofmap;
  DOFmapInterface* temp_pointer=coupling_DOFmap;
  lead_Hamilton_Constructor->get_data(std::string("Hamiltonian"), sorted_momentum, neighbor_of_lead_domain, coupling, coupling_DOFmap,neighbor_domain);
  lead_Hamilton_Constructor->get_data(std::string("overlap_matrix_coupling"), sorted_momentum, neighbor_of_lead_domain, coupling_overlap, coupling_Ov_DOFmap,
                                      neighbor_domain);

  //H00->matrix_convert_dense();
  H00 = new PetscMatrixParallelComplex(*tmp_H00);
  if(PropOptionInterface->get_debug_output())
  {
    coupling_overlap->save_to_matlab_file("SR"+temp_file_string+"coupling.m");
    H00->save_to_matlab_file("H00"+temp_file_string+".m");
  }
  //coupling->save_to_matlab_file("coupling_lead.m");
  if(tmp_S00!=NULL)
  {
    S00 = new PetscMatrixParallelComplex(*tmp_S00);
    if(PropOptionInterface->get_debug_output())
      S00->save_to_matlab_file("S00"+temp_file_string+".m");
  }

  //4. Get the H01 matrix from upper right block of lead coupling matrix
  unsigned int number_of_00_rows;
  unsigned int number_of_00_cols;
  unsigned int number_of_coupling_rows;
  //unsigned int number_of_coupling_cols = coupling->get_num_cols();
  //unsigned int local_number_of_00_rows = H00->get_num_owned_rows();
  //unsigned int local_number_of_coupling_rows = coupling->get_num_owned_rows();

  if (!LRA_RGF)
  {
    number_of_00_rows = H00->get_num_rows();
    number_of_coupling_rows = coupling->get_num_rows();
    if(number_of_coupling_rows <= number_of_00_rows)
    {
      number_of_00_rows = 0;
    }
    number_of_00_cols = number_of_00_rows;

    //unsigned int number_of_coupling_cols = coupling->get_num_cols();
    //unsigned int local_number_of_00_rows = H00->get_num_owned_rows();
    //unsigned int local_number_of_coupling_rows = coupling->get_num_owned_rows();
  }
  else //LRA_RGF
  {
    ////get untransformed subdomain sizes.
    //std::string Hamilton_Constructor2_name = input_options.get_option("Hamilton_constructor_for_coupling", std::string(""));
    //Simulation* Hamilton_Constructor2 = input_simulation->find_simulation(Hamilton_Constructor2_name);
    //const DOFmapInterface& device_DOFmap2      =   Hamilton_Constructor2->get_const_dof_map(input_simulation->get_const_simulation_domain());
    //DOFmapInterface* coupling_DOFmap2 = NULL;

    //std::vector<NemoMeshPoint> sorted_momentum;
    //QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,input_options,PropInterface->get_momentum_mesh_types(),Hamilton_Constructor2);
    //Hamilton_Constructor2->get_data(std::string("Hamiltonian"),
    //                                sorted_momentum, neighbor_of_lead_domain, coupling,coupling_DOFmap2);

    //number_of_00_rows = device_DOFmap2.get_global_dof_number();
    //number_of_00_cols = number_of_00_rows;
    //number_of_coupling_rows = coupling->get_num_rows();
    //delete coupling_DOFmap2;
    //coupling_DOFmap2 = NULL;
    throw std::runtime_error(prefix+"LRA_RGF is no longer supported option setting\n");
  }

  unsigned int number_of_01_rows = number_of_coupling_rows- number_of_00_rows;
  unsigned int number_of_01_cols = number_of_01_rows;

  std::vector<int> temp_rows(number_of_01_rows);
  std::vector<int> temp_cols(number_of_01_cols);

  for (unsigned int i = 0; i < number_of_01_rows; i++)
    temp_rows[i] = i;
  for (unsigned int i = 0; i < number_of_01_cols; i++)
    temp_cols[i] = i + number_of_00_cols;

  PetscMatrixParallelComplex* H01 = NULL;
  H01 = new PetscMatrixParallelComplex(number_of_01_rows, number_of_01_cols, input_simulation->get_simulation_domain()->get_communicator());
  H01->set_num_owned_rows(number_of_01_rows);

  vector<int> rows_diagonal(number_of_01_rows, 0);
  vector<int> rows_offdiagonal(number_of_01_rows, 0);
  unsigned int number_of_nonzero_cols_local = 0;
  unsigned int number_of_nonzero_cols_nonlocal = 0;

  for (unsigned int i = 0; i < number_of_01_rows; i++)
  {
    rows_diagonal[i] = coupling->get_nz_diagonal(i);
    rows_offdiagonal[i] = coupling->get_nz_offdiagonal(i);
    if (rows_diagonal[i] > 0)
      number_of_nonzero_cols_local++;
    if (rows_offdiagonal[i] > 0)
      number_of_nonzero_cols_nonlocal++;
  }
  for (unsigned int i = 0; i < number_of_01_rows; i++)
    H01->set_num_nonzeros_for_local_row(i, rows_diagonal[i],rows_offdiagonal[i]);
  coupling->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, H01);


  //5. Get the Energy point and start the Sancho-Rubio algorithm
  std::complex<double> energy;
  //std::complex<double> temp = cplx(1.0,1.0);
  energy = read_complex_energy_from_momentum(input_simulation,momentum, output_Propagator);

  if(coupling_overlap!=NULL)
  {
    PetscMatrixParallelComplex* S01 = NULL;
    S01 = new PetscMatrixParallelComplex(number_of_01_rows,number_of_01_cols,input_simulation->get_simulation_domain()->get_communicator());
    S01->set_num_owned_rows(number_of_01_rows);

    //vector<int> rows_diagonal(number_of_01_rows,0);
    //vector<int> rows_offdiagonal(number_of_01_rows,0);
    number_of_nonzero_cols_local    = 0;
    number_of_nonzero_cols_nonlocal = 0;
    for(unsigned int i=0; i<number_of_01_rows; i++)
    {
      rows_diagonal[i] = coupling_overlap->get_nz_diagonal(i);
      rows_offdiagonal[i] = coupling_overlap->get_nz_offdiagonal(i);
      if(rows_diagonal[i] > 0)
        number_of_nonzero_cols_local++;
      if(rows_offdiagonal[i] > 0)
        number_of_nonzero_cols_nonlocal++;
    }

    for(unsigned int i=0; i<number_of_01_rows; i++)
      S01->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
    coupling_overlap->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,S01);
    if(PropOptionInterface->get_debug_output())
    {
      
      S01->save_to_matlab_file("S01_red"+temp_file_string+".m");
    }

    *S01 *= -energy; //-E*S01
    //!Huckel Blocks H01-ES01
    H01->add_matrix(*S01,DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
    //H01->save_to_matlab_file("D01_red.m");
    delete S01;
    S01=NULL;
  }

  if (LRA_RGF)
  {
    //perform transformations on coupling and transpose_coupling
    H01->assemble();
    PropOptionInterface->get_Hamilton_Constructor()->get_data("subcoupling_Hamiltonian", &momentum, H01);

    number_of_00_rows = H00->get_num_rows();
    number_of_00_cols = H00->get_num_cols();
    number_of_01_rows = H01->get_num_rows();
    number_of_01_cols = number_of_01_rows;

  }

  H01->assemble();
  if(PropOptionInterface->get_debug_output())
  {
    H01->save_to_matlab_file("H01"+temp_file_string+".m");
  }
  PetscMatrixParallelComplex* H10 = new PetscMatrixParallelComplex(*H01);
  H10->hermitian_transpose_matrix(*H01, MAT_INITIAL_MATRIX);

  
  //5. Get the Energy point and start the Sancho-Rubio algorithm
  double eta = input_options.get_option("constant_lead_eta", 1e-10);
  if(PropOptionInterface->get_particle_type_is_Fermion())
    energy += cplx(0.0, 1.0) * eta;
  else
    energy = energy*energy+cplx(0.0, 1.0) * eta;
  std::complex<double> x = 1.0;

  PetscMatrixParallelComplex* tmp1 = new PetscMatrixParallelComplex(*H00);
  tmp1->matrix_convert_dense();
  tmp1->set_to_zero();

  PetscMatrixParallelComplex* tmp2 = new PetscMatrixParallelComplex(*H00);
  tmp2->matrix_convert_dense();
  tmp2->set_to_zero();

  PetscMatrixParallelComplex* omega;
  if(coupling_overlap==NULL) //!E*[I] identity matrix
  {
    omega= new PetscMatrixParallelComplex(*tmp1);
    omega->matrix_diagonal_shift(energy);
    PetscMatrixParallelComplex* bp_matrix=NULL;
    if(input_options.check_option("Buettiker_probe_solver"))
    {
      std::string bp_name=input_options.get_option("Buettiker_probe_solver", std::string(""));
      Simulation* data_source=find_source_of_data(input_simulation,bp_name);
      data_source->get_data(bp_name,&momentum,bp_matrix,&(lead_Hamilton_Constructor->get_const_dof_map(input_simulation->get_const_simulation_domain())));
      bp_matrix->matrix_convert_dense();
      omega->add_matrix(*bp_matrix, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));
    }
  }
  else
  {
    *S00 *= energy; //E*S00
    //omega = S00->clone();
    omega = new PetscMatrixParallelComplex(*S00);
    omega->matrix_convert_dense();
    if(PropOptionInterface->get_debug_output())
      omega->save_to_matlab_file("omega"+temp_file_string+".m");
    delete S00;
    S00=NULL;
  }

  //Prasad: start_row and end_row store the start and end indices of H01 and H10
  unsigned int start_row_01 = 0;
  unsigned int end_row_01 = 0;

  unsigned int start_row_10 = 0;
  unsigned int end_row_10 = 0;

  unsigned int nonzero_index = 0;

  for(unsigned int i=0;i < number_of_01_rows; i++)
  {
	  nonzero_index = H01->get_nz_diagonal(i);
	  if (nonzero_index > 0)
	  {
		  start_row_01 = i;
		  break;
	  }
  }
  for(unsigned int i=number_of_01_rows-1; i>0; i--)
  {
	  nonzero_index = H01->get_nz_diagonal(i);
	  if (nonzero_index > 0)
	  {
		  end_row_01 = i;
		  break;
	  }
  }

  int n_nonzeros=0;
  const int* col_nums=NULL;
  const std::complex<double>* pointer_to_data = NULL;
  unsigned int minindex = 0;
  unsigned int maxindex = 0;
  unsigned int start_col_01 =0;
  unsigned int end_col_01 = 0;
  int init_flag = 0;
  for(unsigned int i=0; i<number_of_01_rows; i++)
  {
	  H01->get_row(i,&n_nonzeros,col_nums,pointer_to_data);
	  if (n_nonzeros > 0)
	  {
		  minindex = col_nums[0];
		  maxindex = col_nums[n_nonzeros-1];

		  if (init_flag == 0)
		  {
			  start_col_01 = minindex;
			  end_col_01 = maxindex;
			  init_flag = 1;
		  }
		  else
		  {
			  if (start_col_01 > minindex)
				  start_col_01 = minindex;
			  if (end_col_01 < maxindex)
				  end_col_01 = maxindex;
		  }
	  }
	  H01->store_row(i,&n_nonzeros,col_nums,pointer_to_data);
  }

  unsigned int number_of_01_block_rows = 0;
  unsigned int number_of_01_block_cols = 0;

  unsigned int start_col_10 = 0;
  unsigned int end_col_10 = 0;

  for(unsigned int i=0;i < number_of_01_rows; i++)
  {
	  nonzero_index = H10->get_nz_diagonal(i);
	  if (nonzero_index > 0)
	  {
		  start_row_10 = i;
		  break;
	  }
  }
  for(unsigned int i=number_of_01_rows-1; i>0; i--)
  {
	  nonzero_index = H10->get_nz_diagonal(i);
	  if (nonzero_index > 0)
	  {
		  end_row_10 = i;
		  break;
	  }
  }

  minindex = 0;
  maxindex = 0;
  init_flag = 0;

  for(unsigned int i=0; i<number_of_01_rows; i++)
  {
	  H10->get_row(i,&n_nonzeros,col_nums,pointer_to_data);
	  if (n_nonzeros > 0)
	  {
		  minindex = col_nums[0];
		  maxindex = col_nums[n_nonzeros-1];

		  if (init_flag == 0)
		  {
			  start_col_10 = minindex;
			  end_col_10 = maxindex;
			  init_flag = 1;
		  }
		  else
		  {
			  if (start_col_10 > minindex)
				  start_col_10 = minindex;
			  if (end_col_10 < maxindex)
				  end_col_10 = maxindex;
		  }
	  }
	  H10->store_row(i,&n_nonzeros,col_nums,pointer_to_data);
  }

  unsigned int number_of_10_block_rows = 0;
  unsigned int number_of_10_block_cols = 0;

  std::vector<int> row_index_01(1,0);
  std::vector<int> row_index_10(1,0);
  std::vector<int> col_index_01(1,0);
  std::vector<int> col_index_10(1,0);

  if (start_row_01 < number_of_01_rows/2 && end_row_01 < number_of_01_rows/2)
  {
	  if (start_col_01 < number_of_01_cols/2 && end_col_01 < number_of_01_cols/2)
	  {
		  number_of_01_block_rows = end_row_01 + 1;
		  number_of_01_block_cols = end_col_01 + 1;
		  row_index_01.resize(number_of_01_block_rows,0);
		  col_index_01.resize(number_of_01_block_cols,0);
		  for(unsigned int i=0; i < number_of_01_block_rows;i++)
		  {
			  row_index_01[i] = i;
		  }
		  for(unsigned int i=0; i < number_of_01_block_cols;i++)
		  {
			  col_index_01[i] = i;
		  }
	  }
	  else
		  if (start_col_01 < number_of_01_cols/2 && end_col_01 > number_of_01_cols/2)
		  {
			  number_of_01_block_rows = end_row_01 + 1;
			  number_of_01_block_cols = end_col_01 + 1;
			  row_index_01.resize(number_of_01_block_rows,0);
			  col_index_01.resize(number_of_01_block_cols,0);
			  for(unsigned int i=0; i < number_of_01_block_rows;i++)
			  {
				  row_index_01[i] = i;
			  }
			  for(unsigned int i=0; i < number_of_01_block_cols;i++)
			  {
				  col_index_01[i] = i;
			  }
		  }
		  else
			  if (start_col_01 > number_of_01_cols/2 && end_col_01 > number_of_01_cols/2)
			  {
				  number_of_01_block_rows = end_row_01 + 1;
			      number_of_01_block_cols = number_of_01_cols - start_col_01;
				  row_index_01.resize(number_of_01_block_rows,0);
				  col_index_01.resize(number_of_01_block_cols,0);
				  for(unsigned int i=0; i < number_of_01_block_rows;i++)
				  {
					  row_index_01[i] = i;
				  }
				  for(unsigned int i=0; i < number_of_01_block_cols;i++)
				  {
					  col_index_01[i] = start_col_01 + i;
				  }
			  }
  }
  else
	  if (start_row_01 < number_of_01_rows/2 && end_row_01 > number_of_01_rows/2)
	    {
	  	  if (start_col_01 < number_of_01_cols/2 && end_col_01 < number_of_01_cols/2)
	  	  {
	  		  number_of_01_block_rows = end_row_01 + 1;
	  		  number_of_01_block_cols = end_col_01 + 1;
			  row_index_01.resize(number_of_01_block_rows,0);
			  col_index_01.resize(number_of_01_block_cols,0);
			  for(unsigned int i=0; i < number_of_01_block_rows;i++)
			  {
				  row_index_01[i] = i;
			  }
			  for(unsigned int i=0; i < number_of_01_block_cols;i++)
			  {
				  col_index_01[i] = i;
			  }
	  	  }
	  	  else
	  		  if (start_col_01 < number_of_01_cols/2 && end_col_01 > number_of_01_cols/2)
	  		  {
	  			  number_of_01_block_rows = end_row_01 + 1;
	  			  number_of_01_block_cols = end_col_01 + 1;
	  			  row_index_01.resize(number_of_01_block_rows,0);
	  			  col_index_01.resize(number_of_01_block_cols,0);
	  			  for(unsigned int i=0; i < number_of_01_block_rows;i++)
	  			  {
	  				  row_index_01[i] = i;
	  			  }
	  			  for(unsigned int i=0; i < number_of_01_block_cols;i++)
	  			  {
	  				  col_index_01[i] = i;
	  			  }
	  		  }
	  		  else
	  			  if (start_col_01 > number_of_01_cols/2 && end_col_01 > number_of_01_cols/2)
	  			  {
	  				  number_of_01_block_rows = end_row_01 + 1;
	  			      number_of_01_block_cols = number_of_01_cols - start_col_01;
	  				  row_index_01.resize(number_of_01_block_rows,0);
	  				  col_index_01.resize(number_of_01_block_cols,0);
	  				  for(unsigned int i=0; i < number_of_01_block_rows;i++)
	  				  {
	  					  row_index_01[i] = i;
	  				  }
	  				  for(unsigned int i=0; i < number_of_01_block_cols;i++)
	  				  {
	  					  col_index_01[i] = start_col_01 + i;
	  				  }
	  			  }
	    }
	  else
		  if (start_row_01 > number_of_01_rows/2 && end_row_01 > number_of_01_rows/2)
	    {
	  	  if (start_col_01 < number_of_01_cols/2 && end_col_01 < number_of_01_cols/2)
	  	  {
	  		  number_of_01_block_rows = number_of_01_rows - start_row_01;
	  		  number_of_01_block_cols = end_col_01 + 1;
			  row_index_01.resize(number_of_01_block_rows,0);
			  col_index_01.resize(number_of_01_block_cols,0);
			  for(unsigned int i=0; i < number_of_01_block_rows;i++)
			  {
				  row_index_01[i] = start_row_01 + i;
			  }
			  for(unsigned int i=0; i < number_of_01_block_cols;i++)
			  {
				  col_index_01[i] = i;
			  }
	  	  }
	  	  else
	  		  if (start_col_01 < number_of_01_cols/2 && end_col_01 > number_of_01_cols/2)
	  		  {
	  			  number_of_01_block_rows = number_of_01_rows - start_row_01;
	  			  number_of_01_block_cols = end_col_01 + 1;
	  			  row_index_01.resize(number_of_01_block_rows,0);
	  			  col_index_01.resize(number_of_01_block_cols,0);
	  			  for(unsigned int i=0; i < number_of_01_block_rows;i++)
	  			  {
	  				  row_index_01[i] = start_row_01 + i;
	  			  }
	  			  for(unsigned int i=0; i < number_of_01_block_cols;i++)
	  			  {
	  				  col_index_01[i] = i;
	  			  }
	  		  }
	  		  else
	  			  if (start_col_01 > number_of_01_cols/2 && end_col_01 > number_of_01_cols/2)
	  			  {
	  				  number_of_01_block_rows = number_of_01_rows - start_row_01;
  			      number_of_01_block_cols = number_of_01_cols - start_col_01;
  			      row_index_01.resize(number_of_01_block_rows,0);
  			      col_index_01.resize(number_of_01_block_cols,0);
  			      for(unsigned int i=0; i < number_of_01_block_rows;i++)
  			      {
  			    	  row_index_01[i] = start_row_01 + i;
  			      }
  			      for(unsigned int i=0; i < number_of_01_block_cols;i++)
  			      {
  			    	  col_index_01[i] = start_col_01 + i;
  			      }
            }
	    }


  if (start_row_10 < number_of_01_rows/2 && end_row_10 < number_of_01_rows/2)
  {
	  if (start_col_10 < number_of_01_cols/2 && end_col_10 < number_of_01_cols/2)
	  {
		  number_of_10_block_rows = end_row_10 + 1;
		  number_of_10_block_cols = end_col_10 + 1;
		  row_index_10.resize(number_of_10_block_rows,0);
		  col_index_10.resize(number_of_10_block_cols,0);
		  for(unsigned int i=0; i < number_of_10_block_rows;i++)
		  {
			  row_index_10[i] = i;
		  }
		  for(unsigned int i=0; i < number_of_10_block_cols;i++)
		  {
			  col_index_10[i] = i;
		  }
	  }
	  else
		  if (start_col_10 < number_of_01_cols/2 && end_col_10 > number_of_01_cols/2)
		  {
			  number_of_10_block_rows = end_row_10 + 1;
			  number_of_10_block_cols = end_col_10 + 1;
			  row_index_10.resize(number_of_10_block_rows,0);
			  col_index_10.resize(number_of_10_block_cols,0);
			  for(unsigned int i=0; i < number_of_10_block_rows;i++)
			  {
				  row_index_10[i] = i;
			  }
			  for(unsigned int i=0; i < number_of_10_block_cols;i++)
			  {
				  col_index_10[i] = i;
			  }
		  }
		  else
			  if (start_col_10 > number_of_01_cols/2 && end_col_10 > number_of_01_cols/2)
			  {
				  number_of_10_block_rows = end_row_10 + 1;
				  number_of_10_block_cols = number_of_01_cols - start_col_10;
				  row_index_10.resize(number_of_10_block_rows,0);
				  col_index_10.resize(number_of_10_block_cols,0);
				  for(unsigned int i=0; i < number_of_10_block_rows;i++)
				  {
					  row_index_10[i] = i;
				  }
				  for(unsigned int i=0; i < number_of_10_block_cols;i++)
				  {
					  col_index_10[i] = start_col_10 + i;
				  }
			  }
  }
  else
	  if (start_row_10 < number_of_01_rows/2 && end_row_10 > number_of_01_rows/2)
	    {
	  	  if (start_col_10 < number_of_01_cols/2 && end_col_10 < number_of_01_cols/2)
	  	  {
	  		  number_of_10_block_rows = end_row_10 + 1;
	  		  number_of_10_block_cols = end_col_10 + 1;
	  		  row_index_10.resize(number_of_10_block_rows,0);
	  		  col_index_10.resize(number_of_10_block_cols,0);
	  		  for(unsigned int i=0; i < number_of_10_block_rows;i++)
	  		  {
	  			  row_index_10[i] = i;
	  		  }
	  		  for(unsigned int i=0; i < number_of_10_block_cols;i++)
	  		  {
	  			  col_index_10[i] = i;
	  		  }
	  	  }
	  	  else
	  		  if (start_col_10 < number_of_01_cols/2 && end_col_10 > number_of_01_cols/2)
	  		  {
	  			  number_of_10_block_rows = end_row_10 + 1;
	  			  number_of_10_block_cols = end_col_10 + 1;
	  			  row_index_10.resize(number_of_10_block_rows,0);
	  			  col_index_10.resize(number_of_10_block_cols,0);
	  			  for(unsigned int i=0; i < number_of_10_block_rows;i++)
	  			  {
	  				  row_index_10[i] = i;
	  			  }
	  			  for(unsigned int i=0; i < number_of_10_block_cols;i++)
	  			  {
	  				  col_index_10[i] = i;
	  			  }
	  		  }
	  		  else
	  			  if (start_col_10 > number_of_01_cols/2 && end_col_10 > number_of_01_cols/2)
	  			  {
	  				  number_of_10_block_rows = end_row_10 + 1;
	  				  number_of_10_block_cols = number_of_01_cols - start_col_10;
	  				  row_index_10.resize(number_of_10_block_rows,0);
	  				  col_index_10.resize(number_of_10_block_cols,0);
	  				  for(unsigned int i=0; i < number_of_10_block_rows;i++)
	  				  {
	  					  row_index_10[i] = i;
	  				  }
	  				  for(unsigned int i=0; i < number_of_10_block_cols;i++)
	  				  {
	  					  col_index_10[i] = start_col_10 + i;
	  				  }
	  			  }
	    }
	  else
		  if (start_row_10 > number_of_01_rows/2 && end_row_10 > number_of_01_rows/2)
		  	    {
		  	  	  if (start_col_10 < number_of_01_cols/2 && end_col_10 < number_of_01_cols/2)
		  	  	  {
		  	  		  number_of_10_block_rows = number_of_01_rows - start_row_10;
		  	  		  number_of_10_block_cols = end_col_10 + 1;
		  			  row_index_10.resize(number_of_10_block_rows,0);
		  			  col_index_10.resize(number_of_10_block_cols,0);
		  			  for(unsigned int i=0; i < number_of_10_block_rows;i++)
		  			  {
		  				  row_index_10[i] = start_row_10 + i;
		  			  }
		  			  for(unsigned int i=0; i < number_of_10_block_cols;i++)
		  			  {
		  				  col_index_10[i] = i;
		  			  }
		  	  	  }
		  	  	  else
		  	  		  if (start_col_10 < number_of_01_cols/2 && end_col_10 > number_of_01_cols/2)
		  	  		  {
		  	  			  number_of_10_block_rows = number_of_01_rows - start_row_10;
		  	  			  number_of_10_block_cols = end_col_10 + 1;
		  	  			  row_index_10.resize(number_of_10_block_rows,0);
		  	  			  col_index_10.resize(number_of_10_block_cols,0);
		  	  			  for(unsigned int i=0; i < number_of_10_block_rows;i++)
		  	  			  {
		  	  				  row_index_10[i] = start_row_10 + i;
		  	  			  }
		  	  			  for(unsigned int i=0; i < number_of_10_block_cols;i++)
		  	  			  {
		  	  				  col_index_10[i] = i;
		  	  			  }
		  	  		  }
		  	  		  else
		  	  			  if (start_col_10 > number_of_01_cols/2 && end_col_10 > number_of_01_cols/2)
		  	  			  {
		  	  				  number_of_10_block_rows = number_of_01_rows - start_row_10;
		  	  				  number_of_10_block_cols = number_of_01_cols - start_col_10;
		  	  				  row_index_10.resize(number_of_10_block_rows,0);
		  	  				  col_index_10.resize(number_of_10_block_cols,0);
		  	  				  for(unsigned int i=0; i < number_of_10_block_rows;i++)
		  	  				  {
		  	  					  row_index_10[i] = start_row_10 + i;
		  	  				  }
		  	  				  for(unsigned int i=0; i < number_of_10_block_cols;i++)
		  	  				  {
		  	  					  col_index_10[i] = start_col_10 + i;
		  	  				  }

		  	  			  }
		  	    }

  std::vector<int> lh_index_01(number_of_01_block_rows,0);
  std::vector<int> lh_index_10(number_of_10_block_rows,0);
  std::vector<int> rh_index_01(number_of_01_block_cols,0);
  std::vector<int> rh_index_10(number_of_10_block_cols,0);

  for (unsigned int i=0; i < number_of_01_block_rows; i++)
  {
     lh_index_01[i] = i;
  }
  for (unsigned int i=0; i < number_of_10_block_rows; i++)
  {
     lh_index_10[i] = i;
  }
  for (unsigned int i=0; i < number_of_01_block_cols; i++)
  {
     rh_index_01[i] = i;
  }
  for (unsigned int i=0; i < number_of_10_block_cols; i++)
  {
     rh_index_10[i] = i;
  }

  PetscMatrixParallelComplex* alpha_i = new PetscMatrixParallelComplex(number_of_10_block_rows, number_of_10_block_cols,input_simulation->get_simulation_domain()->get_communicator());
  alpha_i->consider_as_full();
  alpha_i->allocate_memory();
  alpha_i->assemble();

  PetscMatrixParallelComplex* beta_i = new PetscMatrixParallelComplex(number_of_01_block_rows, number_of_01_block_cols,input_simulation->get_simulation_domain()->get_communicator());
  beta_i->consider_as_full();
  beta_i->allocate_memory();
  beta_i->assemble();

  H10->get_submatrix(row_index_10,col_index_10,MAT_INITIAL_MATRIX,alpha_i);

  H01->get_submatrix(row_index_01,col_index_01,MAT_INITIAL_MATRIX,beta_i);

  //Prasad: lhmatrix and rhmatrix are rectangular matrices and store the block from H01 or H10 for multiplication

  //int start_rows, end_rows;
  std::vector<int> local_col1(number_of_10_block_rows,0);
  std::vector<int> non_local_col1(number_of_10_block_rows,0);

  PetscMatrixParallelComplex* lhmatrix_alpha = new PetscMatrixParallelComplex(number_of_10_block_rows,number_of_01_cols,input_simulation->get_simulation_domain()->get_communicator());
  lhmatrix_alpha->consider_as_full();
  lhmatrix_alpha->allocate_memory();
  lhmatrix_alpha->set_to_zero();

  std::vector<int> local_col2(number_of_01_block_rows,0);
  std::vector<int> non_local_col2(number_of_01_block_rows,0);
  PetscMatrixParallelComplex* lhmatrix_beta = new PetscMatrixParallelComplex(number_of_01_block_rows,number_of_01_cols,input_simulation->get_simulation_domain()->get_communicator());
  lhmatrix_beta->consider_as_full();
  lhmatrix_beta->allocate_memory();
  lhmatrix_beta->set_to_zero();

  std::vector<int> local_col3(number_of_01_rows,0);
  std::vector<int> non_local_col3(number_of_01_rows,0);
  PetscMatrixParallelComplex* rhmatrix_alpha = new PetscMatrixParallelComplex(number_of_01_rows,number_of_10_block_cols,input_simulation->get_simulation_domain()->get_communicator());
  rhmatrix_alpha->consider_as_full();
  rhmatrix_alpha->allocate_memory();
  rhmatrix_alpha->set_to_zero();

  PetscMatrixParallelComplex* rhmatrix_beta = new PetscMatrixParallelComplex(number_of_01_rows,number_of_01_block_cols,input_simulation->get_simulation_domain()->get_communicator());

  rhmatrix_beta->consider_as_full();
  rhmatrix_beta->allocate_memory();
  rhmatrix_beta->set_to_zero();

  //Prasad: storage_mat1 and storage_mat2 store the solution from linear system of equations
  PetscMatrixParallelComplex* storage_mat1 = new PetscMatrixParallelComplex(number_of_01_rows,number_of_10_block_cols,input_simulation->get_simulation_domain()->get_communicator());
  storage_mat1->consider_as_full();
  storage_mat1->allocate_memory();

  PetscMatrixParallelComplex* storage_mat2 = new PetscMatrixParallelComplex(number_of_01_rows,number_of_01_block_cols,input_simulation->get_simulation_domain()->get_communicator());
  storage_mat2->consider_as_full();
  storage_mat2->allocate_memory();

  PetscMatrixParallelComplex* epsilon_is = new PetscMatrixParallelComplex(*H00);
  epsilon_is->matrix_convert_dense();

  PetscMatrixParallelComplex* epsilon_i = new PetscMatrixParallelComplex(*H00);
  epsilon_i->matrix_convert_dense();
  epsilon_i->assemble();

  unsigned int max_iterations = input_options.get_option( "maximum_Sancho_iterations", 35);
  std::string tic_toc_prefix_2 = NEMOUTILS_PREFIX("Propagation(\"" + tic_toc_prefix+ "\")::Sancho_Rubio_retarded_green_iterations ");
  NemoUtils::tic(tic_toc_prefix_2);
  std::string prefix_2 = "Propagation(\"" + input_simulation->get_name()+ "\")::Sancho_Rubio_retarded_green_iterations ";

  double tolerance = input_options.get_option("Sancho_Rubio_tolerance",double(1E-5));
  double trace_error = 100;
  double old_trace = 100;
  for (unsigned int i = 0; i < max_iterations && trace_error > tolerance  ; i++)
  {
    PetscMatrixParallelComplex* alpha_i1    = new PetscMatrixParallelComplex(*alpha_i);
    PetscMatrixParallelComplex* beta_i1     = new PetscMatrixParallelComplex(*beta_i);
    PetscMatrixParallelComplex* epsilon_i1s = new PetscMatrixParallelComplex(*epsilon_is);

    //Prasad: (omega - eps(i-1))
    PetscMatrixParallelComplex* epsilon_i1 = new PetscMatrixParallelComplex(*epsilon_i);
    epsilon_i1->assemble();
    tmp1->add_matrix(*epsilon_i1, SAME_NONZERO_PATTERN, -x);
    tmp1->add_matrix(*omega, SAME_NONZERO_PATTERN, x);
    tmp1->assemble();

    rhmatrix_alpha->set_block_from_matrix(*alpha_i1,row_index_10,rh_index_10,INSERT_VALUES);
    rhmatrix_alpha->assemble();
    if(PropOptionInterface->get_debug_output())
    {
        rhmatrix_alpha->save_to_matlab_file("rhmatrix_alpha"+temp_file_string+".m");
    }


    //Prasad: Replaced inversion with a linear solver
    //Solve (omega - eps(i-1)^-1*alpha(i-1)
    LinearSolverPetscComplex solver1(*tmp1, *rhmatrix_alpha, storage_mat1);
    InputOptions options_for_linear_solver;
    string value("petsc");
    if (input_options.check_option("linear_system_solver"))
      value = input_options.get_option("linear_system_solver", string("mumps"));
    string key("solver");
    options_for_linear_solver[key] = value;
    solver1.set_options(options_for_linear_solver);
    std::string tic_toc_prefix_M_matrix_compute_inverse = NEMOUTILS_PREFIX("Propagation(\""+ tic_toc_prefix+ "\")::Sancho_Rubio_retarded_green_iterations::Linearsolver1");
    NemoUtils::tic(tic_toc_prefix_M_matrix_compute_inverse);
    solver1.solve();
    NemoUtils::toc(tic_toc_prefix_M_matrix_compute_inverse);

    if(PropOptionInterface->get_debug_output())
    {
    	storage_mat1->assemble();
    	storage_mat1->save_to_matlab_file("storage_mat1.m");
    }

    delete alpha_i;
    alpha_i = NULL;
    //rhmatrix_alpha->set_to_zero();

    lhmatrix_alpha->set_block_from_matrix(*alpha_i1,lh_index_10,col_index_10,INSERT_VALUES);

    //Prasad: Solve alpha(i-1)*(omega-eps(i-1))^-1*alpha(i-1)
    lhmatrix_alpha->assemble();

    if (PropOptionInterface->get_debug_output())
    {
    	lhmatrix_alpha->save_to_matlab_file("lhmatrix_alpha"+temp_file_string+".m");
    }

    storage_mat1->assemble();
    PetscMatrixParallelComplex::mult(*lhmatrix_alpha, *storage_mat1, &alpha_i);

    if (PropOptionInterface->get_debug_output())
    {
    	alpha_i->assemble();
    	alpha_i->save_to_matlab_file("alpha_i"+temp_file_string+".m");
    }

    rhmatrix_beta->set_block_from_matrix(*beta_i1,row_index_01,rh_index_01,INSERT_VALUES);

    if (PropOptionInterface->get_debug_output())
    {
    	rhmatrix_beta->assemble();
    	rhmatrix_beta->save_to_matlab_file("rhmatrix_beta"+temp_file_string+".m");
    }

    //Prasad: Solve (omega-eps(i-1))^-1*beta(i-1)
    LinearSolverPetscComplex solver2(*tmp1, *rhmatrix_beta, storage_mat2);
    solver2.set_options(options_for_linear_solver);
    std::string tic_toc_prefix_M_matrix_compute_inverse2= NEMOUTILS_PREFIX("Propagation(\""+tic_toc_prefix+"\")::Sancho_Rubio_retarded_green_iterations::Linearsolver2");
    NemoUtils::tic(tic_toc_prefix_M_matrix_compute_inverse2);
    solver2.solve();
    NemoUtils::toc(tic_toc_prefix_M_matrix_compute_inverse2);

    delete beta_i;
    beta_i = NULL;

    lhmatrix_beta->set_block_from_matrix(*beta_i1,lh_index_01,col_index_01,INSERT_VALUES);

    //Prasad: Solve beta(i-1)*(omega-eps(i-1))^-1*beta(i-1)
    storage_mat2->assemble();

    if (PropOptionInterface->get_debug_output())
    {
    	storage_mat2->save_to_matlab_file("storage_mat2"+temp_file_string+".m");
    }

    lhmatrix_beta->assemble();

    if (PropOptionInterface->get_debug_output())
    {
    	lhmatrix_beta->save_to_matlab_file("lhmatrix_beta"+temp_file_string+".m");
    }

    PetscMatrixParallelComplex::mult(*lhmatrix_beta, *storage_mat2, &beta_i);

    if (PropOptionInterface->get_debug_output())
    {
    	beta_i->assemble();
    	beta_i->save_to_matlab_file("beta_i"+temp_file_string+".m");
    }

    tmp1->set_to_zero();

    //Prasad: Solve beta(i-1)*(omega-eps(i-1))^-1*alpha(i-1)
    PetscMatrixParallelComplex* t1 = NULL;
    storage_mat1->assemble();
    PetscMatrixParallelComplex::mult(*lhmatrix_beta, *storage_mat1, &t1);

    if (PropOptionInterface->get_debug_output())
    {
    	t1->assemble();
    	t1->save_to_matlab_file("t1_first"+temp_file_string+".m");
    }

    tmp1->set_block_from_matrix(*t1,row_index_01,col_index_10,INSERT_VALUES);

    if (PropOptionInterface->get_debug_output())
    {
    	tmp1->assemble();
    	tmp1->save_to_matlab_file("tmp1"+temp_file_string+".m");
    }

    delete t1;
    t1 = NULL;
    //lhmatrix->set_to_zero();

    //Prasad: Solve alpha(i-1)*(omega-eps(i-1))^-1*beta(i-1)
    lhmatrix_alpha->assemble();
    storage_mat2->assemble();
    PetscMatrixParallelComplex::mult(*lhmatrix_alpha, *storage_mat2, &t1);

    if (PropOptionInterface->get_debug_output())
    {
    	t1->assemble();
    	t1->save_to_matlab_file("t1_second"+temp_file_string+".m");
    }

    //Prasad: Solve eps(i-1) + alpha(i-1)*(omega-eps(i-1))^-1*beta(i-1) + beta(i-1)*(omega-eps(i-1))^-1*alpha(i-1)
    tmp2->set_block_from_matrix(*t1,row_index_10,col_index_01,INSERT_VALUES);

    if (PropOptionInterface->get_debug_output())
    {
    	tmp2->assemble();
    	tmp2->save_to_matlab_file("tmp2"+temp_file_string+".m");
    }

    tmp1->add_matrix(*tmp2,SAME_NONZERO_PATTERN,x);
    tmp1->add_matrix(*epsilon_i1, SAME_NONZERO_PATTERN, x);
    delete epsilon_i;
    epsilon_i = NULL;

    //Prasad: Solve eps_s(i-1) + alpha(i-1)*(omega-eps(i-1))^-1*beta(i-1)
    epsilon_i = new PetscMatrixParallelComplex(*tmp1);

    if (PropOptionInterface->get_debug_output())
    {
    	epsilon_i->assemble();
    	epsilon_i->save_to_matlab_file("epsilon_i"+temp_file_string+".m");
    }

    tmp1->set_to_zero();
    tmp1->set_block_from_matrix(*t1,row_index_10,col_index_01,INSERT_VALUES);
    tmp1->add_matrix(*epsilon_i1s, SAME_NONZERO_PATTERN, x);
    tmp1->assemble();

    //check tolerance
    double new_trace = tmp1->get_trace().real();
    trace_error = std::abs((new_trace-old_trace)/old_trace);
    //cerr << "Iteration number: " << i << " trace error " << trace_error << " \n";
    old_trace = new_trace;

    delete epsilon_is;
    epsilon_is = NULL;
    epsilon_is = new PetscMatrixParallelComplex(*tmp1);

    if (PropOptionInterface->get_debug_output())
    {
    	epsilon_is->assemble();
    	epsilon_is->save_to_matlab_file("epsilon_is"+temp_file_string+".m");
    }

    tmp1->set_to_zero();
    delete t1;
    t1 = NULL;
    delete alpha_i1;
    alpha_i1 = NULL;
    delete beta_i1;
    beta_i1 = NULL;
    delete epsilon_i1s;
    epsilon_i1s = NULL;
    delete epsilon_i1;
    epsilon_i1 = NULL;
  }
  NemoUtils::toc(tic_toc_prefix_2);
  //6. Compute the surface green's function and return in result.
  //result = new PetscMatrixParallelComplex(*tmp1);
  tmp1->add_matrix(*epsilon_is, SAME_NONZERO_PATTERN, -x);
  tmp1->add_matrix(*omega, SAME_NONZERO_PATTERN, x);

  //J.C. Lapack seems more stable for inversions.
  //PetscMatrixParallelComplex::invert(*tmp1,&result,"mumps");
  //PetscMatrixParallelComplex* result = NULL;
  tmp1->invertLapack(*tmp1, &result);
  if (PropOptionInterface->get_debug_output())
    result->save_to_matlab_file(input_simulation->get_name()+"_Sancho_gR"+temp_file_string+".m");
  if(coupling_DOFmap!=temp_pointer)
    delete coupling_DOFmap;
  coupling_DOFmap = NULL;
  delete tmp1;
  tmp1 = NULL;
  delete tmp2;
  tmp2 = NULL;
  delete H00;
  H00 = NULL;
  delete H01;
  H01 = NULL;
  delete H10;
  H10 = NULL;
  delete coupling;
  coupling = NULL;
  delete omega;
  omega = NULL;
  delete alpha_i;
  alpha_i = NULL;
  delete beta_i;
  beta_i = NULL;
  delete epsilon_is;
  epsilon_is = NULL;
  delete epsilon_i;
  epsilon_i = NULL;
  delete lhmatrix_alpha;
  lhmatrix_alpha = NULL;
  delete lhmatrix_beta;
  lhmatrix_beta = NULL;
  delete rhmatrix_alpha;
  rhmatrix_alpha = NULL;
  delete rhmatrix_beta;
  rhmatrix_beta = NULL;
  delete storage_mat1;
  storage_mat1 = NULL;
  delete storage_mat2;
  storage_mat2 = NULL;

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::solve_retarded_Green_SanchoRubio(Simulation* this_simulation,Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_retarded_Green_SanchoRubio ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Greensolver(\""+this_simulation->get_name()+"\")::solve_retarded_Green_SanchoRubio ";
  const InputOptions& input_options=this_simulation->get_options();
  std::string variable_name=output_Propagator->get_name()+std::string("_lead_domain");
  std::string neighbor_domain_name;
  if (input_options.check_option(variable_name))
    neighbor_domain_name=input_options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(prefix+" define \""+variable_name+"\"\n");
  const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);
  Sancho_Rubio_retarded_green(this_simulation,output_Propagator,neighbor_domain,momentum_point,result);
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::extract_coupling_Hamiltonian_RGF(Simulation* this_simulation, int num_target_rows, int num_target_cols, PetscMatrixParallelComplex*& coupling_Hamiltonian,
  PetscMatrixParallelComplex*& result, PetscMatrixParallelComplex* sigmaR)
{
  std::string tic_toc_prefix = "PropagationUtilities(\"" + this_simulation->get_name() + "\")::extract_coupling_Hamiltonian_RGF ";
  NemoUtils::tic(tic_toc_prefix);

  unsigned int number_of_super_rows;
  unsigned int number_of_super_cols;
  int start_own_super_rows;
  int end_own_super_rows_p1;
  unsigned int number_of_rows1;
  unsigned int number_of_cols1;

  number_of_super_rows = coupling_Hamiltonian->get_num_rows();
  number_of_super_cols = coupling_Hamiltonian->get_num_cols();
  coupling_Hamiltonian->get_ownership_range(start_own_super_rows, end_own_super_rows_p1);
  NEMO_ASSERT(number_of_super_rows == number_of_super_cols, tic_toc_prefix + "rectangular matrix received for the super-domain\n");

  //these are the dimensions of this domain's matrices:
  number_of_rows1 = num_target_rows;//half_way_g_matrix->get_num_rows();
  number_of_cols1 = num_target_cols;//half_way_g_matrix->get_num_cols();

  //these are the dimensions of the neighbor domain matrices:
  int number_of_rows2 = number_of_super_rows - number_of_rows1;
  int number_of_cols2 = number_of_super_cols - number_of_cols1;
  NEMO_ASSERT(number_of_rows2 == number_of_cols2, tic_toc_prefix + "rectangular matrix received for neighbor domain\n");
  if (coupling_Hamiltonian->get_num_rows() <= number_of_rows1)
  {
    number_of_rows1 = number_of_super_rows;
    number_of_cols1 = 0;//number_of_super_cols;
    number_of_rows2 = number_of_super_rows;
    number_of_cols2 = number_of_super_cols;
  }
  //1. get the submatrix (upper right block) from coupling Hamiltonian
  std::vector<int> temp_rows(number_of_rows1);
  std::vector<int> temp_cols(number_of_cols2);
  for (unsigned int i = 0; i < number_of_rows1; i++)
    temp_rows[i] = i;
  for (int i = 0; i < number_of_cols2; i++)
    temp_cols[i] = i + number_of_cols1;
  result = new PetscMatrixParallelComplex(number_of_rows1, number_of_cols2, this_simulation->get_simulation_domain()->get_communicator());
  result->set_num_owned_rows(number_of_rows1);
  vector<int> rows_diagonal(number_of_rows1, 0);
  vector<int> rows_offdiagonal(number_of_rows1, 0);

  //if sigmaR exists then the sparsity pattern must be setup to reflect
  //so that sigmaR and coupling can be added
  if(sigmaR==NULL)
  {
    for (unsigned int i = 0; i < number_of_rows1; i++)
    {
      rows_diagonal[i] = coupling_Hamiltonian->get_nz_diagonal(i);
      rows_offdiagonal[i] = coupling_Hamiltonian->get_nz_offdiagonal(i);
    }
  }
  else
  {
    for (unsigned int i = 0; i < number_of_rows1; i++)
     {
       rows_diagonal[i] = std::max(coupling_Hamiltonian->get_nz_diagonal(i),
                                   sigmaR->get_nz_diagonal(i));
       rows_offdiagonal[i] = coupling_Hamiltonian->get_nz_offdiagonal(i);
     }
  }
  for (unsigned int i = 0; i < number_of_rows1; i++)
    result->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);
  coupling_Hamiltonian->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, result);

  if(sigmaR!=NULL)
  {
    result->matrix_convert_dense();
    sigmaR->matrix_convert_dense();
    result->add_matrix(*sigmaR,DIFFERENT_NONZERO_PATTERN,cplx(1.0,0.0));
  }

  result->assemble();
  delete coupling_Hamiltonian;
  coupling_Hamiltonian = NULL;

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::convert_self_energy_to_device_size_RGF(Simulation* this_simulation, PetscMatrixParallelComplex*& small_self_matrix, const DOFmapInterface& small_DOFmap, const DOFmapInterface& full_DOFmap,
  PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities(\"" + this_simulation->get_name() + "\")::convert_self_energy_to_device_size_RGF ";
  NemoUtils::tic(tic_toc_prefix);

  int start_own_self_rows;
  int end_own_self_rows;
  small_self_matrix->get_ownership_range(start_own_self_rows, end_own_self_rows);
  const int number_of_rows = full_DOFmap.get_global_dof_number();
  result = new PetscMatrixParallelComplex(number_of_rows, number_of_rows,
    this_simulation->get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
  std::vector<int> row_indexes;
  full_DOFmap.get_local_row_indexes(&row_indexes);
  const int number_of_own_super_rows = row_indexes.size();
  result->set_num_owned_rows(number_of_own_super_rows);
  std::map<unsigned int, unsigned int> full_self_subindex_map;
  full_DOFmap.get_sub_DOF_index_map(&small_DOFmap, full_self_subindex_map);
  std::map<unsigned int, unsigned int>::iterator temp_it1 = full_self_subindex_map.begin();
  for (int i = 0; i < number_of_own_super_rows; i++)
  {
    temp_it1 = full_self_subindex_map.find(i);
    if (temp_it1 != full_self_subindex_map.end())
    {
      int j = temp_it1->second;//translated matrix index
      if (j<start_own_self_rows || j>end_own_self_rows)
        result->set_num_nonzeros(i, 0, 0);
      else
        result->set_num_nonzeros(i, small_self_matrix->get_nz_diagonal(j), small_self_matrix->get_nz_offdiagonal(j));
    }
    else
      result->set_num_nonzeros(i, 0, 0);
  }
  result->allocate_memory();
  result->set_to_zero();

  //fill the self-energy matrix of the device domain with elements of the smaller self-energy
  temp_it1 = full_self_subindex_map.begin();
  for (; temp_it1 != full_self_subindex_map.end(); temp_it1++)
  {
    unsigned int i = temp_it1->first;
    int j = temp_it1->second;//translated row index
    if (j >= start_own_self_rows || j <= end_own_self_rows)
    {
      std::map<unsigned int, unsigned int>::iterator temp_it2 = full_self_subindex_map.begin();
      for (; temp_it2 != full_self_subindex_map.end(); temp_it2++)
      {
        unsigned int ii = temp_it2->first;
        unsigned int jj = temp_it2->second;//translated column index
        if (jj < small_self_matrix->get_num_cols())
          result->set(i, ii, small_self_matrix->get(j, jj));
      }
    }
  }
  result->assemble();

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::calculate_gamma_given_sigma(Simulation* this_simulation, PetscMatrixParallelComplex*& Sigma, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities for \"" + this_simulation->get_name() + "\" calculate_gamma_given_sigma ";
  NemoUtils::tic(tic_toc_prefix);
/*
  if(Sigma->check_if_sparse())
   	  std::cout<<"has sparse"<<endl;
   if(Sigma->is_scalar)
   	  std::cout<<"has scalar"<<endl;
*/

  PropagationOptionsInterface* PropOptionInterface = get_PropagationOptionsInterface(this_simulation);

  PetscMatrixParallelComplex* temp_Sigma = new PetscMatrixParallelComplex(Sigma->get_num_cols(),
    Sigma->get_num_rows(),
    Sigma->get_communicator());

  Sigma->hermitian_transpose_matrix(*temp_Sigma, MAT_INITIAL_MATRIX); //sigma'

  result = new PetscMatrixParallelComplex(*Sigma); //copy sigma to result
  result->add_matrix(*temp_Sigma, DIFFERENT_NONZERO_PATTERN, std::complex<double>(-1.0, 0.0)); //sigma-sigma'
  delete temp_Sigma;
  temp_Sigma = NULL;
  (*result) *= std::complex<double>(0.0, 1.0); //gamma=i*(sigma-sigma')

  if (PropOptionInterface->get_debug_output())
  {
    Sigma->save_to_matlab_file("sigma.m");
    result->save_to_matlab_file("gamma.m");
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::core_interdomain_Green_RGF(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum_point,
  PetscMatrixParallelComplex*& exact_matrix,
  PetscMatrixParallelComplex*& half_way_g_matrix,
  PetscMatrixParallelComplex*& coupling,
  PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities for \"" + this_simulation->get_name() + "\" core_interdomain_Green_RGF ";
  NemoUtils::tic(tic_toc_prefix);

  PropagationOptionsInterface* PropOptionInterface = get_PropagationOptionsInterface(this_simulation);

  //do the actual multiplication(Eq.(64) of J.Appl.Phys. 81, 7845 (1997))
  PetscMatrixParallelComplex* temp_matrix = NULL;
  //std::string tic_toc_mult1= tic_toc_prefix+" half_way_g x couplingH";
  //NemoUtils::tic(tic_toc_mult1);
  Greensolver::Mat_mult_method mat_mult = Greensolver::petsc;

  std::string invert_solver_option;
  std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;
  if (PropOptionInterface->get_offload_solver_initialized() && PropOptionInterface->get_offload_solver()->offload)
  {
    c_it_momentum_offload_info = PropOptionInterface->get_offload_solver()->offloading_momentum_map.find(momentum_point);

    if (c_it_momentum_offload_info->second.offload_to == ResourceUtils::GPU)
      mat_mult = Greensolver::magma;
    else
      mat_mult = Greensolver::petsc;
  }

#ifdef MAGMA_ENABLE
  magmaDoubleComplex* temp_gpu = NULL;
#endif

  if (mat_mult == Greensolver::petsc)
  {
    PropagationUtilities::supress_noise(this_simulation, half_way_g_matrix);
    PropagationUtilities::supress_noise(this_simulation, coupling);
    PetscMatrixParallelComplex::mult(*half_way_g_matrix, *coupling, &temp_matrix);
    PropagationUtilities::supress_noise(this_simulation, temp_matrix);
  }
  else
  {
#ifdef MAGMA_ENABLE
    PetscMatrixParallelComplex::multMAGMA(*half_way_g_matrix, MagmaNoTrans, NULL,
      *coupling, MagmaNoTrans, NULL,
      &temp_matrix, false, &temp_gpu);
#else
    NEMO_EXCEPTION("solve_interdomain_Green_RGF: Can't use MAGMA without compiling with MAGMA!");
#endif
  }
  //NemoUtils::toc(tic_toc_mult1);
  //delete coupling;
  //coupling = NULL;
  //std::string tic_toc_mult2= tic_toc_prefix+" temp_matrix x full_exact";
  //NemoUtils::tic(tic_toc_mult2);
  if (mat_mult == Greensolver::petsc)
  {
    PropagationUtilities::supress_noise(this_simulation, exact_matrix);
    PetscMatrixParallelComplex::mult(*temp_matrix, *exact_matrix, &result);
    PropagationUtilities::supress_noise(this_simulation, result);
  }
  else
  {
#ifdef MAGMA_ENABLE
    PetscMatrixParallelComplex::multMAGMA(*temp_matrix, MagmaNoTrans, &temp_gpu,
      *exact_matrix, MagmaNoTrans, NULL,
      &result, true, NULL);
    magma_free(temp_gpu);
#else
    NEMO_EXCEPTION("solve_interdomain_Green_RGF: Can't use MAGMA without compiling with MAGMA!");
#endif
  }
  result->consider_as_full();
  delete temp_matrix;
  temp_matrix = NULL;

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::solve_interdomain_Green_RGF(Simulation* this_simulation,Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_interdomain_Green_RGF ";
  NemoUtils::tic(tic_toc_prefix);
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  const InputOptions& input_options=this_simulation->get_options();
  {
    //NOTE about the domains: this domain is "1", target_domain is "L", coupling_domain="L-1"
    //1a. get the Hamiltonian and the DOFmap of the full device
    std::string variable_name = "full_device_Hamiltonian_constructor";
    NEMO_ASSERT(input_options.check_option(variable_name),tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string full_H_constructor_name = input_options.get_option(variable_name,std::string(""));
    Simulation* full_H_constructor = this_simulation->find_simulation(full_H_constructor_name);
    NEMO_ASSERT(full_H_constructor!=NULL,tic_toc_prefix+"simulation \""+full_H_constructor_name+"\" not found\n");
    const DOFmapInterface& device_DOFmap = full_H_constructor->get_const_dof_map(this_simulation->get_const_simulation_domain());

    //1b. get some device matrix characteristics
    unsigned int number_of_super_rows = device_DOFmap.get_global_dof_number();
    unsigned int number_of_super_cols = number_of_super_rows;

    NEMO_ASSERT(number_of_super_rows==number_of_super_cols,tic_toc_prefix+"rectangular matrix received for the super-domain\n");

    //2. get the DOFmap of the target domain "L"
    variable_name = "target_domain_Hamiltonian_constructor";
    NEMO_ASSERT(input_options.check_option(variable_name),tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string target_H_constructor_name = input_options.get_option(variable_name,std::string(""));
    Simulation* target_H_constructor = this_simulation->find_simulation(target_H_constructor_name);
    NEMO_ASSERT(target_H_constructor!=NULL,tic_toc_prefix+"simulation \""+target_H_constructor_name+"\" not found\n");
    const Domain* target_domain=target_H_constructor->get_const_simulation_domain();
    if(input_options.check_option("target_domain"))
    {
      std::string temp_domain_name=input_options.get_option("target_domain",std::string(""));
      target_domain=Domain::get_domain(temp_domain_name);
      NEMO_ASSERT(target_domain!=NULL,tic_toc_prefix+"have not found target_domain \""+temp_domain_name+"\"\n");
    }
    const DOFmapInterface& target_DOFmap = target_H_constructor->get_const_dof_map(target_domain);
    //3. if the domains "1" and "L" are equal - throw exception
    if(target_domain==this_simulation->get_const_simulation_domain())
      throw std::invalid_argument(tic_toc_prefix+"domain of \""+target_H_constructor_name+"\" and this domain are equal\n");

    //std::string tic_toc_couplingH= tic_toc_prefix+" get Hamiltonian and DOFmap for L-1";
    //NemoUtils::tic(tic_toc_couplingH);
    //4. get the Hamiltonian and the DOFmap of the coupling domain "L-1"
    variable_name ="coupling_domain_Hamiltonian_constructor"; //"L-1" in Eq.64 of J. Appl. Phys. 81, 7845 (1997)
    if(!input_options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string coupling_H_constructor_name=input_options.get_option(variable_name,std::string(""));
    Simulation* coupling_H_constructor = this_simulation->find_simulation(coupling_H_constructor_name);
    const Domain* coupling_domain = coupling_H_constructor->get_const_simulation_domain();
    if(input_options.check_option("coupling_domain"))
    {
      std::string temp_domain_name=input_options.get_option("coupling_domain",std::string(""));
      coupling_domain=Domain::get_domain(temp_domain_name);
      NEMO_ASSERT(coupling_domain!=NULL,tic_toc_prefix+"have not found target_domain \""+temp_domain_name+"\"\n");
    }
    DOFmapInterface& coupling_DOFmap = coupling_H_constructor->get_dof_map(coupling_domain);

    DOFmapInterface* large_coupling_DOFmap=&coupling_DOFmap; //will store the DOFmap for the matrix of the target (L) domain and the coupling (L-1) domain
    
    DOFmapInterface* temp_DOFmap_pointer=large_coupling_DOFmap;

    //NemoUtils::toc(tic_toc_couplingH);


    //========================================
    //Yu He: new version -- avoid super matrix
    //----------------------------------------
    //std::string tic_toc_HLLm1= tic_toc_prefix+" get interdomain Hamiltonian and DOFmap for L,L-1";
    //NemoUtils::tic(tic_toc_HLLm1);


    //5a. get the interdomain Hamiltonian and the large_interdomain DOFmap ( H_(L,L-1) )
    PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;  //should become a matrix in the upper right corner
    PetscMatrixParallelComplex* temp_overlap = NULL;
    {
      std::vector<NemoMeshPoint> sorted_momentum;
      QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum,input_options,PropInterface->get_momentum_mesh_types(),target_H_constructor);
      target_H_constructor->get_data(std::string("Hamiltonian"), sorted_momentum, coupling_domain, coupling_Hamiltonian,large_coupling_DOFmap,target_domain);
      target_H_constructor->get_data(std::string("overlap_matrix_coupling"),sorted_momentum,coupling_domain, temp_overlap, large_coupling_DOFmap,target_domain);
      if(temp_DOFmap_pointer!=large_coupling_DOFmap)
      delete large_coupling_DOFmap;
      //coupling_Hamiltonian->save_to_matlab_file("Ham_int.m");
      //temp_overlap->save_to_matlab_file("S_int.m");

      if(temp_overlap!=NULL)
      {
        std::complex<double> energy = std::complex<double>(read_energy_from_momentum(this_simulation,momentum_point,output_Propagator),0.0);
        //cerr<<energy;
        *temp_overlap *= -energy; //-ES
        coupling_Hamiltonian->add_matrix(*temp_overlap, DIFFERENT_NONZERO_PATTERN, std::complex<double>(1.0, 0.0));
        //coupling_Hamiltonian->save_to_matlab_file("T_int.m");
      }
    }

    //NemoUtils::toc(tic_toc_HLLm1);

    
    //6. get the exact_greens_function (G_L-1,1 in Eq. 64 of J. Appl. Phys. 81, 7845 (1997))
    variable_name="exact_greens_function";
    if(!input_options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string exact_green_name = input_options.get_option(variable_name,std::string(""));
    GreensfunctionInterface* exact_green_source=find_source_of_Greensfunction(this_simulation,exact_green_name);
    PetscMatrixParallelComplex* exact_matrix=NULL;
    if(PropOptionInterface->get_particle_type_is_Fermion())
      exact_green_source->get_Greensfunction(momentum_point,exact_matrix,&coupling_DOFmap,&(PropOptionInterface->get_Hamilton_Constructor()->get_const_dof_map(this_simulation->get_const_simulation_domain())),
                                             NemoPhys::Fermion_retarded_Green);
    else
      exact_green_source->get_Greensfunction(momentum_point,exact_matrix,&coupling_DOFmap,&(PropOptionInterface->get_Hamilton_Constructor()->get_const_dof_map(this_simulation->get_const_simulation_domain())),
                                             NemoPhys::Boson_retarded_Green);
    int start_own_exact_rows;
    int end_own_exact_rows;
    exact_matrix->get_ownership_range(start_own_exact_rows,end_own_exact_rows);

    //7. get the half_way_greens_function and the DOFmap
    variable_name="half_way_greens_function";
    if(!input_options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string half_way_green_name = input_options.get_option(variable_name,std::string(""));
    GreensfunctionInterface* half_way_green_source=find_source_of_Greensfunction(this_simulation,half_way_green_name);
    PetscMatrixParallelComplex* half_way_g_matrix=NULL;
    if(PropOptionInterface->get_particle_type_is_Fermion())
      half_way_green_source->get_Greensfunction(momentum_point,half_way_g_matrix,&target_DOFmap,NULL,NemoPhys::Fermion_retarded_Green);
    else
      half_way_green_source->get_Greensfunction(momentum_point,half_way_g_matrix,&target_DOFmap,NULL,NemoPhys::Boson_retarded_Green);

    //8. do the actual multiplication (Eq.(64) of J. Appl. Phys. 81, 7845 (1997))
    PetscMatrixParallelComplex* coupling = NULL;
    extract_coupling_Hamiltonian_RGF(this_simulation, half_way_g_matrix->get_num_rows(), half_way_g_matrix->get_num_cols(), coupling_Hamiltonian, coupling);
    core_interdomain_Green_RGF(this_simulation, momentum_point, exact_matrix, half_way_g_matrix, coupling, result);
    delete coupling;
    coupling = NULL;

    //----------------------------------------
    //Yu He: end of new version
    //========================================
  }

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::core_retarded_Green_back_RGF(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum_point, 
                                                        PetscMatrixParallelComplex*& exact_neighbor_matrix,
                                                        PetscMatrixParallelComplex*& half_way_g_matrix,
                                                        PetscMatrixParallelComplex*& coupling,
                                                        PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities(\"" + this_simulation->get_name() + "\")::core_retarded_Green_back_RGF ";
  NemoUtils::tic(tic_toc_prefix);

  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  const InputOptions& input_options=this_simulation->get_options();
  //G_LL = g_LL + g_LL (t_LL-1) G_L-1L-1 (t_L-1L) gLL
  //for this equation, we are avoid using the super-matrix concept of the contact self-energy methods again, i.e creating matrices of the size of both domains...
  //if we used it, we would need four super matrices (for g_LL, G_L-1L-1, t_LL-1 and the result)
  //Note: g_LL is of this domain and therefore in the upper left corner of the super-matrix
  // G_L-1L-1 is of the neighbor domain and therefore in the lower right corner of the super-matrix
  // t_LL-1 (coupling_Hamiltonian) is in the upper right corner of the super-matrix
  //construct the domain3=domain1+domain2 matrices (i.e. "super-matrices")
  //this code uses that super-matrix concept only implicitly...

  //2. perform the multiplication
  Greensolver::Mat_mult_method mat_mult = Greensolver::petsc;

  // Obtain propagator offload map to determine if this momentum should be offloaded
  std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;

  if (PropOptionInterface->get_offload_solver_initialized() && PropOptionInterface->get_offload_solver()->offload)
  {
    c_it_momentum_offload_info = PropOptionInterface->get_offload_solver()->offloading_momentum_map.find(momentum_point);

    if (c_it_momentum_offload_info->second.offload_to == ResourceUtils::MIC)
    {
      if (input_options.get_option("use_pcl_lapack", bool(true)))
        mat_mult = Greensolver::pcl_MIC;
      else
        mat_mult = Greensolver::blas_MIC;
    }
    else if(c_it_momentum_offload_info->second.offload_to == ResourceUtils::GPU)
    {
      if(input_options.get_option("use_cuda_interface", bool(false)))
        mat_mult = Greensolver::gpu_interface;
      else
        mat_mult = Greensolver::magma;
    }

    else
      mat_mult = Greensolver::petsc;
  }

  PetscMatrixParallelComplex* temp_matrix1 = NULL, *temp_matrix2=NULL;


  if(mat_mult == Greensolver::blas)
  {
    std::complex<double> one = std::complex<double>(1.0,0.0);
    std::complex<double> zero = std::complex<double>(0.0,0.0);
    std::string no_transpose = std::string("N");
    std::string conj_transpose = std::string("C");
    coupling->multBLAS(exact_neighbor_matrix, temp_matrix1, no_transpose, no_transpose, one, zero); //t_LL-1*G_L-1L-1
    temp_matrix1->multBLAS(coupling, temp_matrix2, no_transpose, conj_transpose, one, zero);  //t_LL-1*G_L-1L-1*t_L-1L

    delete temp_matrix1;
    temp_matrix1=NULL;
    //delete coupling;
    //coupling=NULL;
    temp_matrix2->multBLAS(half_way_g_matrix, temp_matrix1, no_transpose, no_transpose, one, zero);  //t_LL-1*G_L-1L-1*t_L-1L*g_LL

    delete temp_matrix2;
    temp_matrix2=NULL;
    half_way_g_matrix->multBLAS(temp_matrix1, temp_matrix2, no_transpose, no_transpose, one, zero);  // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
  }
  else if(mat_mult == Greensolver::blas_MIC)
  {
    std::complex<double> one = std::complex<double>(1.0,0.0);
    std::complex<double> zero = std::complex<double>(0.0,0.0);
    std::string no_transpose = std::string("N");
    std::string conj_transpose = std::string("C");

    PetscMatrixParallelComplex::multBLAS_MIC(coupling, exact_neighbor_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        c_it_momentum_offload_info->second); //t_LL-1*G_L-1L-1

    PetscMatrixParallelComplex::multBLAS_MIC(temp_matrix1, coupling, temp_matrix2, no_transpose, conj_transpose, one, zero,
        c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L

    delete temp_matrix1;
    temp_matrix1=NULL;
    //delete coupling;
    //coupling=NULL;

    PetscMatrixParallelComplex::multBLAS_MIC(temp_matrix2, half_way_g_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L*g_LL

    delete temp_matrix2;
    temp_matrix2=NULL;
    PetscMatrixParallelComplex::multBLAS_MIC(half_way_g_matrix, temp_matrix1, temp_matrix2, no_transpose, no_transpose, one, zero,
        c_it_momentum_offload_info->second);  // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
  }
  else if(mat_mult == Greensolver::pcl_MIC)
  {
    NEMO_ASSERT(PropOptionInterface->get_offload_solver()!= NULL, this_simulation->get_name() + ": Offload solver has not been initialized!\n");

    std::complex<double> one = std::complex<double>(1.0,0.0);
    std::complex<double> zero = std::complex<double>(0.0,0.0);
    std::string no_transpose = std::string("N");
    std::string conj_transpose = std::string("C");

    NemoMICUtils::pcl_lapack_zgemm_interface(coupling, exact_neighbor_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        coupling->get_communicator(), *(PropOptionInterface->get_offload_solver()), 
        c_it_momentum_offload_info->second); //t_LL-1*G_L-1L-1

    NemoMICUtils::pcl_lapack_zgemm_interface(temp_matrix1, coupling, temp_matrix2, no_transpose, conj_transpose, one, zero,
        temp_matrix1->get_communicator(), *(PropOptionInterface->get_offload_solver()), 
        c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L

    delete temp_matrix1;
    temp_matrix1=NULL;
    //delete coupling;
    //coupling=NULL;

    //NemoUtils::tic(tic_toc_prefix + "temp2 x gR");
    NemoMICUtils::pcl_lapack_zgemm_interface(temp_matrix2, half_way_g_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        temp_matrix2->get_communicator(), *(PropOptionInterface->get_offload_solver()), c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L*g_LL
    //NemoUtils::toc(tic_toc_prefix + "temp2 x gR");

    delete temp_matrix2;
    temp_matrix2=NULL;

    //NemoUtils::tic(tic_toc_prefix + "gR x temp1");
    NemoMICUtils::pcl_lapack_zgemm_interface(half_way_g_matrix, temp_matrix1, temp_matrix2, no_transpose, no_transpose, one, zero,
        half_way_g_matrix->get_communicator(), *(PropOptionInterface->get_offload_solver()), c_it_momentum_offload_info->second);  // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
    //NemoUtils::toc(tic_toc_prefix + "gR x temp1");
  }
  else if(mat_mult == Greensolver::magma)
  {
#ifdef MAGMA_ENABLE
    magmaDoubleComplex* coupling_gpu = NULL, *temp1_gpu = NULL, *temp2_gpu = NULL, *half_way_g_matrix_gpu = NULL;
    //NemoUtils::tic(tic_toc_prefix + "couplingH x GR");

    PetscMatrixParallelComplex::multMAGMA(*coupling,MagmaNoTrans,&coupling_gpu,
                                          *exact_neighbor_matrix,MagmaNoTrans,NULL,
                                          &temp_matrix1,false,&temp1_gpu); //t_LL-1*G_L-1L-1

    //NemoUtils::toc(tic_toc_prefix + "couplingH x GR");
    //NemoUtils::tic(tic_toc_prefix + "temp1 x couplingH'");

    PetscMatrixParallelComplex::multMAGMA(*temp_matrix1,MagmaNoTrans,&temp1_gpu,
                                          *coupling,MagmaConjTrans,&coupling_gpu,
                                          &temp_matrix2,false,&temp2_gpu); //t_LL-1*G_L-1L-1*t_L-1L

    //NemoUtils::toc(tic_toc_prefix + "temp1 x couplingH'");
    NemoUtils::tic(tic_toc_prefix + "MAGMA memfree 1");
    magma_free(coupling_gpu);
    coupling_gpu = NULL;
    magma_free(temp1_gpu);
    temp1_gpu = NULL;
    //delete coupling;
    //coupling = NULL;
    delete temp_matrix1;
    temp_matrix1=NULL;
    //NemoUtils::tic(tic_toc_prefix + "temp2 x gR");
    NemoUtils::toc(tic_toc_prefix + "MAGMA memfree 1");

    PetscMatrixParallelComplex::multMAGMA(*temp_matrix2,MagmaNoTrans,&temp2_gpu,
                                          *half_way_g_matrix,MagmaNoTrans,&half_way_g_matrix_gpu,
                                          &temp_matrix1,false,&temp1_gpu); //t_LL-1*G_L-1L-1*t_L-1L*g_LL

    NemoUtils::tic(tic_toc_prefix + "MAGMA memfree 2");
    magma_free(temp2_gpu);
    temp2_gpu = NULL;
    //NemoUtils::toc(tic_toc_prefix + "temp2 x gR");
    delete temp_matrix2;
    temp_matrix2=NULL;
    //NemoUtils::tic(tic_toc_prefix + "gR x temp1");
    NemoUtils::toc(tic_toc_prefix + "MAGMA memfree 2");

    PetscMatrixParallelComplex::multMAGMA(*half_way_g_matrix,MagmaNoTrans,&half_way_g_matrix_gpu,
                                          *temp_matrix1,MagmaNoTrans,&temp1_gpu,
                                          &temp_matrix2,true,NULL); // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL

    NemoUtils::tic(tic_toc_prefix + "MAGMA memfree 3");
    magma_free(half_way_g_matrix_gpu);
    half_way_g_matrix_gpu = NULL;
    magma_free(temp1_gpu);
    temp1_gpu = NULL;
    //NemoUtils::toc(tic_toc_prefix + "gR x temp1");
    NemoUtils::toc(tic_toc_prefix + "MAGMA memfree 3");

#else
    NEMO_EXCEPTION("solve_retarded_Green_back_RGF: Cannot use MAGMA without compiling with MAGMA!");
#endif
  }
  else if(mat_mult == Greensolver::gpu_interface)
  {
#ifdef MAGMA_ENABLE

    NemoUtils::tic(tic_toc_prefix + " CUDA mult, single allocation");

    int coupling_rows = coupling->get_num_rows();
    int coupling_cols = coupling->get_num_cols();

    if(!NemoGPUUtils::mult_gpu_initialized)
    {
      gpuErrchk( cudaStreamCreate(&NemoGPUUtils::compute_stream) );
      cublasErrchk( cublasCreate(&NemoGPUUtils::cublas_handle) );
      cublasErrchk( cublasSetStream(NemoGPUUtils::cublas_handle, NemoGPUUtils::compute_stream) );

      //Coupling workspace on GPU: coupling_rows x coupling_cols
      gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat1_gpu, coupling_rows*coupling_cols*sizeof(cuDoubleComplex)) );
      //exact neighbor workspace on GPU: coupling_cols x coupling_cols
      gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat2_gpu, coupling_cols*coupling_cols*sizeof(cuDoubleComplex)) );
      // halfway g workspace on GPU: coupling_rows x coupling_rows
      gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat3_gpu, coupling_rows*coupling_rows*sizeof(cuDoubleComplex)) );
      // t*G workspace on GPU: coupling_rows x coupling_cols
      gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat4_gpu, coupling_rows*coupling_cols*sizeof(cuDoubleComplex)) );
      // t*G*t^dagger & h*t*G*t^dagger*h (result): coupling_rows x coupling_rows
      gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat5_gpu, coupling_rows*coupling_rows*sizeof(cuDoubleComplex)) );
      // h*t*G*t^dagger workspace on GPU: coupling_rows x coupling_rows
      gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat6_gpu, coupling_rows*coupling_rows*sizeof(cuDoubleComplex)) );

      NemoGPUUtils::coupling_m = coupling_rows;
      NemoGPUUtils::coupling_n = coupling_cols;

      NemoGPUUtils::mult_gpu_initialized = true;
    }
    else
    {
      if(coupling_rows*coupling_cols > NemoGPUUtils::coupling_m*NemoGPUUtils::coupling_n)
      {
        //Coupling workspace on GPU: coupling_rows x coupling_cols
        gpuErrchk( cudaFree((void*)NemoGPUUtils::mult_mat1_gpu) );
        gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat1_gpu, coupling_rows*coupling_cols*sizeof(cuDoubleComplex)) );
        //exact neighbor workspace on GPU: coupling_cols x coupling_cols
        // t*G workspace on GPU: coupling_rows x coupling_cols
        gpuErrchk( cudaFree((void*)NemoGPUUtils::mult_mat4_gpu) );
        gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat4_gpu, coupling_rows*coupling_cols*sizeof(cuDoubleComplex)) );
      }
      if(coupling_rows > NemoGPUUtils::coupling_m)
      {
        // halfway g workspace on GPU: coupling_rows x coupling_rows
        gpuErrchk( cudaFree(NemoGPUUtils::mult_mat3_gpu) );
        gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat3_gpu, coupling_rows*coupling_rows*sizeof(cuDoubleComplex)) );
        // t*G*t^dagger: coupling_rows x coupling_rows
        gpuErrchk( cudaFree(NemoGPUUtils::mult_mat5_gpu) );
        gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat5_gpu, coupling_rows*coupling_rows*sizeof(cuDoubleComplex)) );
        // h*t*G*t^dagger workspace on GPU: coupling_rows x coupling_rows
        gpuErrchk( cudaFree(NemoGPUUtils::mult_mat6_gpu) );
        gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat6_gpu, coupling_rows*coupling_rows*sizeof(cuDoubleComplex)) );

        NemoGPUUtils::coupling_m = coupling_rows;
      }
      if(coupling_cols > NemoGPUUtils::coupling_n)
      {
        //exact neighbor workspace on GPU: coupling_cols x coupling_cols
        gpuErrchk( cudaFree((void*)NemoGPUUtils::mult_mat2_gpu) );
        gpuErrchk( cudaMalloc((void **)&NemoGPUUtils::mult_mat2_gpu, coupling_cols*coupling_cols*sizeof(cuDoubleComplex)) );

        NemoGPUUtils::coupling_n = coupling_cols;
      }
    }

    std::complex<double> *half_way_g_cpu_array = NULL;
    std::complex<double> *coupling_cpu_array = NULL;
    std::complex<double> *exact_neighbor_cpu_array = NULL;
    std::complex<double> *result_cpu_array = NULL;

    cuDoubleComplex one = {1.0, 0.0};
    cuDoubleComplex zero = {0.0, 0.0};

    if (!(coupling->is_ready()))
      coupling->assemble();
    if(coupling->if_sparse())
      coupling->matrix_convert_dense();
    coupling->get_array_for_matrix(coupling_cpu_array);
    cublasErrchk( cublasSetMatrix(coupling_rows, coupling_cols, sizeof(cuDoubleComplex),
        (const void*)coupling_cpu_array, coupling_cols, (void*)NemoGPUUtils::mult_mat1_gpu, coupling_cols) );

    if (!(exact_neighbor_matrix->is_ready()))
      exact_neighbor_matrix->assemble();
    if(exact_neighbor_matrix->if_sparse())
      exact_neighbor_matrix->matrix_convert_dense();
    exact_neighbor_matrix->get_array_for_matrix(exact_neighbor_cpu_array);
    cublasErrchk( cublasSetMatrix(coupling_cols, coupling_cols, sizeof(cuDoubleComplex),
            (const void*)exact_neighbor_cpu_array, coupling_cols, (void*)NemoGPUUtils::mult_mat2_gpu, coupling_cols) );

    //t_LL-1*G_L-1L-1
    cublasErrchk( cublasZgemm(NemoGPUUtils::cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, coupling_rows, coupling_cols, coupling_cols,
        &one, NemoGPUUtils::mult_mat1_gpu, coupling_cols, NemoGPUUtils::mult_mat2_gpu, coupling_cols, &zero,
        NemoGPUUtils::mult_mat4_gpu, coupling_cols) );

    //t_LL-1*G_L-1L-1*t_L-1L
    cublasErrchk( cublasZgemm(NemoGPUUtils::cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C, coupling_rows, coupling_cols, coupling_rows,
        &one, NemoGPUUtils::mult_mat4_gpu, coupling_cols, NemoGPUUtils::mult_mat1_gpu, coupling_cols, &zero,
        NemoGPUUtils::mult_mat5_gpu, coupling_rows) );

    if (!(half_way_g_matrix->is_ready()))
      half_way_g_matrix->assemble();
    if(half_way_g_matrix->if_sparse())
      half_way_g_matrix->matrix_convert_dense();
    half_way_g_matrix->get_array_for_matrix(half_way_g_cpu_array);
    cublasErrchk( cublasSetMatrix(coupling_rows, coupling_rows, sizeof(cuDoubleComplex),
            (const void*)half_way_g_cpu_array, coupling_rows, (void*)NemoGPUUtils::mult_mat3_gpu, coupling_rows) );

    gpuErrchk( cudaStreamSynchronize(NemoGPUUtils::compute_stream) );

    //t_LL-1*G_L-1L-1*t_L-1L*g_LL
    cublasErrchk( cublasZgemm(NemoGPUUtils::cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, coupling_rows, coupling_rows, coupling_rows,
        &one, NemoGPUUtils::mult_mat5_gpu, coupling_rows, NemoGPUUtils::mult_mat3_gpu, coupling_rows, &zero,
        NemoGPUUtils::mult_mat6_gpu, coupling_rows) );

    // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
    cublasErrchk( cublasZgemm(NemoGPUUtils::cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, coupling_rows, coupling_rows, coupling_rows,
        &one, NemoGPUUtils::mult_mat3_gpu, coupling_rows, NemoGPUUtils::mult_mat6_gpu, coupling_rows, &zero,
        NemoGPUUtils::mult_mat5_gpu, coupling_rows) );

    temp_matrix2 = new PetscMatrixParallelComplex(coupling_rows, coupling_rows, coupling->get_communicator());
    temp_matrix2->consider_as_full();
    temp_matrix2->allocate_memory();
//    temp_matrix2->set_to_zero();
    temp_matrix2->assemble();
    temp_matrix2->get_array_for_matrix(result_cpu_array);

    gpuErrchk( cudaStreamSynchronize(NemoGPUUtils::compute_stream) );

    cublasErrchk( cublasGetMatrix(coupling_rows, coupling_rows, sizeof(cuDoubleComplex),
        (const void*)NemoGPUUtils::mult_mat5_gpu, coupling_rows, (void*)result_cpu_array, coupling_rows) );

    temp_matrix2->restore_array_for_matrix(result_cpu_array);
    coupling->restore_array_for_matrix(coupling_cpu_array);
    half_way_g_matrix->restore_array_for_matrix(half_way_g_cpu_array);
    exact_neighbor_matrix->restore_array_for_matrix(exact_neighbor_cpu_array);

    NemoUtils::toc(tic_toc_prefix + " CUDA mult, single allocation");
#else
    NEMO_EXCEPTION("solve_retarded_Green_back_RGF: Cannot use MAGMA without compiling with MAGMA!");
#endif
  }
  else
  {
//#define USE_NMC
#ifndef USE_NMC
	PropagationOptionsInterface* temp_interface = get_PropagationOptionsInterface(this_simulation);
	bool supress_noise_enabled = temp_interface->get_use_matrix_0_threshold();
    double noise_threshold = 0;
    if (supress_noise_enabled)
    {
      InputOptions input_options=this_simulation->get_options();
	  noise_threshold = input_options.get_option("matrix_0_threshold",NemoMath::d_zero_tolerance);
    }

    PropagationUtilities::supress_noise(coupling, noise_threshold);
    PropagationUtilities::supress_noise(exact_neighbor_matrix, noise_threshold);
    PetscMatrixParallelComplex::mult(*coupling,*exact_neighbor_matrix,&temp_matrix1); //t_LL-1*G_L-1L-1
    PropagationUtilities::supress_noise(temp_matrix1,noise_threshold);

    PetscMatrixParallelComplex transpose_coupling(coupling->get_num_cols(),coupling->get_num_rows(),
        coupling->get_communicator());
    coupling->hermitian_transpose_matrix(transpose_coupling,MAT_INITIAL_MATRIX);
    //delete coupling;
    //coupling=NULL;

    PropagationUtilities::supress_noise(&transpose_coupling,noise_threshold);
    PetscMatrixParallelComplex::mult(*temp_matrix1,transpose_coupling,&temp_matrix2); //t_LL-1*G_L-1L-1*t_L-1L
    PropagationUtilities::supress_noise(temp_matrix2,noise_threshold);
    delete temp_matrix1;
    temp_matrix1=NULL;

    PropagationUtilities::supress_noise(half_way_g_matrix,noise_threshold);
    PetscMatrixParallelComplex::mult(*temp_matrix2,*half_way_g_matrix,&temp_matrix1); //t_LL-1*G_L-1L-1*t_L-1L*g_LL
    PropagationUtilities::supress_noise(temp_matrix1,noise_threshold);
    delete temp_matrix2;
    temp_matrix2=NULL;

    PetscMatrixParallelComplex::mult(*half_way_g_matrix,*temp_matrix1,&temp_matrix2); // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
    PropagationUtilities::supress_noise(temp_matrix2,noise_threshold);
#else

    #define OUTPUTDBG
    #define TIME_TICTOC
    #ifdef TIME_TICTOC
	NemoUtils::tic("convert_from_PETSC_to_NMC");
#endif
	NemoMatrixInterface* NMC_coupling = NULL, *NMC_exact_neighbor_matrix = NULL,
			*NMC_half_way_g_matrix = NULL, *NMC_result = NULL;


	NMC_exact_neighbor_matrix = NemoFactory::matrix_instance(
			DEFAULT_MAT_TYPE, exact_neighbor_matrix->get_num_cols(),
				exact_neighbor_matrix->get_num_rows(), exact_neighbor_matrix->get_communicator());
	NMC_exact_neighbor_matrix->consider_as_full();
	NMC_exact_neighbor_matrix->allocate_memory();
	exact_neighbor_matrix->convert_to(NemoMatNMC, NMC_exact_neighbor_matrix);
	NMC_exact_neighbor_matrix->assemble();
#ifdef OUTPUTDBG
	if (NMC_exact_neighbor_matrix == NULL)
	{
		cout << "NMC_exact_neighbor_matrix == NULL\n" << endl;
		cout << exact_neighbor_matrix->get_num_rows() << " "
				<< exact_neighbor_matrix->get_num_cols() << " "
				<< exact_neighbor_matrix->check_if_sparse() << " "
				<< endl;
  }
	NemoMatrixInterface::check_mat_eq(exact_neighbor_matrix, NMC_exact_neighbor_matrix);
#endif
	NMC_half_way_g_matrix = NemoFactory::matrix_instance(DEFAULT_MAT_TYPE,
			half_way_g_matrix->get_num_cols(), half_way_g_matrix->get_num_rows(),
			half_way_g_matrix->get_communicator());
	NMC_half_way_g_matrix->consider_as_full();
	NMC_half_way_g_matrix->allocate_memory();
	half_way_g_matrix->convert_to(NemoMatNMC, NMC_half_way_g_matrix);
	NMC_half_way_g_matrix->assemble();
#ifdef OUTPUTDBG
	if (NMC_exact_neighbor_matrix == NULL)
		cout << "NMC_half_way_g_matrix == NULL\n" << endl;
	NemoMatrixInterface::check_mat_eq(half_way_g_matrix, NMC_half_way_g_matrix);
#endif

	NMC_coupling = NemoFactory::matrix_instance(DEFAULT_MAT_TYPE,
			coupling->get_num_cols(), coupling->get_num_rows(),
			coupling->get_communicator());
	NMC_coupling->consider_as_full();
	NMC_coupling->allocate_memory();
	coupling->convert_to(NemoMatNMC, NMC_coupling);
	NMC_coupling->assemble();
#ifdef OUTPUTDBG
	if (NMC_coupling == NULL)
		cout << "NMC_coupling == NULL \n" << endl;
	NemoMatrixInterface::check_mat_eq(coupling, NMC_coupling);
#endif
#ifdef TIME_TICTOC
	NemoUtils::toc("convert_from_PETSC_to_NMC");
#endif


	NemoMatrixInterface* NMC_temp_matrix1 = NULL, *NMC_temp_matrix2 = NULL;



	PropagationUtilities::supress_noise(this_simulation, NMC_coupling);
	PropagationUtilities::supress_noise(this_simulation,
			NMC_exact_neighbor_matrix);
#ifdef OUTPUTDBG
	if (NMC_coupling == NULL)
		cout << "NMC_coupling == NULL after supress\n" << endl;
	if (NMC_exact_neighbor_matrix == NULL)
		cout << "NMC_exact_neighbor_matrix == NULL after supress\n" << endl;
#endif
	NemoMatrixInterface::mult(*NMC_coupling, *NMC_exact_neighbor_matrix,
			&NMC_temp_matrix1); //t_LL-1*G_L-1L-1
	PropagationUtilities::supress_noise(this_simulation, NMC_temp_matrix1);
	NemoMatrixInterface* NMC_transpose_coupling = NemoFactory::matrix_instance(
			DEFAULT_MAT_TYPE, NMC_coupling->get_num_cols(),
			NMC_coupling->get_num_rows(), NMC_coupling->get_communicator());
	NMC_coupling->hermitian_transpose_matrix(*NMC_transpose_coupling,
			NEMO_MAT_INITIAL_MATRIX);
	PropagationUtilities::supress_noise(this_simulation, NMC_transpose_coupling);
	NemoMatrixInterface::mult(*NMC_temp_matrix1, *NMC_transpose_coupling,
			&NMC_temp_matrix2); //t_LL-1*G_L-1L-1*t_L-1L
	PropagationUtilities::supress_noise(this_simulation, NMC_temp_matrix2);
	delete NMC_temp_matrix1;
	NMC_temp_matrix1 = NULL;
	PropagationUtilities::supress_noise(this_simulation, NMC_half_way_g_matrix);
	NemoMatrixInterface::mult(*NMC_temp_matrix2, *NMC_half_way_g_matrix,
			&NMC_temp_matrix1); //t_LL-1*G_L-1L-1*t_L-1L*g_LL
	PropagationUtilities::supress_noise(this_simulation, NMC_temp_matrix1);
	delete NMC_temp_matrix2;
	NMC_temp_matrix2 = NULL;
	NemoMatrixInterface::mult(*NMC_half_way_g_matrix, *NMC_temp_matrix1,
			&NMC_temp_matrix2); // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
	PropagationUtilities::supress_noise(this_simulation, NMC_temp_matrix2);

	delete NMC_temp_matrix1;
	NMC_temp_matrix1 = NULL;

	NMC_result = NemoFactory::matrix_instance(DEFAULT_MAT_TYPE,
			NMC_half_way_g_matrix->get_num_rows(),
			NMC_half_way_g_matrix->get_num_cols(),
			NMC_half_way_g_matrix->get_communicator());
	NMC_result->consider_as_full();
	NMC_result->allocate_memory();
	std::complex<double> *dest = NULL, *src = NULL;
	NMC_result->get_array(dest);
	NMC_half_way_g_matrix->get_array(src);
	memcpy(dest, src,
			sizeof(std::complex<double>) * NMC_result->get_num_cols()
					* NMC_result->get_num_rows());
	NMC_result->assemble();
	NMC_result->add_matrix(*NMC_temp_matrix2, NEMO_SAME_NONZERO_PATTERN);
	delete NMC_temp_matrix2;
	NMC_temp_matrix2 = NULL;
	PropagationUtilities::supress_noise(this_simulation, NMC_result);


#ifdef TIME_TICTOC
	NemoUtils::tic("convert_from_NMC_to_PETSC");
#endif
	result = new PetscMatrixParallelComplex(*half_way_g_matrix);
	result->matrix_convert_dense();
	NMC_result->convert_to(NemoMatPETSC, result);

	delete NMC_coupling;
	NMC_coupling = NULL;
	delete NMC_exact_neighbor_matrix;
	NMC_exact_neighbor_matrix = NULL;
	delete NMC_half_way_g_matrix;
	NMC_half_way_g_matrix = NULL;
	delete NMC_result;
	NMC_result = NULL;
	delete NMC_transpose_coupling;
	NMC_transpose_coupling = NULL;
#ifdef TIME_TICTOC
	NemoUtils::toc("convert_from_NMC_to_PETSC");
#endif
#endif
  }

#ifndef USE_NMC
  delete temp_matrix1;
  temp_matrix1=NULL;

  result = new PetscMatrixParallelComplex(*half_way_g_matrix);
  result->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN); //g_LL + g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
  delete temp_matrix2;
  temp_matrix2=NULL;
  PropagationUtilities::supress_noise(this_simulation,result);
#endif
  result->consider_as_full();
  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::solve_retarded_Green_back_RGF(Simulation* this_simulation,Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_retarded_Green_back_RGF ";
  NemoUtils::tic(tic_toc_prefix);
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  const InputOptions& input_options=this_simulation->get_options();
  //1. get the g-function of this domain, i.e. the Greensfunction that is only connected to one neighbor domain (cf. J. Appl. Phys. 81, 7845; Eq.59)
  std::string half_way_name;
  std::string variable_name="half_way_retarded_green";
  if (input_options.check_option(variable_name))
    half_way_name=input_options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix+"define \""+variable_name+"\"\n");
  GreensfunctionInterface* source_of_half_way = find_source_of_Greensfunction(this_simulation,half_way_name);
  PetscMatrixParallelComplex* half_way_g_matrix=NULL;
  const DOFmapInterface& half_way_DOFmap = PropOptionInterface->get_Hamilton_Constructor()->get_const_dof_map(this_simulation->get_const_simulation_domain());
  
  if(PropOptionInterface->get_particle_type_is_Fermion())
    source_of_half_way->get_Greensfunction(momentum_point,half_way_g_matrix,&half_way_DOFmap,NULL,NemoPhys::Fermion_retarded_Green);
  else
    source_of_half_way->get_Greensfunction(momentum_point,half_way_g_matrix,&half_way_DOFmap,NULL,NemoPhys::Boson_retarded_Green);

  //what is the neighboring domain:
  variable_name=output_Propagator->get_name()+std::string("_lead_domain");
  std::string neighbor_domain_name;
  if (input_options.check_option(variable_name))
    neighbor_domain_name=input_options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix+"define \""+variable_name+"\"\n");
  const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);

  //2. get the exact Greensfunction of the neighboring domain (cf. J. Appl. Phys. 81, 7845; Eq.60)
  std::string exact_neighbor_name;
  variable_name="exact_lead_retarded_green";
  if (input_options.check_option(variable_name))
    exact_neighbor_name=input_options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix+"define \""+variable_name+"\"\n");
  //Simulation* source_of_exact_neighbor = find_source_of_data(exact_neighbor_name);
  GreensfunctionInterface* source_of_exact_neighbor =find_source_of_Greensfunction(this_simulation,exact_neighbor_name);

  //if lead_Hamilton_constructor is defined, get the lead DOFmap from it, otherwise use source_of_exact_neighbor
  DOFmapInterface* exact_lead_DOFmap=NULL;
  if(input_options.check_option("exact_lead_Hamilton_constructor"))
  {
    std::string temp_name = input_options.get_option("exact_lead_Hamilton_constructor",std::string(""));
    Simulation* lead_Hamilton_constructor = this_simulation->find_simulation(temp_name);
    exact_lead_DOFmap=&(lead_Hamilton_constructor->get_dof_map(neighbor_domain));
  }
  else
    throw std::invalid_argument(tic_toc_prefix+"please define \"exact_lead_Hamilton_constructor\"\n");

  //3. get the coupling Hamiltonian that couples this domain to the neighboring one
  PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;
  DOFmapInterface* coupling_DOFmap=exact_lead_DOFmap;

  std::vector<NemoMeshPoint> sorted_momentum;
  QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum,input_options,PropInterface->get_momentum_mesh_types(),PropOptionInterface->get_Hamilton_Constructor());
  PropOptionInterface->get_Hamilton_Constructor()->get_data(std::string("Hamiltonian"), sorted_momentum, neighbor_domain, coupling_Hamiltonian,coupling_DOFmap,
                                 this_simulation->get_const_simulation_domain());
  if(exact_lead_DOFmap!=coupling_DOFmap)
    delete coupling_DOFmap;


  PetscMatrixParallelComplex* exact_neighbor_matrix=NULL;
  if(PropOptionInterface->get_particle_type_is_Fermion())
    source_of_exact_neighbor->get_Greensfunction(momentum_point,exact_neighbor_matrix,exact_lead_DOFmap, NULL, NemoPhys::Fermion_retarded_Green);
  else
    source_of_exact_neighbor->get_Greensfunction(momentum_point,exact_neighbor_matrix,exact_lead_DOFmap, NULL, NemoPhys::Boson_retarded_Green);

  PetscMatrixParallelComplex* coupling = NULL;
  extract_coupling_Hamiltonian_RGF(this_simulation, half_way_g_matrix->get_num_rows(),half_way_g_matrix->get_num_cols(), coupling_Hamiltonian, coupling);
  core_retarded_Green_back_RGF(this_simulation,momentum_point,exact_neighbor_matrix,half_way_g_matrix,coupling,result);
  delete coupling;
  coupling = NULL;

  //update the list of allocated propagation matrices
  output_Propagator->allocated_momentum_Propagator_map[momentum_point]=true;

  set_job_done_momentum_map(this_simulation,&(output_Propagator->get_name()), &momentum_point, true);
  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::do_solve_retarded(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::do_solve_retarded ";
  NemoUtils::tic(tic_toc_prefix);

  //specify how to solve(invert)
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  if(PropOptionInterface->get_inversion_method() == NemoMath::exact_inversion)
  {
    GreensfunctionInterface* green_source=NULL;
    if(PropOptionInterface->get_read_NEGF_object_list())
    {
      //find the name of the inverse retarded Greens function
      std::string inverse_name; //name of the inverse Green's function
      std::map<std::string,Simulation*>* Propagator_constructors=PropInterface->list_Propagator_constructors();
      std::map<std::string, Simulation*>::const_iterator c_it=Propagator_constructors->begin();
      for(; c_it!=Propagator_constructors->end(); ++c_it)
      {
        PropagatorInterface* temp_interface=get_PropagatorInterface(c_it->second);
        Propagator* temp_Propagator=NULL;
        temp_interface->get_Propagator(temp_Propagator);
        std::string name_of_writeable_Propagator=temp_Propagator->get_name();
        if(name_of_writeable_Propagator!=c_it->first)
          if(c_it->first.find("inverse")!=std::string::npos)
          {
            inverse_name=c_it->first;
          }
      }
      //Simulation* data_source=find_source_of_data(inverse_name);
      green_source=find_source_of_Greensfunction(this_simulation,inverse_name);
    }
    else
    {
      NEMO_ASSERT(PropOptionInterface->get_inverse_GR_solver()!=NULL,tic_toc_prefix+"inverse_GR_solver is NULL\n");
      green_source=dynamic_cast<GreensfunctionInterface*>(PropOptionInterface->get_inverse_GR_solver());
      NEMO_ASSERT(green_source!=NULL,tic_toc_prefix+"inverse_GR_solver is not a GreensfunctionInterface\n");
    }

    PetscMatrixParallelComplex* inverse_Green=NULL;
    green_source->get_Greensfunction(momentum_point,inverse_Green,
      &(PropOptionInterface->get_Hamilton_Constructor()->get_dof_map(this_simulation->get_const_simulation_domain())),
      NULL,NemoPhys::Inverse_Green);
    exact_inversion(this_simulation, momentum_point, inverse_Green, result);
  }
  else if(PropOptionInterface->get_inversion_method() == NemoMath::recursive_backward_inversion)
    solve_retarded_Green_back_RGF(this_simulation,output_Propagator, momentum_point, result);
  else if(PropOptionInterface->get_inversion_method() == NemoMath::recursive_off_diagonal_inversion)
    PropagationUtilities::solve_interdomain_Green_RGF(this_simulation,output_Propagator, momentum_point, result);
  else if(PropOptionInterface->get_inversion_method() == NemoMath::sancho_rubio_inversion)
    solve_retarded_Green_SanchoRubio(this_simulation,output_Propagator, momentum_point, result);
  else
    throw std::invalid_argument(tic_toc_prefix+"unknown inversion type\n");
  set_job_done_momentum_map(this_simulation,&(output_Propagator->get_name()), &momentum_point, true);


  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::do_solve_spectral(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                             PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities for \""+this_simulation->get_name()+"\" do_solve_spectral ";
  NemoUtils::tic(tic_toc_prefix);
  const InputOptions& input_options=this_simulation->get_options();
  std::string solve_option=input_options.get_option(output_Propagator->get_name()+"_method",std::string("GR_diagonal"));
  PropagationOptionsInterface* this_prop_interface=get_PropagationOptionsInterface(this_simulation);
  if(solve_option==std::string("GR_diagonal"))
  {
    //Fabio: the function is solving the full spectral function A_L,L = i*(G_L,L - G_L,L^+)
    //Yu: GR is dense
    //1. Fabio: get GR
    std::string variable_name="retarded_greens_function";
    if(!input_options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string retarded_green_name = input_options.get_option(variable_name,std::string(""));
    GreensfunctionInterface* retarded_green_source=find_source_of_Greensfunction(this_simulation,retarded_green_name);
    PetscMatrixParallelComplex* temp_GR=NULL;
    if(this_prop_interface->get_particle_type_is_Fermion())
      retarded_green_source->get_Greensfunction(momentum_point,temp_GR,NULL,NULL,NemoPhys::Fermion_retarded_Green);
    else
      retarded_green_source->get_Greensfunction(momentum_point,temp_GR,NULL,NULL,NemoPhys::Boson_retarded_Green);
    
    //2. Fabio: do the math operation
    core_spectral_imag_of_GR(this_simulation, temp_GR, result);
  }
  else if(solve_option==std::string("GR_Gamma_GA"))
  {
    //Fabio: the function is solving the full spectral function A_L,L = G_L,N * Gamma * G_L,N^+
    //Yu: self-energy is always sparse
    //1. Fabio: get self-energy 
    std::string variable_name="self_energy";
    if(!input_options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string self_name = input_options.get_option(variable_name,std::string(""));
    Simulation* self_source = find_source_of_data(this_simulation,self_name);
    PetscMatrixParallelComplex* self_matrix=NULL;
    self_source->get_data(self_name,&momentum_point,self_matrix);

    //2. Fabio: get G_L,N 
    //Yu: off-diagonal GR is dense
    variable_name="offdiagonal_greens_function";
    if(!input_options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string offdiag_green_name = input_options.get_option(variable_name,std::string(""));
    //Simulation* offdiag_green_source = find_source_of_data(offdiag_green_name);
    GreensfunctionInterface* offdiag_green_source=find_source_of_Greensfunction(this_simulation,offdiag_green_name);
    PetscMatrixParallelComplex* offdiag_matrix=NULL;
    if(this_prop_interface->get_particle_type_is_Fermion())
      offdiag_green_source->get_Greensfunction(momentum_point,offdiag_matrix,NULL,NULL,NemoPhys::Fermion_retarded_Green);
    else
      offdiag_green_source->get_Greensfunction(momentum_point,offdiag_matrix,NULL,NULL,NemoPhys::Boson_retarded_Green);
    //offdiag_green_source->get_data(offdiag_green_name,&momentum_point,offdiag_matrix);

    //3. Fabio: do the math operation
    core_spectral_GR_Gamma_GA(this_simulation, offdiag_matrix, self_matrix, result);

  }
  else
    throw std::invalid_argument("PropagationUtilities for \""+this_simulation->get_name()+"\" do_solve_spectral: unknown inversion type\n");
  //result->save_to_matlab_file("subAL.m");
  set_job_done_momentum_map(this_simulation,&(output_Propagator->get_name()), &momentum_point,true);
  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::core_spectral_imag_of_GR(Simulation* this_simulation, PetscMatrixParallelComplex*& GR, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities for \"" + this_simulation->get_name() + "\" core_spectral_imag_of_GR ";
  NemoUtils::tic(tic_toc_prefix);

  GR->get_diagonal_matrix(result);
  result->imaginary_part(); //cplx(Im(GR),0)
  *result *= std::complex<double>(-2.0, 0.0); // A = -2*Im(GR) = i (GR-GA)
  supress_noise(this_simulation, result);

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::core_spectral_GR_Gamma_GA(Simulation* this_simulation, PetscMatrixParallelComplex*& offdiag_matrix,
  PetscMatrixParallelComplex*& self_matrix,
  PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "PropagationUtilities for \"" + this_simulation->get_name() + "\" core_spectral_GR_Gamma_GA ";
  NemoUtils::tic(tic_toc_prefix);

  //1. calculate gamma from self energy
  PetscMatrixParallelComplex* temp_self = new PetscMatrixParallelComplex(*self_matrix);//do a copy of sigma
  temp_self->matrix_convert_dense(); //Yu: convert to dense for efficient mult
  PetscMatrixParallelComplex temp_self1(temp_self->get_num_cols(), temp_self->get_num_rows(), temp_self->get_communicator());
  temp_self->hermitian_transpose_matrix(temp_self1, MAT_INITIAL_MATRIX);
  supress_noise(this_simulation, &temp_self1);
  temp_self->add_matrix(temp_self1, SAME_NONZERO_PATTERN, std::complex<double>(-1.0, 0.0)); //sigma-sigma'
  (*temp_self) *= std::complex<double>(0.0, 1.0); //gamma=i*(sigma-sigma')
  supress_noise(this_simulation, temp_self);
  int number_of_self_rows = temp_self->get_num_rows(); //number of rows of self-energy

  //2. do the multiplication
  //2.1 GR x Gamma
  PetscMatrixParallelComplex* temp_matrix = NULL;
  //std::string tic_toc_mult = tic_toc_prefix+" GR x Gamma ";
  //NemoUtils::tic(tic_toc_mult);
  supress_noise(this_simulation, offdiag_matrix);
  PetscMatrixParallelComplex::mult(*offdiag_matrix, *temp_self, &temp_matrix);
  supress_noise(this_simulation, temp_matrix);
  //NemoUtils::toc(tic_toc_mult);
  delete temp_self;
  temp_self = NULL;

  //2.2 dot_product between vectors from GR x Gamma and GA to get diagonal elements of the spectral function
  int num_of_GR_rows = offdiag_matrix->get_num_rows();
  std::vector<std::complex<double> > diagonal_result(num_of_GR_rows, std::complex<double>(0.0, 0.0));
  PetscVectorNemo<std::complex<double> > sub_GR_vector(number_of_self_rows, number_of_self_rows, offdiag_matrix->get_communicator());
  PetscVectorNemo<std::complex<double> > temp_vector(number_of_self_rows, number_of_self_rows, offdiag_matrix->get_communicator());
  std::complex<double>* pointer_GR = NULL;
  std::complex<double>* pointer_temp = NULL;
  offdiag_matrix->get_array(pointer_GR);
  temp_matrix->get_array(pointer_temp);
  //iterate over all matrix rows (i is result-row index)
  for (int i = 0; i < num_of_GR_rows; i++)
  {
    cplx* pointer_to_GR_vector = NULL;
    sub_GR_vector.get_array(pointer_to_GR_vector);
    for (int j = 0; j < number_of_self_rows; j++)
      pointer_to_GR_vector[j] = pointer_GR[j*num_of_GR_rows + i];
    sub_GR_vector.store_array(pointer_to_GR_vector);

    cplx* pointer_to_temp_vector = NULL;
    temp_vector.get_array(pointer_to_temp_vector);
    for (int j = 0; j < number_of_self_rows; j++)
      pointer_to_temp_vector[j] = pointer_temp[j*num_of_GR_rows + i];
    temp_vector.store_array(pointer_to_temp_vector);

    //dot_product
    std::complex<double> temp_result(0.0, 0.0);
    PetscVectorNemo<std::complex<double> >::dot_product(temp_vector, sub_GR_vector, temp_result);

    //store the local result in the result matrix
    diagonal_result[i] = temp_result;
    //NemoUtils::toc(tic_toc_prefix4);
  }
  temp_matrix->store_array(pointer_temp);
  offdiag_matrix->store_array(pointer_GR);
  delete temp_matrix;
  temp_matrix = NULL;

  //3. store diagonal into result matrix
  PetscVectorNemo<std::complex<double> > petsc_vector_diagonal(num_of_GR_rows, num_of_GR_rows, offdiag_matrix->get_communicator());
  std::vector<int> indices(num_of_GR_rows, 0);
  for (unsigned int i = 0; i < indices.size(); i++) indices[i] = i; //dense vector...
  petsc_vector_diagonal.set_values(indices, diagonal_result);
  //3.1 create a new matrix
  result = new PetscMatrixParallelComplex(num_of_GR_rows, num_of_GR_rows, offdiag_matrix->get_communicator());
  //3.2 set the sparsity pattern
  result->set_num_owned_rows(offdiag_matrix->get_num_owned_rows());
  int start_own_rows;
  int end_own_rows;
  offdiag_matrix->get_ownership_range(start_own_rows, end_own_rows);
  for (int i = start_own_rows; i < end_own_rows; i++)
    result->set_num_nonzeros(i, 1, 0);
  result->allocate_memory();
  result->set_to_zero();
  result->matrix_diagonal_shift(petsc_vector_diagonal);
  result->assemble();
  supress_noise(this_simulation, result);

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::core_calculate_transmission(Simulation* this_simulation, PetscMatrixParallelComplex*& self_matrix1,
  PetscMatrixParallelComplex*& self_matrix2,
  PetscMatrixParallelComplex*& green_matrix, double& transmission)
{
  std::string tic_toc_prefix = "PropagationUtilities for \"" + this_simulation->get_name() + "\" core_calculate_transmission ";
  NemoUtils::tic(tic_toc_prefix);

/*
  if(self_matrix1->check_if_sparse())
  	  std::cout<<"has sparse"<<endl;
  if(self_matrix1->is_scalar)
  	  std::cout<<"has scalar"<<endl;
  if(self_matrix2->check_if_sparse())
  	  std::cout<<"has sparse"<<endl;
  if(self_matrix2->is_scalar)
  	  std::cout<<"has scalar"<<endl;
  if(green_matrix->check_if_sparse())
  	  std::cout<<"has sparse"<<endl;
  if(green_matrix->is_scalar)
  	  std::cout<<"has scalar"<<endl;
*/



  //transmission = tr(gamma_p*Gr*gama_q*Ga); Gamma=i*(sigma_R-sigma_A)
  PetscMatrixParallelComplex* gamma1 = new PetscMatrixParallelComplex(*self_matrix1);
  PetscMatrixParallelComplex* gamma2 = new PetscMatrixParallelComplex(*self_matrix2);
  PetscMatrixParallelComplex* temp = new PetscMatrixParallelComplex(self_matrix1->get_num_cols(),
    self_matrix1->get_num_rows(),
    self_matrix1->get_communicator());
  //gamma1
  self_matrix1->hermitian_transpose_matrix(*temp, MAT_INITIAL_MATRIX);
  *temp *= std::complex<double>(-1.0, 0.0);
  gamma1->add_matrix(*temp, DIFFERENT_NONZERO_PATTERN);
  *gamma1 *= std::complex<double>(0.0, 1.0);
  delete temp;
  //gamma2
  temp = new PetscMatrixParallelComplex(self_matrix2->get_num_cols(),
    self_matrix2->get_num_rows(),
    self_matrix2->get_communicator());
  self_matrix2->hermitian_transpose_matrix(*temp, MAT_INITIAL_MATRIX);
  *temp *= std::complex<double>(-1.0, 0.0);
  gamma2->add_matrix(*temp, DIFFERENT_NONZERO_PATTERN);
  *gamma2 *= std::complex<double>(0.0, 1.0);
  delete temp;

  //temp=gamma_p*Gr*gama_q*Ga
  temp = new PetscMatrixParallelComplex(green_matrix->get_num_cols(),
    green_matrix->get_num_rows(),
    green_matrix->get_communicator());

  green_matrix->hermitian_transpose_matrix(*temp, MAT_INITIAL_MATRIX);
  PetscMatrixParallelComplex* temp2 = NULL;
  PetscMatrixParallelComplex::mult(*gamma2, *temp, &temp2);//temp2 = gamma2*GA
  delete temp;
  temp = NULL;
  PetscMatrixParallelComplex::mult(*green_matrix, *temp2, &temp); //temp = GR*gamma2*GA
  delete temp2;
  temp2 = NULL;
  PetscMatrixParallelComplex::mult(*gamma1, *temp, &temp2); //temp2 = gamm1*GR*gamma2*GA
  delete gamma1;
  delete gamma2;
  delete temp;

  // get the trace of temp2
  std::complex<double> trace(std::complex<double>(0.0, 0.0));
  trace = temp2->get_trace();
  delete temp2;

  transmission = trace.real();

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::do_solve_lesser(Simulation* this_simulation, string solve_option, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result,
    std::set<const Propagator*>& lesser_self,const Propagator* retarded_G, const Propagator* advanced_G)
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::do_solve_lesser2 ";
  NemoUtils::tic(tic_toc_prefix);

  std::string error_prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::do_solve_lesser() ";

  //PetscMatrixParallelComplex* temp_matrix=NULL;
  //get the total lesser self-energy
  //std::string solve_option=this_simulation->options.get_option(output_Propagator->get_name()+"_solution",std::string("exact"));
  Propagator::PropagatorMap::const_iterator momentum_it;
  //PetscMatrixParallelComplex* summed_lesser_self=NULL;

  if (solve_option == std::string("RGF_complex"))
    solve_lesser_Green_RGF_complex(this_simulation, output_Propagator, momentum_point, result, retarded_G, advanced_G);
  else if(solve_option==std::string("RGF"))
    solve_lesser_Green_RGF(this_simulation, output_Propagator,momentum_point,result,retarded_G,advanced_G);
  else if(solve_option==std::string("Block_RGF"))
    solve_correlation_Green_RGF2(this_simulation, output_Propagator, momentum_point, lesser_self, result);
  else if(solve_option==std::string("equilibrium_model"))
    do_solve_lesser_equilibrium(this_simulation, output_Propagator,momentum_point,result);
  else if(solve_option==std::string("nonlocal_RGF"))
    solve_correlation_Green_nonlocalRGF(this_simulation, output_Propagator,momentum_point, lesser_self, result, retarded_G, advanced_G);
  else
  {
    solve_correlation_Green_exact(this_simulation, output_Propagator,momentum_point,result, lesser_self, retarded_G, advanced_G);
  }

  
  PropagationOptionsInterface* temp_interface=dynamic_cast<PropagationOptionsInterface*>(this_simulation);
  if(temp_interface->get_symmetrize_results())
  {
    NemoMath::symmetry_type type = NemoMath::antihermitian;
    symmetrize(this_simulation,result,type);
  }


  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::solve_lesser_Green_RGF_complex(Simulation* this_simulation, Propagator*& , const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result,
  const Propagator* retarded_G, const Propagator* advanced_G)
{
  std::string tic_toc_prefix = "PropagationUtilities(\"" + this_simulation->get_name() + "\")::solve_lesser_Green_RGF_complex ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\"" + this_simulation->get_name() + "\")::solve_lesser_Green_RGF_complex ";

  PropagationOptionsInterface* PropOptionInterface = get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface = get_PropagatorInterface(this_simulation);
  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();
  std::map<std::string, Simulation*>* Propagators = PropInterface->list_Propagator_constructors();
  InputOptions options = this_simulation->get_reference_to_options();

  //1. check that there is exactly one retarded self-energy in the readable Propagator list and store its name
  std::map<std::string, Simulation*>::const_iterator prop_cit = Propagators->begin();
  Propagator* writeable_Propagator = NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  std::string self_name = "";
  std::string name_of_writeable_Propagator = writeable_Propagator->get_name();
  for (; prop_cit != Propagators->end(); prop_cit++)
  {
    //write_prop_cit=writeable_Propagators.find(prop_cit->first);
    //if(write_prop_cit==writeable_Propagators.end())
    if (prop_cit->first != name_of_writeable_Propagator)
    {
      NemoPhys::Propagator_type temp_type = PropInterface->get_Propagator_type(prop_cit->first);
      if (temp_type == NemoPhys::Fermion_retarded_self || temp_type == NemoPhys::Boson_retarded_self)
      {
        NEMO_ASSERT(self_name == "", prefix + "found more than one retarded self-energy\n");
        self_name = prop_cit->first;
      }
    }
  }

  //2. distinguish eq and neq energy
  NEMO_ASSERT(options.check_option("energy_mesh_type"), prefix + "please define energy mesh type through option 'energy_mesh_type', either eq or neq\n");
  bool eq_energy_mesh;
  std::string energy_mesh_type = options.get_option("energy_mesh_type", std::string(""));
  if (energy_mesh_type == "eq")
  {
    eq_energy_mesh = true;
  }
  else if (energy_mesh_type == "neq")
  {
    eq_energy_mesh = false;
  }
  else
    throw std::invalid_argument("unrecognized energy_mesh_type, either eq or neq\n");

  //3. get needed options
  // 1=start lead, 2=end lead
  bool calc_spectral_in_backward = options.get_option("calculate_spectral_in_backward", false);
  bool use_backward_RGF_solver = options.get_option("use_backward_RGF_solver", false);
  double temperature1 = options.get_option("temperature_for_" + self_name, NemoPhys::temperature)*NemoPhys::boltzmann_constant / NemoPhys::elementary_charge;
  double temperature2 = options.get_option("temperature_for_second_lead", NemoPhys::temperature)*NemoPhys::boltzmann_constant / NemoPhys::elementary_charge;
  NEMO_ASSERT(options.check_option("chemical_potential_for_" + self_name), prefix + "please define \"chemical_potential_for_" + self_name + "\"\n");
  double chem_pot1 = options.get_option("chemical_potential_for_" + self_name, 0.0);  //start leads
  double chem_pot2 = options.get_option("chemical_potential_for_second_lead", 0.0);  //end leads

  //4. get the full retarded or advanced Green's function if needed(assuming it is given on the full device)
  PetscMatrixParallelComplex* retarded_Green = NULL;
  PetscMatrixParallelComplex* advanced_Green = NULL;
  if (!calc_spectral_in_backward)
  {
    std::string tic_toc_G = tic_toc_prefix + " get Green function";
    NemoUtils::tic(tic_toc_G);
    if (retarded_G != NULL && advanced_G == NULL)
    {
      GreensfunctionInterface* temp_data_source = find_source_of_Greensfunction(this_simulation, retarded_G->get_name());
      if (PropOptionInterface->get_particle_type_is_Fermion())
        temp_data_source->get_Greensfunction(momentum_point, retarded_Green, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          NemoPhys::Fermion_retarded_Green);
      else
        temp_data_source->get_Greensfunction(momentum_point, retarded_Green, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          NemoPhys::Boson_retarded_Green);

      //translate retarded Green container into a matrix if there is need to calculate spectral
      if (!calc_spectral_in_backward)
      {
        std::string tic_toc_assemble = tic_toc_prefix + " assemble container";
        NemoUtils::tic(tic_toc_assemble);
        if (retarded_Green->if_container())
          retarded_Green->assemble();
        NemoUtils::toc(tic_toc_assemble);
      }

      if (!options.get_option("custom_mult", true))
      {
        advanced_Green = new PetscMatrixParallelComplex(retarded_Green->get_num_cols(),
          retarded_Green->get_num_rows(),
          retarded_Green->get_communicator());
        retarded_Green->hermitian_transpose_matrix(*advanced_Green, MAT_INITIAL_MATRIX);
      }
    }
    else
      throw std::invalid_argument(prefix + "both, retarded and advanced Green's function pointers are NULL\n");
    NemoUtils::toc(tic_toc_G);

    if (PropOptionInterface->get_debug_output())
      retarded_Green->save_to_matlab_file("Gr.m");
  }

  //5. calculate G<
  if (eq_energy_mesh)
  {//eq energy
    //5.1 get energy
    std::complex<double> energy = read_complex_energy_from_momentum(this_simulation, momentum_point, writeable_Propagator);

    //5.2 check if the energy is a pole
    //find the mesh constructor of the complex energy
    bool ispole;
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it = PropInterface->get_momentum_mesh_types().begin();
    std::string energy_name = std::string("");
    for (; momentum_name_it != PropInterface->get_momentum_mesh_types().end() && energy_name == std::string(""); ++momentum_name_it)
      if (momentum_name_it->second == NemoPhys::Complex_energy)
        energy_name = momentum_name_it->first;
    std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
    std::map<std::string, Simulation*>::const_iterator temp_cit = Mesh_Constructors.find(energy_name);
    NEMO_ASSERT(temp_cit != Mesh_Constructors.end(), "Propagation(\"" + this_simulation->get_name() + "\")::do_solve_lesser_equilibrium have not found constructor of mesh \"" + energy_name + "\"\n");
    Simulation* mesh_constructor = temp_cit->second;
    mesh_constructor->get_data(energy, ispole);

    bool calculate_dos = options.get_option("calculate_dos", false);

    //5.3 get diagonal of Gr
    NEMO_ASSERT(!calc_spectral_in_backward && retarded_Green != NULL, prefix + "backward RGF of eq energy point should store GR instead of spectral\n");
    if (result != NULL)
    {
      delete result;
      result = NULL;
    }
    retarded_Green->get_diagonal_matrix(result);

    //5.4 check if G< or Jacobian is needed
    bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);
    bool eq_energy_contribute_to_jacobian = options.get_option("eq_energy_contribute_to_jacobian", true);

    //5.5 check if energy mesh is non_rectangular
    InputOptions& energy_mesh_options = mesh_constructor->get_reference_to_options();
    bool non_rectangular = energy_mesh_options.get_option(std::string("non_rectangular"), false);

    if (!differentiate_over_energy)
    { //5.6 calculate equilibrium G lesser
      if (!ispole) 
      { // energy is not a pole
        std::complex<double> fermi_factor = NemoMath::complex_fermi_distribution(chem_pot1, temperature1, energy);
        if (calculate_dos)
        {
          fermi_factor = 1.0;
          cout << "calculating A instead of G<\n";
        }
        std::complex<double> Glesser_weight(1.0, 0.0);
        if (!non_rectangular)
        {
          mesh_constructor->get_data("integration_weight", momentum_point[0], Glesser_weight);
        }
        else
        {
          std::vector<double> temp_kvector = PropagationUtilities::read_kvector_from_momentum(this_simulation, momentum_point, writeable_Propagator);
          NemoMeshPoint temp_momentum(0, temp_kvector);
          std::vector<NemoMeshPoint> temp_kmomentum(1, temp_momentum);
          std::vector<double> temp_energy(2, 0.0);
          temp_energy[0] = energy.real();
          temp_energy[1] = energy.imag();
          NemoMeshPoint temp_emomentum(0, temp_energy);
          mesh_constructor->get_data("integration_weight", temp_kmomentum, temp_emomentum, Glesser_weight);
        }
        *result *= Glesser_weight * fermi_factor;  // weight*GR*fL
        result->imaginary_part();   // imag(weight*GR*fL)
        *result *= std::complex<double>(0.0, 1.0);  //G< = i*imag(weight*GR*fL)
      }
      else
      {// energy is a pole
        result->real_part();  // real(GR)
        *result *= std::complex<double>(0.0, 4.0*NemoMath::pi*NemoPhys::temperature*NemoPhys::boltzmann_constant / NemoPhys::elementary_charge);  // G< = i*4*pi*k*T*real(GR)
      }
    }
    else
    {   
      //5.6 calculate equilibrium G lesser for Jacobian
      if(!eq_energy_contribute_to_jacobian)
        *result *= std::complex<double>(0.0, 0.0);  // G< = 0.0
      else
      {
        if (!ispole)
        { // energy is not a pole
          std::complex<double> fermi_factor = NemoMath::complex_dfermi_distribution_over_dE(chem_pot1, temperature1, energy);  //(dfL/dE)
          std::complex<double> Glesser_weight(1.0, 0.0);
          if (!non_rectangular)
          {
            mesh_constructor->get_data("integration_weight", momentum_point[0], Glesser_weight);
          }
          else
          {
            std::vector<double> temp_kvector = PropagationUtilities::read_kvector_from_momentum(this_simulation, momentum_point, writeable_Propagator);
            NemoMeshPoint temp_momentum(0, temp_kvector);
            std::vector<NemoMeshPoint> temp_kmomentum(1, temp_momentum);
            std::vector<double> temp_energy(2, 0.0);
            temp_energy[0] = energy.real();
            temp_energy[1] = energy.imag();
            NemoMeshPoint temp_emomentum(0, temp_energy);
            mesh_constructor->get_data("integration_weight", temp_kmomentum, temp_emomentum, Glesser_weight);
          }
          *result *= Glesser_weight * fermi_factor;  // weight*GR*(dfL/dE)
          result->imaginary_part();   // imag(weight*GR*(dfL/dE))
          *result *= std::complex<double>(0.0, 1.0);  //G< = i*imag(weight*GR*(dfL/dE))
        }
        else
        { // energy is a pole
          //find the two nearby energies from mesh constructor
          ComplexContourEnergyGrid* complex_energy_mesh_constructor = dynamic_cast<ComplexContourEnergyGrid*>(mesh_constructor);
          NEMO_ASSERT(complex_energy_mesh_constructor != NULL, prefix + "energy mesh constructor is not ComplexContourEnergyGrid\n");
          std::complex<double> E1(0.0, 0.0);
          std::complex<double> E2(0.0, 0.0);
          std::vector<double> temp_energy(2, 0.0);
          NemoMeshPoint temp_emomentum(0, temp_energy);
          std::vector<NemoMeshPoint> mesh_point1(1, temp_emomentum);
          std::vector<NemoMeshPoint> mesh_point2(1, temp_emomentum);
          complex_energy_mesh_constructor->get_data(momentum_point, E1, E2, mesh_point1, mesh_point2);

          //get the Fermion retarded Green's function at these two energy points
          PetscMatrixParallelComplex* GR1 = NULL;
          PetscMatrixParallelComplex* GR2 = NULL;
          GreensfunctionInterface* temp_data_source = find_source_of_Greensfunction(this_simulation, retarded_G->get_name());

          temp_data_source->get_Greensfunction(mesh_point1, GR1, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL, NemoPhys::Fermion_retarded_Green);
          temp_data_source->get_Greensfunction(mesh_point2, GR2, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL, NemoPhys::Fermion_retarded_Green);

          //calculate the derivative of retarded Green's function
          PetscMatrixParallelComplex* dGRdE = NULL;
          PropagationUtilities::calculate_derivative_of_G_over_E(this_simulation, E1, E2, dGRdE, GR1, GR2);

          dGRdE->get_diagonal_matrix(result);
          result->real_part();
          *result *= std::complex<double>(0.0, -4.0*NemoMath::pi*NemoPhys::temperature*NemoPhys::boltzmann_constant / NemoPhys::elementary_charge);

          delete dGRdE;
          dGRdE = NULL;
        }
      }
    }
  }
  else
  {//neq energy
    //5.1 get energy
    std::complex<double> complex_energy = read_complex_energy_from_momentum(this_simulation, momentum_point, writeable_Propagator);
    NEMO_ASSERT(complex_energy.imag() == 0.0, prefix + "imaginary part of neq energy point is not 0.0\n");
    double energy = complex_energy.real();

    //5.2 calculate Gamma^R if needed
    PetscMatrixParallelComplex* full_self_matrix = NULL;
    if (!calc_spectral_in_backward)
    {
      std::string tic_toc_sigma = tic_toc_prefix + " get self-energy";
      NemoUtils::tic(tic_toc_sigma);

      //neq.1.1 get the one relevant retarded contact self-energy
      PetscMatrixParallelComplex* small_self_matrix;
      Simulation* data_source = find_source_of_data(this_simulation, self_name);
      data_source->get_data(self_name, &momentum_point, small_self_matrix); //NOTE: we use the default NULL as the Domain* to get the full self-energy
      NemoUtils::toc(tic_toc_sigma);
      const DOFmapInterface& self_DOFmap = data_source->get_const_dof_map(this_simulation->get_const_simulation_domain());

      //neq.1.2 convert the self-energy matrix into the full device size (defined by the Hamilton_Constructor) - if required
      Domain* this_domain = this_simulation->get_simulation_domain();
      std::string tic_toc_convert = tic_toc_prefix + " convert self-energy into device size";
      NemoUtils::tic(tic_toc_convert);
      const DOFmapInterface& full_DOFmap = Hamilton_Constructor->get_const_dof_map(this_domain);
      if (Hamilton_Constructor->get_const_simulation_domain() != data_source->get_const_simulation_domain())
      {
        convert_self_energy_to_device_size_RGF(this_simulation, small_self_matrix, self_DOFmap, full_DOFmap, full_self_matrix);
      }
      else
      {
        //just copy the self-energy to full_self_matrix
        full_self_matrix = new PetscMatrixParallelComplex(*small_self_matrix);
      }

      PetscMatrixParallelComplex* gamma = NULL;
      calculate_gamma_given_sigma(this_simulation, full_self_matrix, gamma);
      delete full_self_matrix;
      full_self_matrix = gamma;
      gamma = NULL;

      NemoUtils::toc(tic_toc_convert);
    }

    //5.3 calculate A^R
    std::string tic_toc_AR = tic_toc_prefix + " get AR";
    NemoUtils::tic(tic_toc_AR);
    if (result != NULL)
    {
      delete result;
      result = NULL;
    }
    PetscMatrixParallelComplex* temp_matrix = NULL;

    if (!calc_spectral_in_backward) //need to solve A^R from Gamma^R
    {
      if (!options.get_option("custom_mult", true))
      {
        std::string tic_toc_gamma_mult_Ga = tic_toc_prefix + " gamma x Ga";
        NemoUtils::tic(tic_toc_gamma_mult_Ga);
        PetscMatrixParallelComplex::mult(*full_self_matrix, *advanced_Green, &temp_matrix);//gamma*Ga
        if (advanced_G == NULL)
          delete advanced_Green;
        NemoUtils::toc(tic_toc_gamma_mult_Ga);
        std::string tic_toc_Gr_gamma_Ga = tic_toc_prefix + " Gr x gamma x Ga";
        NemoUtils::tic(tic_toc_Gr_gamma_Ga);
        PetscMatrixParallelComplex::mult(*retarded_Green, *temp_matrix, &result); // AR=Gr*gamma*Ga
        NemoUtils::toc(tic_toc_Gr_gamma_Ga);
      }
      else
      {
        std::string tic_toc_gamma_mult_Ga = tic_toc_prefix + " custom_mult";
        NemoUtils::tic(tic_toc_gamma_mult_Ga);
        custom_matrix_product_RGF(this_simulation, 0, retarded_Green, full_self_matrix, result); //new version of AR=Gr*gamma*Ga
        NemoUtils::toc(tic_toc_gamma_mult_Ga);
      }
    }
    else // A^R already solved in backward RGF
    {
      NemoPhys::Propagator_type Fermion_AL_type;
      NemoPhys::Propagator_type Boson_AL_type;
      if (use_backward_RGF_solver)
      {
        Fermion_AL_type = NemoPhys::Fermion_one_side_spectral_Green;
        Boson_AL_type = NemoPhys::Boson_one_side_spectral_Green;
      }
      else
      {
        Fermion_AL_type = NemoPhys::Fermion_spectral_Green;
        Boson_AL_type = NemoPhys::Boson_spectral_Green;
      }
      std::map<std::string, Simulation*>::const_iterator prop_cit1 = Propagators->begin();
      std::string spectral_name = "";
      for (; prop_cit1 != Propagators->end(); prop_cit1++)
      {
        NemoPhys::Propagator_type temp_type = PropInterface->get_Propagator_type(prop_cit1->first);
        if (temp_type == Fermion_AL_type || temp_type == Boson_AL_type)
        {
          if (!use_backward_RGF_solver)
          {
            if (prop_cit1->first.find("GR") == std::string::npos)
              spectral_name = prop_cit1->first;
          }
          else
            spectral_name = prop_cit1->first;
        }
      }
      PetscMatrixParallelComplex* temp1 = NULL;
      GreensfunctionInterface* temp_data_source1 = find_source_of_Greensfunction(this_simulation, spectral_name);
      if (PropOptionInterface->get_particle_type_is_Fermion())
        temp_data_source1->get_Greensfunction(momentum_point, temp1, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          Fermion_AL_type);
      else
        temp_data_source1->get_Greensfunction(momentum_point, temp1, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          Boson_AL_type);
      temp1->get_diagonal_matrix(result);
    }
    NemoUtils::toc(tic_toc_AR);

    //5.4 check if G< or Jacobian is needed
    bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);

    //5.5 get spectral function A if Jacobian is needed
    PetscMatrixParallelComplex* full_spectral = NULL;
    if (differentiate_over_energy)
    {
      std::string tic_toc_A = tic_toc_prefix + " get A";
      NemoUtils::tic(tic_toc_A);

      if (calc_spectral_in_backward)
      {
        std::map<std::string, Simulation*>::const_iterator prop_cit1 = Propagators->begin();
        std::string spectral_name = "";
        for (; prop_cit1 != Propagators->end(); prop_cit1++)
        {
          NemoPhys::Propagator_type temp_type = PropInterface->get_Propagator_type(prop_cit1->first);
          if (temp_type == NemoPhys::Fermion_spectral_Green || temp_type == NemoPhys::Boson_spectral_Green)
          {
            if (!use_backward_RGF_solver)
            {
              if (prop_cit1->first.find("GR") != std::string::npos)
                spectral_name = prop_cit1->first;
            }
            else
              spectral_name = prop_cit1->first;
          }
        }
        GreensfunctionInterface* temp_data_source2 = find_source_of_Greensfunction(this_simulation, spectral_name);
        PetscMatrixParallelComplex* temp1 = NULL;
        if (PropOptionInterface->get_particle_type_is_Fermion())
          temp_data_source2->get_Greensfunction(momentum_point, temp1, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
            NemoPhys::Fermion_spectral_Green);
        else
          temp_data_source2->get_Greensfunction(momentum_point, temp1, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
            NemoPhys::Boson_spectral_Green);
        temp1->get_diagonal_matrix(full_spectral);
      }
      else
      {
        NEMO_ASSERT(retarded_Green != NULL, prefix + "GR pointer is NULL\n");
        retarded_Green->get_diagonal_matrix(full_spectral);
        if (retarded_G == NULL)
          delete retarded_Green;

        full_spectral->imaginary_part(); //cplx(Im(GR),0)

        *full_spectral *= std::complex<double>(-2.0, 0.0); // A = 2i*Im(GR) = i (GR-GA)
      }
      NemoUtils::toc(tic_toc_A);
    }

    if (!differentiate_over_energy)
    { //5.5 calculate non-equilibrium G lesser
      double fermi_factor1 = NemoMath::fermi_distribution(chem_pot1, temperature1, energy);
      double fermi_factor2 = NemoMath::fermi_distribution(chem_pot2, temperature2, energy);
      *result *= std::complex<double>(0.0, 1.0)*(fermi_factor2 - fermi_factor1); //result = i * (f2 - f1) * A^R
    }
    else
    { //5.5 calculate non-equilibrium G lesser for Jacobian
      double fermi_factor1 = NemoMath::dfermi_distribution_over_dE(chem_pot1, temperature1, energy);
      double fermi_factor2 = NemoMath::dfermi_distribution_over_dE(chem_pot2, temperature2, energy);
      *result *= std::complex<double>(0.0, 1.0)*(fermi_factor2 - fermi_factor1); //result = i*[(df2/dE)-(df1/dE)]*A^R
      *full_spectral *= std::complex<double>(0.0, 1.0)*fermi_factor1; //full_spectral = i*(df1/dE)*A
      result->add_matrix(*full_spectral, SAME_NONZERO_PATTERN, std::complex<double>(1.0, 0.0)); //result = i*(df1/dE)*A + i*[(df2/dE)-(df1/dE)]*A^R
      delete full_spectral;
    }
  }

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::calculate_derivative_of_G_over_E(Simulation* this_simulation, const std::complex<double> E1, const std::complex<double> E2, PetscMatrixParallelComplex*& result,
  PetscMatrixParallelComplex*& G1, PetscMatrixParallelComplex*& G2)
{
  std::string tic_toc_prefix = "PropagationUtilities(\"" + this_simulation->get_name() + "\")::solve_derivative_of_G ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\"" + this_simulation->get_name() + "\")::solve_derivative_of_G ";

  NEMO_ASSERT(G1 != NULL && G2 != NULL, prefix + "neither Green's function can be null\n");

  std::complex<double> dE(E2.real() - E1.real(), E2.imag() - E1.imag());
  double modulo_square = std::pow(dE.real(), 2.0) + std::pow(dE.imag(), 2.0);
  std::complex<double> dE_inverse(dE.real() / modulo_square, -dE.imag() / modulo_square);

  if (result != NULL)
  {
    delete result;
    result = NULL;
  }

  result = new PetscMatrixParallelComplex(*G2);
  result->add_matrix(*G1, DIFFERENT_NONZERO_PATTERN, std::complex<double>(-1.0, 0.0)); //result = G2-G1
  *result *= dE_inverse; //result = (G2-G1)/dE

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::solve_lesser_Green_RGF(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result,
    const Propagator* retarded_G, const Propagator* advanced_G)
{

    std::string tic_toc_prefix = "Greensolver(\""+this_simulation->get_name()+"\")::solve_lesser_Green_RGF ";
    NemoUtils::tic(tic_toc_prefix);

    std::string prefix = "Greensolver(\""+this_simulation->get_name()+"\")::solve_lesser_Green_RGF ";

    PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
    PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
    Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();

    std::map<std::string,Simulation*>* Propagators=PropInterface->list_Propagator_constructors();

    //1. check that there is exactly one retarded self-energy in the readable Propagator list and store its name
    std::map<std::string, Simulation*>::const_iterator prop_cit=Propagators->begin();
    Propagator* writeable_Propagator=NULL;
    PropInterface->get_Propagator(writeable_Propagator);
    std::string self_name="";
    std::string name_of_writeable_Propagator = writeable_Propagator->get_name();
    for(; prop_cit!=Propagators->end(); prop_cit++)
    {
      //write_prop_cit=writeable_Propagators.find(prop_cit->first);
      //if(write_prop_cit==writeable_Propagators.end())
      if(prop_cit->first!=name_of_writeable_Propagator)
      {
        NemoPhys::Propagator_type temp_type=PropInterface->get_Propagator_type(prop_cit->first);
        if(temp_type==NemoPhys::Fermion_retarded_self || temp_type == NemoPhys::Boson_retarded_self)
        {
          NEMO_ASSERT(self_name=="",prefix+"found more than one retarded self-energy\n");
          self_name=prop_cit->first;
        }
      }
    }


    //2. get the properties of the two contacts
    InputOptions options = this_simulation->get_reference_to_options();
    bool use_backward_RGF_solver = options.get_option("use_backward_RGF_solver", false);
    double temperature1 = options.get_option("temperature_for_"+self_name,NemoPhys::temperature)*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
    double temperature2 = options.get_option("temperature_for_second_lead",NemoPhys::temperature)*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
    NEMO_ASSERT(options.check_option("chemical_potential_for_"+self_name),prefix+"please define \"chemical_potential_for_"+self_name+"\"\n");
    double chem_pot1    = options.get_option("chemical_potential_for_"+self_name,0.0);
    double chem_pot2    = options.get_option("chemical_potential_for_second_lead",0.0);
    double threshold_energy_for_lead1 = options.get_option("threshold_energy_for_"+self_name,-1E10);
    double threshold_energy_for_lead2    = options.get_option("threshold_energy_for_second_lead",-1E10);
    double energy = read_energy_from_momentum(this_simulation, momentum_point, writeable_Propagator);
    bool lead1_electron_like = true;
    if(energy<threshold_energy_for_lead1)
      lead1_electron_like=false;
    bool lead2_electron_like = true;
    if(energy<threshold_energy_for_lead2)
      lead2_electron_like=false;

    PetscMatrixParallelComplex* full_self_matrix = NULL;
    bool calc_spectral_in_backward=options.get_option("calculate_spectral_in_backward",false);
    bool store_DOS=options.get_option("store_DOS_only",false);

    //3. get the full retarded or advanced Green's function if needed(assuming it is given on the full device)
    PetscMatrixParallelComplex* retarded_Green = NULL;
    PetscMatrixParallelComplex* advanced_Green = NULL;
    if (!calc_spectral_in_backward || !store_DOS)
    {
      std::string tic_toc_G = tic_toc_prefix + " get Green function";
      NemoUtils::tic(tic_toc_G);
      if (retarded_G != NULL && advanced_G == NULL)
      {
        GreensfunctionInterface* temp_data_source = find_source_of_Greensfunction(this_simulation, retarded_G->get_name());
        if (PropOptionInterface->get_particle_type_is_Fermion())
          temp_data_source->get_Greensfunction(momentum_point, retarded_Green, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
            NemoPhys::Fermion_retarded_Green);
        else
          temp_data_source->get_Greensfunction(momentum_point, retarded_Green, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
            NemoPhys::Boson_retarded_Green);

        //translate retarded Green container into a matrix if there is need to calculate spectral
        if (!calc_spectral_in_backward)
        {
          std::string tic_toc_assemble = tic_toc_prefix + " assemble container";
          NemoUtils::tic(tic_toc_assemble);
          if (retarded_Green->if_container())
            retarded_Green->assemble();
          NemoUtils::toc(tic_toc_assemble);
        }

        if (!options.get_option("custom_mult", true))
        {
          advanced_Green = new PetscMatrixParallelComplex(retarded_Green->get_num_cols(),
            retarded_Green->get_num_rows(),
            retarded_Green->get_communicator());
          retarded_Green->hermitian_transpose_matrix(*advanced_Green, MAT_INITIAL_MATRIX);
        }
      }
      else
        throw std::invalid_argument(prefix + "both, retarded and advanced Green's function pointers are NULL\n");
      NemoUtils::toc(tic_toc_G);

      if (PropOptionInterface->get_debug_output())
        retarded_Green->save_to_matlab_file("Gr.m");
    }

    //4. calculate Gamma^L for A^L if spectral is not calculated in backward
    if(!calc_spectral_in_backward)
    {
      std::string tic_toc_sigma= tic_toc_prefix+" get self-energy";
      NemoUtils::tic(tic_toc_sigma);
      //4.1 get the one relevant retarded contact self-energy
      PetscMatrixParallelComplex* small_self_matrix;
      Simulation* data_source=find_source_of_data(this_simulation,self_name);
      data_source->get_data(self_name,&momentum_point,small_self_matrix); //NOTE: we use the default NULL as the Domain* to get the full self-energy
      NemoUtils::toc(tic_toc_sigma);
      const DOFmapInterface& self_DOFmap = data_source->get_const_dof_map(this_simulation->get_const_simulation_domain());

      //4.2 convert the self-energy matrix into the full device size (defined by the Hamilton_Constructor) - if required
      Domain* this_domain = this_simulation->get_simulation_domain();
      std::string tic_toc_convert= tic_toc_prefix+" convert self-energy into device size";
      NemoUtils::tic(tic_toc_convert);
      const DOFmapInterface& full_DOFmap = Hamilton_Constructor->get_const_dof_map(this_domain);
      //const int number_of_rows=full_DOFmap.get_global_dof_number();
      if(Hamilton_Constructor->get_const_simulation_domain()!=data_source->get_const_simulation_domain())
      {
        convert_self_energy_to_device_size_RGF(this_simulation, small_self_matrix, self_DOFmap, full_DOFmap, full_self_matrix);
      }
      else
      {
        //just copy the self-energy to full_self_matrix
        full_self_matrix = new PetscMatrixParallelComplex(*small_self_matrix);
      }

      PetscMatrixParallelComplex* gamma = NULL;
      calculate_gamma_given_sigma(this_simulation, full_self_matrix, gamma);
      delete full_self_matrix;
      full_self_matrix = gamma;
      gamma = NULL;
 
      NemoUtils::toc(tic_toc_convert);
    }

    //5. get the spectral function A^L (see Eq.55 of J. Appl. Phys. 81, 7845)
    std::string tic_toc_AL= tic_toc_prefix+" get AL";
    NemoUtils::tic(tic_toc_AL);
    if(result!=NULL)
    {
      delete result;
      result = NULL;
    }
    PetscMatrixParallelComplex* temp_matrix = NULL;

    if(!calc_spectral_in_backward)
    {
      //5.1 solve A^L
      if(!options.get_option("custom_mult",true))
      {
        std::string tic_toc_gamma_mult_Ga= tic_toc_prefix+" gamma x Ga";
        NemoUtils::tic(tic_toc_gamma_mult_Ga);
        PetscMatrixParallelComplex::mult(*full_self_matrix,*advanced_Green,&temp_matrix);//gamma*Ga
        if(advanced_G==NULL)
          delete advanced_Green;
        NemoUtils::toc(tic_toc_gamma_mult_Ga);
        std::string tic_toc_Gr_gamma_Ga= tic_toc_prefix+" Gr x gamma x Ga";
        NemoUtils::tic(tic_toc_Gr_gamma_Ga);
        PetscMatrixParallelComplex::mult(*retarded_Green,*temp_matrix,&result); // AL=Gr*gamma*Ga
        NemoUtils::toc(tic_toc_Gr_gamma_Ga);
      }
      else
      {
        std::string tic_toc_gamma_mult_Ga= tic_toc_prefix+" custom_mult";
        NemoUtils::tic(tic_toc_gamma_mult_Ga);
        custom_matrix_product_RGF(this_simulation, 0, retarded_Green, full_self_matrix,result); //new version of AL=Gr*gamma*Ga
        NemoUtils::toc(tic_toc_gamma_mult_Ga);
      }
    }
    else
    {
      NemoPhys::Propagator_type Fermion_AL_type;
      NemoPhys::Propagator_type Boson_AL_type;
      if (use_backward_RGF_solver)
      {
        Fermion_AL_type = NemoPhys::Fermion_one_side_spectral_Green;
        Boson_AL_type = NemoPhys::Boson_one_side_spectral_Green;
      }
      else
      {
        Fermion_AL_type = NemoPhys::Fermion_spectral_Green;
        Boson_AL_type = NemoPhys::Boson_spectral_Green;
      }

      std::map<std::string, Simulation*>::const_iterator prop_cit1=Propagators->begin();
      std::string spectral_name="";
      for (; prop_cit1 != Propagators->end(); prop_cit1++)
      {
        NemoPhys::Propagator_type temp_type = PropInterface->get_Propagator_type(prop_cit1->first);
        if (temp_type == Fermion_AL_type || temp_type == Boson_AL_type)
        {
          if (!use_backward_RGF_solver)
          {
            if (prop_cit1->first.find("GR") == std::string::npos)
              spectral_name = prop_cit1->first;
          }
          else
            spectral_name = prop_cit1->first;
        }
      }
      //Simulation* temp_data_source1=find_source_of_data(spectral_name);
      //NEMO_ASSERT(spectral_name!=std::string(""),tic_toc_prefix+"spectral function solver not defined\n");
      PetscMatrixParallelComplex* temp1=NULL;
      //std::string tic_toc_getAL= tic_toc_prefix+" get_data spectral_function";
      //NemoUtils::tic(tic_toc_getAL);
      GreensfunctionInterface* temp_data_source1=find_source_of_Greensfunction(this_simulation,spectral_name);
      if (PropOptionInterface->get_particle_type_is_Fermion())
        temp_data_source1->get_Greensfunction(momentum_point, temp1, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          Fermion_AL_type);
      else
        temp_data_source1->get_Greensfunction(momentum_point, temp1, &(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          Boson_AL_type);
      //temp_data_source1->get_data(spectral_name,&momentum_point,temp1,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      //NemoUtils::toc(tic_toc_getAL);
      temp1->get_diagonal_matrix(result);
    }
    NemoUtils::toc(tic_toc_AL);

    //6. get the full spectral function A 
    std::string tic_toc_A = tic_toc_prefix + " get A";
    NemoUtils::tic(tic_toc_A);

    delete full_self_matrix;
    delete temp_matrix;
    temp_matrix=NULL;

    PetscMatrixParallelComplex* diagonal_GR=NULL;
    if(store_DOS)
    {
      std::map<std::string, Simulation*>::const_iterator prop_cit1=Propagators->begin();
      std::string spectral_name="";
      for(; prop_cit1!=Propagators->end(); prop_cit1++)
      {
        NemoPhys::Propagator_type temp_type=PropInterface->get_Propagator_type(prop_cit1->first);
        if(temp_type==NemoPhys::Fermion_spectral_Green || temp_type==NemoPhys::Boson_spectral_Green)
        {
          if (!use_backward_RGF_solver)
          {
            if (prop_cit1->first.find("GR") != std::string::npos)
              spectral_name = prop_cit1->first;
          }
          else
            spectral_name = prop_cit1->first;
        }
      }
      //NEMO_ASSERT(spectral_name!=std::string(""),tic_toc_prefix+"spectral function solver not defined\n");
      //Simulation* temp_data_source2=find_source_of_data(spectral_name);
      GreensfunctionInterface* temp_data_source2=find_source_of_Greensfunction(this_simulation,spectral_name);
      PetscMatrixParallelComplex* temp1=NULL;
      if(PropOptionInterface->get_particle_type_is_Fermion())
        temp_data_source2->get_Greensfunction(momentum_point,temp1,&(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())),NULL,
                                              NemoPhys::Fermion_spectral_Green);
      else
        temp_data_source2->get_Greensfunction(momentum_point,temp1,&(Hamilton_Constructor->get_dof_map(this_simulation->get_const_simulation_domain())),NULL,
                                              NemoPhys::Boson_spectral_Green);
      //temp_data_source2->get_data(spectral_name,&momentum_point,temp1,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      temp1->get_diagonal_matrix(diagonal_GR);
    }
    else
    {
      std::string tic_toc_Gr_minus_Ga= tic_toc_prefix+" Gr-Ga";
      NemoUtils::tic(tic_toc_Gr_minus_Ga);
      //M.P.: SAME_NONZERO_PATTERN does not work
      retarded_Green->get_diagonal_matrix(diagonal_GR);
      if(retarded_G==NULL || options.get_option("delete_GR_after_use", false))
        delete retarded_Green;

      diagonal_GR->imaginary_part(); //cplx(Im(GR),0)
      NemoUtils::toc(tic_toc_Gr_minus_Ga);

      *diagonal_GR *= std::complex<double>(-2.0,0.0); // A = 2i*Im(GR) = i (GR-GA)
    }
    if(PropOptionInterface->get_debug_output())
    {
      result->save_to_matlab_file("AL.m");
      diagonal_GR->save_to_matlab_file("A.m");
    }

    diagonal_GR->add_matrix(*result,SAME_NONZERO_PATTERN,std::complex<double>(-1.0, 0.0)); // A-A^L

    NemoUtils::toc(tic_toc_A);

    //7. solve Eq.56 of J. Appl. Phys. 81, 7845 which gives the lesser Green's function (store in result)
    std::string tic_toc_Glesser= tic_toc_prefix+" solve G_lesser";
    NemoUtils::tic(tic_toc_Glesser);

    if(!options.get_option("electron_hole_model",false))
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(momentum_point);
      translate_momentum_vector(this_simulation,temp_pointer,data_string);

      bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);

      //double energy = read_energy_from_momentum(momentum_point,output_Propagator);
      double temp_factor1 = 0.0;
      double temp_factor2 = 0.0;
      if(!PropOptionInterface->get_use_analytical_momenta())
      {

        if(!differentiate_over_energy) //need to check whether electron or hole
        {
          if(lead1_electron_like)
            temp_factor1 = NemoMath::fermi_distribution(chem_pot1,temperature1,energy);
          else // -1*(1-f)
            temp_factor1 = NemoMath::fermi_distribution(chem_pot1,temperature1,energy) - 1.0;

          if(lead2_electron_like)
            temp_factor2 = NemoMath::fermi_distribution(chem_pot2,temperature2,energy);
          else
            temp_factor2 = NemoMath::fermi_distribution(chem_pot2,temperature2,energy)- 1.0;
        }
        else //no need to check (compensating signs)
        {
          temp_factor1 = NemoMath::dfermi_distribution_over_dE(chem_pot1,temperature1,energy);
          temp_factor2 = NemoMath::dfermi_distribution_over_dE(chem_pot2,temperature2,energy);
        }
      }
      else  //analytical momenta
      {
        std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
        NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n");

        double periodic_space_volume = this_simulation->get_const_simulation_domain()->return_periodic_space_volume();
        double analytical_momenta_me = options.get_option("analytical_momenta_me",-1.0);
        double analytical_momenta_mh = options.get_option("analytical_momenta_mh",-1.0);

        double electron_mass = -1.0;
        double hole_mass = -1.0;
        //this has to be done here not in the core
        //if(momentum_type!="")
        {
          std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
          Simulation* temp_simulation=this_simulation->find_simulation(temp_simulation_name);
          NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
          //if(electron_type)
          if(analytical_momenta_me==-1.0)
            temp_simulation->get_data("averaged_effective_mass_conduction_band",electron_mass);
          else
            electron_mass = analytical_momenta_me*NemoPhys::electron_mass;

          if(analytical_momenta_mh==-1.0)
            temp_simulation->get_data("averaged_effective_mass_valence_band",hole_mass);
          else
            hole_mass = analytical_momenta_mh*NemoPhys::electron_mass;

        }

        if(lead1_electron_like)
        {
          temp_factor1 = get_analytically_integrated_distribution(energy, temperature1, chem_pot1, momentum_type,
              true, differentiate_over_energy, false,electron_mass, periodic_space_volume);
        }
        else
        {
            temp_factor1 = get_analytically_integrated_distribution(energy, temperature1, chem_pot1, momentum_type,
                false, differentiate_over_energy, false,hole_mass, periodic_space_volume);
        }


        if(lead2_electron_like)
        {
          if(options.check_option("analytical_momenta_me"))
            temp_factor2 = get_analytically_integrated_distribution(energy, temperature2, chem_pot2, momentum_type,
                true, differentiate_over_energy, false, electron_mass, periodic_space_volume);
        }
        else
        {
            temp_factor2 = get_analytically_integrated_distribution(energy, temperature2, chem_pot2, momentum_type,
                false, differentiate_over_energy, false,hole_mass, periodic_space_volume);
        }

      }
      *result *= std::complex<double>(0.0,1.0)*temp_factor2; //result = i * f2 * A^L
      *diagonal_GR *= std::complex<double>(0.0,1.0)*temp_factor1; //temp_matrix = i * f1 *(A-A^L)

      //if (!options.get_option("LRA_RGF", bool(false)))
      result->add_matrix(*diagonal_GR,SAME_NONZERO_PATTERN); //result = i * f1 * A^L + i * f2 *(A-A^L)
      //else
      //{
      //  result->add_matrix(*diagonal_GR,SAME_NONZERO_PATTERN);
      //  //transform result using transformation matrix from

      //  Hamilton_Constructor->get_data("diagonal_glesser", &momentum_point, result);

      //  //change DOFmap solver back to original DOFmap solver
      //  InputOptions trans_opt = Hamilton_Constructor->get_options();
      //  std::string real_space_dofmap_solver = trans_opt.get_option("DOFmap_solver",std::string(""));
      //  options.set_option("DOFmap_solver", real_space_dofmap_solver); //for density calculation
      //}
    }
    else  // differentate between electrons and holes via hole factor
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(momentum_point);
      translate_momentum_vector(this_simulation,temp_pointer,data_string);

      bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);

      double energy = read_energy_from_momentum(this_simulation,momentum_point,output_Propagator);

      if(!options.get_option("position_dependent_hole_factor",true))
      {
        std::vector<double> hole_factors;
        Hamilton_Constructor->get_data(std::string("boundary_hole_factors"),energy, hole_factors);
        NEMO_ASSERT(hole_factors.size()==2,prefix+"received hole factors of wrong size\n");

        double hole_factor1=hole_factors[0];
        double hole_factor2=hole_factors[1];
        double electron_distribution1 = 0.0;
        double electron_distribution2 = 0.0;
        double hole_distribution1 = 0.0;
        double hole_distribution2 = 0.0;
        if(!PropOptionInterface->get_use_analytical_momenta())
        {
          if(!differentiate_over_energy)
          {
            electron_distribution1 = NemoMath::fermi_distribution(chem_pot1,temperature1,energy);
            electron_distribution2 = NemoMath::fermi_distribution(chem_pot2,temperature2,energy);
          }
          else
          {
            electron_distribution1 = NemoMath::dfermi_distribution_over_dE(chem_pot1,temperature1,energy);
            electron_distribution2 = NemoMath::dfermi_distribution_over_dE(chem_pot2,temperature2,energy);
          }
          *result *= std::complex<double>(0.0,1.0)*(electron_distribution1-hole_factor1); //result = i * f1 * A^L
          *diagonal_GR *= std::complex<double>(0.0,1.0)*(electron_distribution2-hole_factor2); //temp_matrix = i * f2 *(A-A^L)
          //if (!options.get_option("LRA_RGF", bool(false)))
          result->add_matrix(*diagonal_GR,SAME_NONZERO_PATTERN); //result = i * f1 * A^L + i * f2 *(A-A^L)
          //else
          //{
          //  result->add_matrix(*diagonal_GR,SAME_NONZERO_PATTERN);
          //  //transform result using transformation matrix from
          //  Hamilton_Constructor->get_data("diagonal_glesser", &momentum_point, result);
          //  //change DOFmap solver back to original DOFmap solver
          //  InputOptions trans_opt = Hamilton_Constructor->get_options();
          //  std::string real_space_dofmap_solver = trans_opt.get_option("DOFmap_solver",std::string(""));
          //  options.set_option("DOFmap_solver", real_space_dofmap_solver); //for density calculation
          //}
        }
        else //analytical integration + hole_factor
        {
          std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
          NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n");

          double periodic_space_volume = this_simulation->get_const_simulation_domain()->return_periodic_space_volume();
          double analytical_momenta_me = options.get_option("analytical_momenta_me",-1.0);
          double analytical_momenta_mh = options.get_option("analytical_momenta_mh",-1.0);

          double electron_mass = -1.0;
          double hole_mass = -1.0;
          //this has to be done here not in the core
          //if(momentum_type!="")
          {
            std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
            Simulation* temp_simulation=this_simulation->find_simulation(temp_simulation_name);
            NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
            //if(electron_type)
            if(analytical_momenta_me==-1.0)
              temp_simulation->get_data("averaged_effective_mass_conduction_band",electron_mass);
            else
              electron_mass = analytical_momenta_me*NemoPhys::electron_mass;

            if(analytical_momenta_mh==-1.0)
              temp_simulation->get_data("averaged_effective_mass_valence_band",hole_mass);
            else
              hole_mass = analytical_momenta_mh*NemoPhys::electron_mass;

          }
          NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n");
          if(options.check_option("analytical_momenta_me") && options.check_option("analytical_momenta_mh"))
          {
           // double me = options.get_option("analytical_momenta_me",1.0);
            //double mh = options.get_option("analytical_momenta_mh",1.0);
            electron_distribution1 = get_analytically_integrated_distribution(energy, temperature1, chem_pot1, momentum_type,
                true, differentiate_over_energy, false, electron_mass, periodic_space_volume);
            electron_distribution2 = get_analytically_integrated_distribution(energy, temperature2, chem_pot2, momentum_type,
                true, differentiate_over_energy, false,electron_mass, periodic_space_volume);
            hole_distribution1 =  get_analytically_integrated_distribution(energy, temperature1, chem_pot1, momentum_type,
                false, differentiate_over_energy, false,hole_mass, periodic_space_volume);
            hole_distribution2 = get_analytically_integrated_distribution(energy, temperature2, chem_pot2, momentum_type,
                false, differentiate_over_energy, false,hole_mass, periodic_space_volume);
          }
          else
          {
            electron_distribution1 = get_analytically_integrated_distribution(this_simulation, energy, temperature1, chem_pot1, momentum_type, true, differentiate_over_energy, false);
            electron_distribution2 = get_analytically_integrated_distribution(this_simulation, energy, temperature2, chem_pot2, momentum_type, true, differentiate_over_energy, false);
            hole_distribution1 =  get_analytically_integrated_distribution(this_simulation, energy, temperature1, chem_pot1, momentum_type, false, differentiate_over_energy, false);
            hole_distribution2 = get_analytically_integrated_distribution(this_simulation, energy, temperature2, chem_pot2, momentum_type, false, differentiate_over_energy, false);
          }

          *result *= std::complex<double>(0.0,1.0)*(electron_distribution1-hole_factor1*electron_distribution1-hole_factor1+hole_factor1*hole_distribution1);
          *diagonal_GR *= std::complex<double>(0.0,1.0)*(electron_distribution2-hole_factor2*electron_distribution2-hole_factor2+hole_factor2*hole_distribution2);
          result->add_matrix(*diagonal_GR,SAME_NONZERO_PATTERN);
        }
      }
      else //position_dependent_hole_factor
      {
        std::map<unsigned int, double> threshold_map;
        std::map<unsigned int, double> threshold_map_right;
        Hamilton_Constructor->get_data(std::string("hole_factor_dof"),energy, threshold_map, threshold_map_right);
        std::vector<complex<double> > AL;
        std::vector<complex<double> > A_minus_AL;
        std::vector<complex<double> > G_lesser;
        std::vector<int> indices;
        result->get_diagonal(&AL);
        diagonal_GR->get_diagonal(&A_minus_AL);

        //loop over all degrees of freedom
        for(unsigned int j=0; j<AL.size(); j++)
        {
          std::complex<double> temp(std::complex<double>(0.0,0.0));
          std::map<unsigned int, double>::const_iterator temp_threshold_it = threshold_map.find(j);
          std::map<unsigned int, double>::const_iterator temp_threshold_R_it = threshold_map_right.find(j);
          NEMO_ASSERT(temp_threshold_it!=threshold_map.end(),prefix+"have not found a DOFindex in threshold map\n");
          NEMO_ASSERT(temp_threshold_R_it!=threshold_map_right.end(),prefix+"have not found a DOFindex in right side threshold map\n");

          double Hole_fact_L = temp_threshold_it->second;
          double Hole_fact_R = temp_threshold_R_it->second;

          double electron_distribution1 = 0.0;
          double electron_distribution2 = 0.0;
          double hole_distribution1 = 0.0;
          double hole_distribution2 = 0.0;
          if(!PropOptionInterface->get_use_analytical_momenta())
          {
            if (!differentiate_over_energy)
            {
              electron_distribution1 = NemoMath::fermi_distribution(chem_pot1,temperature1,energy);
              electron_distribution2 = NemoMath::fermi_distribution(chem_pot2,temperature2,energy);
              hole_distribution1 = 1-electron_distribution1;
              hole_distribution2 = 1-electron_distribution2;
            }
            else
            {
              electron_distribution1 = NemoMath::dfermi_distribution_over_dE(chem_pot1,temperature1,energy);
              electron_distribution2 = NemoMath::dfermi_distribution_over_dE(chem_pot2,temperature2,energy);
              hole_distribution1 = -NemoMath::dfermi_distribution_over_dE(chem_pot1,temperature1,energy);
              hole_distribution2 = -NemoMath::dfermi_distribution_over_dE(chem_pot2,temperature2,energy);
            }
          }
          else
          {
            double periodic_space_volume = this_simulation->get_const_simulation_domain()->return_periodic_space_volume();
            double analytical_momenta_me = options.get_option("analytical_momenta_me",-1.0);
            double analytical_momenta_mh = options.get_option("analytical_momenta_mh",-1.0);

            double electron_mass = -1.0;
            double hole_mass = -1.0;
            //this has to be done here not in the core
            //if(momentum_type!="")
            {
              std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
              Simulation* temp_simulation=this_simulation->find_simulation(temp_simulation_name);
              NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
              //if(electron_type)
              if(analytical_momenta_me==-1.0)
                temp_simulation->get_data("averaged_effective_mass_conduction_band",electron_mass);
              else
                electron_mass = analytical_momenta_me*NemoPhys::electron_mass;

              if(analytical_momenta_mh==-1.0)
                temp_simulation->get_data("averaged_effective_mass_valence_band",hole_mass);
              else
                hole_mass = analytical_momenta_mh*NemoPhys::electron_mass;

            }

            std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
            NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n");
            if(options.check_option("analytical_momenta_me") && options.check_option("analytical_momenta_mh"))
            {
//              double me = options.get_option("analytical_momenta_me",1.0);
  //            double mh = options.get_option("analytical_momenta_mh",1.0);
              electron_distribution1 = get_analytically_integrated_distribution(energy, temperature1, chem_pot1, momentum_type,
                  true, differentiate_over_energy, false,electron_mass,periodic_space_volume);
              electron_distribution2 = get_analytically_integrated_distribution(energy, temperature2, chem_pot2, momentum_type,
                  true, differentiate_over_energy, false,electron_mass,periodic_space_volume);
              hole_distribution1 =  get_analytically_integrated_distribution(energy, temperature1, chem_pot1, momentum_type,
                  false, differentiate_over_energy, false,hole_mass,periodic_space_volume);
              hole_distribution2 = get_analytically_integrated_distribution(energy, temperature2, chem_pot2, momentum_type,
                  false, differentiate_over_energy, false,hole_mass,periodic_space_volume);
            }
            else
            {
              electron_distribution1 = get_analytically_integrated_distribution(this_simulation, energy, temperature1, chem_pot1, momentum_type, true, differentiate_over_energy, false);
              electron_distribution2 = get_analytically_integrated_distribution(this_simulation, energy, temperature2, chem_pot2, momentum_type, true, differentiate_over_energy, false);
              hole_distribution1 =  get_analytically_integrated_distribution(this_simulation, energy, temperature1, chem_pot1, momentum_type, false, differentiate_over_energy, false);
              hole_distribution2 = get_analytically_integrated_distribution(this_simulation, energy, temperature2, chem_pot2, momentum_type, false, differentiate_over_energy, false);
            }
          }
          //G<(i) = i*(1-C_h(i))(fs*A^L+fd*(A-A^L)) - i*C_h((1-fs)A^L+(1-fd)(A-A^L))
          temp = std::complex<double>(0.0,1.0)*((1-Hole_fact_R)*electron_distribution2*AL[j]+(1-Hole_fact_L)*electron_distribution1*A_minus_AL[j]) -
                 std::complex<double>(0.0,1.0)*(Hole_fact_R*hole_distribution2*AL[j]+Hole_fact_L*hole_distribution1*A_minus_AL[j]);
          G_lesser.push_back(temp);
          indices.push_back(j);
        }
        //convert G_lesser to a petsc vector
        PetscVectorNemo<std::complex<double> > temp_vector(result->get_num_rows(),result->get_num_rows(),result->get_communicator());
        temp_vector.set_values(indices,G_lesser);
        result->matrix_diagonal_shift(temp_vector,INSERT_VALUES);
      }
    }
    delete diagonal_GR;
    NemoUtils::toc(tic_toc_Glesser);
    NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::solve_correlation_Green_RGF2(Simulation* this_simulation, Propagator*& /*output_Propagator*/, const std::vector<NemoMeshPoint>& momentum_point,
    std::set<const Propagator*>& /*lesser_self*/
    ,PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+this_simulation->get_name()+"\")::solve_lesser_Green_RGF2 ";
  NemoUtils::tic(tic_toc_prefix);
  //std::string prefix = "Greensolver(\""+this->get_name()+"\")::solve_lesser_Green_RGF2 ";

  if(result!=NULL)
    delete result;
  result=NULL;

  //1. get the contact self-energies (either retarded and lesser or retarded and greater)
  //consistency check: there is only one writeable Propagator and that is either G< or G>
  //NEMO_ASSERT(writeable_Propagators.size()==1,prefix+"called with more than one writeable Propagator\n");

  //1.1 loop over the readable Propagators and store the Propagator pointers (including the self-enegies for point 2.)
  //1.2 get the solvers of all Propagators (including the self-energies for 2.1)
  Simulation* correlation_sigma_solver = NULL;
  const Propagator* correlation_sigma=NULL;
  Simulation* retarded_sigma_solver = NULL;
  const Propagator* retarded_sigma=NULL;
  GreensfunctionInterface* half_way_gR_solver = NULL;
  //const Propagator* half_way_gR=NULL;
  GreensfunctionInterface* half_way_correlation_solver = NULL;
  //const Propagator* half_way_correlation=NULL;


  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  //Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();
  InputOptions options = this_simulation->get_reference_to_options();

  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  //std::map<std::string, Propagator*>::const_iterator write_prop_cit=writeable_Propagators.begin();
  std::string self_name="";
  std::string name_of_writeable_Propagator = writeable_Propagator->get_name();

  std::map<std::string,Simulation*>* Propagators=PropInterface->list_Propagator_constructors();
  std::map<std::string, Simulation*>::const_iterator prop_c_it=Propagators->begin();
  for(; prop_c_it!=Propagators->end(); prop_c_it++)
  {
    NemoPhys::Propagator_type temp_prop_type = PropInterface->get_Propagator_type(prop_c_it->first);
    Simulation* temp_simulation = find_source_of_data(this_simulation,prop_c_it->first);
    if(temp_prop_type==NemoPhys::Fermion_lesser_Green||temp_prop_type==NemoPhys::Boson_lesser_Green)
    {
      //make sure prop_c_it->second is not within the writeable Propagators (which is the full correlation G)
      //std::map<std::string, Propagator*>::const_iterator temp_cit=writeable_Propagators.find(prop_c_it->first);
      //if(temp_cit==writeable_Propagators.end())
      if(prop_c_it->first!=name_of_writeable_Propagator)
      {
        half_way_correlation_solver=dynamic_cast<GreensfunctionInterface*>(temp_simulation);
        //half_way_correlation=prop_c_it->second;
      }
    }
    else if(temp_prop_type==NemoPhys::Fermion_lesser_self||temp_prop_type==NemoPhys::Boson_lesser_self)
    {
      correlation_sigma_solver=temp_simulation;
      //correlation_sigma=prop_c_it->second;
    }
    else if(temp_prop_type==NemoPhys::Fermion_retarded_self||temp_prop_type==NemoPhys::Boson_retarded_self)
    {
      retarded_sigma_solver=temp_simulation;
      //retarded_sigma=prop_c_it->second;
    }
    else if(temp_prop_type==NemoPhys::Fermion_retarded_Green||temp_prop_type==NemoPhys::Boson_retarded_Green)
    {
      half_way_gR_solver=dynamic_cast<GreensfunctionInterface*>(temp_simulation);
      //half_way_gR=prop_c_it->second;
    }
    else
      throw std::runtime_error(tic_toc_prefix+"found Propagator \""+prop_c_it->first+"\" with inappropriate type\n");
  }
  NEMO_ASSERT(correlation_sigma_solver!=NULL,tic_toc_prefix+"have not found the solver of the correlation self-energy\n");
  NEMO_ASSERT(retarded_sigma_solver!=NULL,tic_toc_prefix+"have not found the solver of the retarded self-energy\n");
  NEMO_ASSERT(half_way_correlation_solver!=NULL,tic_toc_prefix+"have not found the solver of the half-way correlation Green's function\n");
  NEMO_ASSERT(half_way_gR_solver!=NULL,tic_toc_prefix+"have not found the solver of the half-way retarded Green's function\n");

  //2. get the half-way Green's functions (retarded or advanced) and lesser or greater
  //2.1 get the solvers of the Propagators (done in 1.2)

  //3. perform the product of gr*sigma_R

  bool use_modified_correlation = options.get_option("use_modified_correlation_equation",bool(true));

  //call core function
  //need: correlation_sigma_matrix
  PetscMatrixParallelComplex* correlation_sigma_matrix = NULL;
  correlation_sigma_solver->get_data(correlation_sigma->get_name(),&momentum_point,correlation_sigma_matrix,
      &this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()),NULL);
  if(PropOptionInterface->get_sigma_convert_dense())
    correlation_sigma_matrix->matrix_convert_dense();

  //need: retarded_sigma_matrix
  PetscMatrixParallelComplex* retarded_sigma_matrix= NULL;
  retarded_sigma_solver->get_data(retarded_sigma->get_name(),&momentum_point,retarded_sigma_matrix,&this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()));
  if(PropOptionInterface->get_sigma_convert_dense())
    retarded_sigma_matrix->matrix_convert_dense();

  //need: half_way_correlation_matrix
  PetscMatrixParallelComplex* half_way_correlation_matrix= NULL;
  if(PropOptionInterface->get_particle_type_is_Fermion())
    half_way_correlation_solver->get_Greensfunction(momentum_point,half_way_correlation_matrix,
        &this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()),NULL,
        NemoPhys::Fermion_lesser_Green);
  else
    half_way_correlation_solver->get_Greensfunction(momentum_point,half_way_correlation_matrix,
        &this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()),NULL,
        NemoPhys::Boson_lesser_Green);

  //need: half_way_gR_solver
  PetscMatrixParallelComplex* half_way_gR_matrix = NULL;
  if(PropOptionInterface->get_particle_type_is_Fermion())
    half_way_gR_solver->get_Greensfunction(momentum_point,half_way_gR_matrix,&this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()),NULL,
        NemoPhys::Fermion_retarded_Green);
  else
    half_way_gR_solver->get_Greensfunction(momentum_point,half_way_gR_matrix,&this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()),NULL,NemoPhys::Boson_retarded_Green);

  //const Domain* name_domain = this_simulation->get_const_simulation_domain();
  //correlation_sigma_matrix->save_to_matlab_file("SigmaL_GL_"+name_domain->get_name()+".m");
  //retarded_sigma_matrix->save_to_matlab_file("SigmaR_GL_"+name_domain->get_name()+".m");
  //half_way_gR_matrix->save_to_matlab_file("gR_GL_"+name_domain->get_name()+".m");
  //half_way_correlation_matrix->save_to_matlab_file("gL_GL_"+name_domain->get_name()+".m");

	NemoMatrixInterface *NMI_result = NULL;
	NemoMatrixInterface *NMI_correlation_sigma_matrix =
			correlation_sigma_matrix->convert_to_NMI(DEFAULT_MAT_TYPE);
	NemoMatrixInterface *NMI_retarded_sigma_matrix =
			retarded_sigma_matrix->convert_to_NMI(DEFAULT_MAT_TYPE);
	NemoMatrixInterface *NMI_half_way_correlation_matrix =
			half_way_correlation_matrix->convert_to_NMI(DEFAULT_MAT_TYPE);
	NemoMatrixInterface *NMI_half_way_gR_matrix =
			half_way_gR_matrix->convert_to_NMI(DEFAULT_MAT_TYPE);
  //call core function
	core_correlation_Green_RGF2(this_simulation, NMI_correlation_sigma_matrix,
			NMI_retarded_sigma_matrix, NMI_half_way_correlation_matrix, NMI_half_way_gR_matrix,
			NMI_result, use_modified_correlation);
  //result->save_to_matlab_file("GL_"+this_simulation->get_const_simulation_domain()->get_name()+".m");
	result =
			dynamic_cast<PetscMatrixParallelComplex *>(NMI_result->convert_to_PMPC());

	NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE, NMI_result);
	NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE,
			NMI_correlation_sigma_matrix);
	NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE,
			NMI_retarded_sigma_matrix);
	NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE,
			NMI_retarded_sigma_matrix);
	NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE, NMI_half_way_gR_matrix);

  //debugging:
  if(PropOptionInterface->get_debug_output())
    result->save_to_matlab_file("test_result.m");

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::core_correlation_Green_RGF2(Simulation* this_simulation, NemoMatrixInterface* correlation_sigma_matrix, NemoMatrixInterface* retarded_sigma_matrix,
		NemoMatrixInterface* half_way_correlation_matrix, NemoMatrixInterface* half_way_gR_matrix,
		NemoMatrixInterface*& result, bool modified_correlation)
{

  //J.C. This modification uses gR*sigma_correlation*g_correlation - (gR*sigma_correlation*(g_correlation)')'
    //     instead of gR*sigma_correlation*g_correlation - (gR*sigma_correlation*g_correlation)'
    //3.1 get the actual matrices of sigmaR and g_correlation using get_data

    NemoMatrixInterface* temp_matrix = NULL;
    NemoMatrixInterface* temp_matrix3 = NULL;

    if(modified_correlation)
    {
      PropagationUtilities::supress_noise(this_simulation,half_way_gR_matrix);
      PropagationUtilities::supress_noise(this_simulation,retarded_sigma_matrix);
      NemoMatrixInterface::mult(*half_way_gR_matrix,*retarded_sigma_matrix,&temp_matrix);
      PropagationUtilities::supress_noise(this_simulation,temp_matrix);
      //3.3 perform the product of temp_matrix*g_correlation
      //PetscMatrixParallelComplex* half_way_correlation_matrix= NULL;
      //half_way_correlation_solver->get_data(half_way_correlation->get_name(),&momentum_point,half_way_correlation_matrix,&get_const_dof_map(get_const_simulation_domain()));

      //PetscMatrixParallelComplex* temp_matrix3 = NULL;
      PropagationUtilities::supress_noise(this_simulation,half_way_correlation_matrix);
      NemoMatrixInterface::mult(*temp_matrix,*half_way_correlation_matrix,&temp_matrix3);
      PropagationUtilities::supress_noise(this_simulation,temp_matrix3);
      //conjugate of half_way_correlation_matrix
      //PetscMatrixParallelComplex temp_matrix_conj(half_way_correlation_matrix->get_num_cols(),half_way_correlation_matrix->get_num_rows(),
      //    half_way_correlation_matrix->get_communicator());
		  NemoMatrixInterface* temp_matrix_conj = NemoFactory::matrix_instance(
		  		half_way_correlation_matrix->get_factory_type(), half_way_correlation_matrix->get_num_cols(),
				half_way_correlation_matrix->get_num_rows(),
				half_way_correlation_matrix->get_communicator());
      half_way_correlation_matrix->hermitian_transpose_matrix(*temp_matrix_conj,NEMO_MAT_INITIAL_MATRIX);
      NemoMatrixInterface* temp_matrix2 = NULL;
      PropagationUtilities::supress_noise(this_simulation,temp_matrix_conj);
      NemoMatrixInterface::mult(*temp_matrix,*temp_matrix_conj,&temp_matrix2);
      PropagationUtilities::supress_noise(this_simulation,temp_matrix2);
		  delete temp_matrix_conj;
		  temp_matrix_conj = NULL;
      {
        //PetscMatrixParallelComplex hermitian_transpose_result3(temp_matrix3->get_num_cols(),temp_matrix3->get_num_rows(),
        //    temp_matrix3->get_communicator());
			  NemoMatrixInterface* hermitian_transpose_result3 =
					NemoFactory::matrix_instance(temp_matrix3->get_factory_type(),
							temp_matrix3->get_num_cols(), temp_matrix3->get_num_rows(),
							temp_matrix3->get_communicator());
        temp_matrix2->hermitian_transpose_matrix(*hermitian_transpose_result3,NEMO_MAT_INITIAL_MATRIX);
        //4.2 perform the difference result_of_3-(result_of_3)^dagger and store in temp_result
        temp_matrix3->add_matrix(*hermitian_transpose_result3,NEMO_DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
        //4.3 delete (result_of_3)^dagger (done by closing the block)
        delete hermitian_transpose_result3;
        hermitian_transpose_result3 = NULL;
      }
      delete temp_matrix2;
      temp_matrix2 = NULL;
    }
    else
    {
      //3. perform the product of gR*sigma_R*g_correlation (< or >)
      //3.1 get the actual matrices of sigmaR and g_correlation using get_data

    	NemoMatrixInterface::mult(*retarded_sigma_matrix,*half_way_correlation_matrix,&temp_matrix);

    	NemoMatrixInterface::mult(*half_way_gR_matrix,*temp_matrix,&temp_matrix3);
      //3.4 delete temp_matrix, delete sigma_R
      delete temp_matrix;
      temp_matrix=NULL;
      //delete the sigma_R
      //4. solve result_of_3-(result_of_3)^dagger
      //4.1 create conjg_transpose(temp_result2)
      {
        //PetscMatrixParallelComplex hermitian_transpose_result3(temp_matrix3->get_num_cols(),temp_matrix3->get_num_rows(),
        //    temp_matrix3->get_communicator());
			  NemoMatrixInterface* hermitian_transpose_result3 =
					NemoFactory::matrix_instance(temp_matrix3->get_factory_type(),
							temp_matrix3->get_num_cols(), temp_matrix3->get_num_rows(),
							temp_matrix3->get_communicator());
        temp_matrix3->hermitian_transpose_matrix(*hermitian_transpose_result3,NEMO_MAT_INITIAL_MATRIX);
        //4.2 perform the difference result_of_3-(result_of_3)^dagger and store in temp_result
        temp_matrix3->add_matrix(*hermitian_transpose_result3,NEMO_DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));
        //4.3 delete (result_of_3)^dagger (done by closing the block)
        delete hermitian_transpose_result3;
        hermitian_transpose_result3 = NULL;
      }
    }

    //5. solve gR*sigma_correlation*gA
    //5.1 get the sigma_correlation_matrix

    //5.2 perform the product gR*sigma_correlation and store in temp_matrix
    PropagationUtilities::supress_noise(this_simulation,correlation_sigma_matrix);
    NemoMatrixInterface::mult(*half_way_gR_matrix,*correlation_sigma_matrix,&temp_matrix);
    PropagationUtilities::supress_noise(this_simulation,temp_matrix);
    //5.3 perform the product of temp_matrix*gA and store in temp_matrix5
    {
      //half_way_gR_matrix->save_to_matlab_file("half_way_gR_matrix.m");
      //PetscMatrixParallelComplex half_way_gA_matrix(half_way_gR_matrix->get_num_cols(),
      //    half_way_gR_matrix->get_num_rows(),
      //    half_way_gR_matrix->get_communicator());
		  NemoMatrixInterface* half_way_gA_matrix = NemoFactory::matrix_instance(
		  		half_way_gR_matrix->get_factory_type(), half_way_gR_matrix->get_num_cols(),
				half_way_gR_matrix->get_num_rows(),
				half_way_gR_matrix->get_communicator());
      half_way_gR_matrix->hermitian_transpose_matrix(*half_way_gA_matrix,NEMO_MAT_INITIAL_MATRIX);
      PropagationUtilities::supress_noise(this_simulation,half_way_gA_matrix);
      NemoMatrixInterface::mult(*temp_matrix,*half_way_gA_matrix,&result);
      PropagationUtilities::supress_noise(this_simulation,result);
      delete half_way_gA_matrix;
      half_way_gA_matrix = NULL;
    }
    //5.4 delete the temp_matrix
    delete temp_matrix;
    temp_matrix=NULL;

    //6. solve result_5(result)+result_4(temp_matrix3)
    PropagationUtilities::supress_noise(this_simulation,temp_matrix3);
    result->add_matrix(*temp_matrix3,NEMO_DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
    delete temp_matrix3;
    temp_matrix3=NULL;
    //7.(taken from OMEN): add g<
    PropagationUtilities::supress_noise(this_simulation,half_way_correlation_matrix);
    result->add_matrix(*half_way_correlation_matrix,NEMO_DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
    result->consider_as_full();
}


void PropagationUtilities::do_solve_lesser_equilibrium(Simulation* this_simulation, Propagator*& output_Propagator,const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+this_simulation->get_name()+"\")::do_solve_lesser_equilibrium ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this_simulation->get_name()+"\")::do_solver_lesser_equilibrium(): ";

  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
   PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
   Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();
   InputOptions options = this_simulation->get_reference_to_options();
  bool complex_energy=PropInterface->complex_energy_used();
  //1. find the retarded and/or the advanced Green's functions or self-energies in the list of Propagators
  const Propagator* retarded_Propagator=NULL;
  const Propagator* advanced_Propagator=NULL;
  PetscMatrixParallelComplex* temp_advanced=NULL;
  PetscMatrixParallelComplex* temp_retarded=NULL;
  /*if(exact_GR_solver!=NULL)
  {
    exact_GR_solver->get_Propagator(retarded
  }*/


  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  //std::map<std::string, Propagator*>::const_iterator write_prop_cit=writeable_Propagators.begin();
  std::string self_name="";
  std::string name_of_writeable_Propagator = writeable_Propagator->get_name();

  std::map<std::string,Simulation*>* Propagators=PropInterface->list_Propagator_constructors();
  std::map<std::string, Simulation*>::const_iterator prop_c_it=Propagators->begin();

  std::map<std::string, Simulation*>::iterator it=Propagators->begin();
  for(; it!=Propagators->end(); it++)
  {
    PropagatorInterface* temp_interface=get_PropagatorInterface(it->second);
    Propagator* temp_Propagator=NULL;
    temp_interface->get_Propagator(temp_Propagator);

    NemoPhys::Propagator_type p_type=PropInterface->get_Propagator_type(temp_Propagator->get_name());
    //2. find the retarded and/or advanced Green's function
    if(p_type == NemoPhys::Fermion_retarded_Green ||p_type == NemoPhys::Boson_retarded_Green||p_type == NemoPhys::Fermion_retarded_self ||
        p_type == NemoPhys::Boson_retarded_self)
    {
      //if(it->second==NULL)
      //{
      //  Propagators->find(it->first)->second->get_data(it->first,retarded_Propagator);
      //  it->second=retarded_Propagator;
      //}
      //else
      NEMO_ASSERT(it->second!=NULL,prefix + " retarded_Propagator constructor is NULL");
        retarded_Propagator=temp_Propagator;
    }
    else if(p_type == NemoPhys::Fermion_advanced_Green ||p_type == NemoPhys::Boson_advanced_Green||p_type == NemoPhys::Fermion_advanced_self ||
            p_type == NemoPhys::Boson_advanced_self)
    {
      //if(it->second==NULL)
      //{
      //  Propagators->find(it->first)->second->get_data(it->first,advanced_Propagator);
      //  it->second=advanced_Propagator;
      //}
      //else
      NEMO_ASSERT(it->second!=NULL,prefix + " retarded_Propagator constructor is NULL");
        advanced_Propagator=temp_Propagator;
    }
  }
  //get the retarded and advanced matrices
  if(retarded_Propagator!=NULL)
  {
    //Simulation* retarded_solver = pointer_to_Propagator_Constructors->find(retarded_Propagator->get_name())->second;
    Simulation* retarded_solver = find_source_of_data(this_simulation,retarded_Propagator->get_name());
    retarded_solver->get_data(retarded_Propagator->get_name(),&momentum_point,temp_retarded,
                              &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));
    if(advanced_Propagator==NULL)
    {
      temp_advanced=new PetscMatrixParallelComplex(temp_retarded->get_num_cols(),
          temp_retarded->get_num_rows(),
          temp_retarded->get_communicator());

      if(temp_retarded->if_container())
        temp_retarded->assemble();
      temp_retarded->hermitian_transpose_matrix(*temp_advanced,MAT_INITIAL_MATRIX);
    }
    else
    {
      //Simulation* advanced_solver = pointer_to_Propagator_Constructors->find(advanced_Propagator->get_name())->second;
      Simulation* advanced_solver = find_source_of_data(this_simulation,advanced_Propagator->get_name());
      advanced_solver->get_data(advanced_Propagator->get_name(),&momentum_point,temp_advanced,
                                &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));
    }
  }
  else if(advanced_Propagator!=NULL)
  {
    //Simulation* advanced_solver = pointer_to_Propagator_Constructors->find(advanced_Propagator->get_name())->second;
    Simulation* advanced_solver = find_source_of_data(this_simulation,advanced_Propagator->get_name());
    advanced_solver->get_data(advanced_Propagator->get_name(),&momentum_point,temp_advanced,
                              &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));
    if(retarded_Propagator==NULL)
    {
      temp_retarded=new PetscMatrixParallelComplex(temp_advanced->get_num_cols(),
          temp_advanced->get_num_rows(),
          temp_advanced->get_communicator());
      if(temp_advanced->if_container())
        temp_advanced->assemble();
      temp_advanced->hermitian_transpose_matrix(*temp_retarded,MAT_INITIAL_MATRIX);
    }
    else
    {
      //Simulation* retarded_solver = pointer_to_Propagator_Constructors->find(retarded_Propagator->get_name())->second;
      Simulation* retarded_solver = find_source_of_data(this_simulation,retarded_Propagator->get_name());
      retarded_solver->get_data(retarded_Propagator->get_name(),&momentum_point,temp_retarded,
                                &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));
    }
  }
  else
    throw std::invalid_argument(prefix+"at least one, either a retarded or an advanced Propagator must be given as input\n");

  //2. call the core routine to multiply by a factor for distribution and do the matrix operations
  //options that will be arguments to the core routine
  //a hack to calculate jacobian
  bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);
  double threshold_energy_for_lead = options.get_option("threshold_energy_for_lead",double(-1E10));
  double chemical_potential = PropOptionInterface->get_chemical_potential();
  double temperature1 = PropOptionInterface->get_temperature();

  bool lead_electron_like = true;
  double energy = 0.0;
  if (!complex_energy)
  {
    energy = read_energy_from_momentum(this_simulation, momentum_point, output_Propagator);
    if (energy < threshold_energy_for_lead)
      lead_electron_like = false;
  }

  bool particle_type_is_Fermion = PropOptionInterface->get_particle_type_is_Fermion();
  bool electron_hole_model = options.get_option("electron_hole_model",false);

  std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
  double periodic_space_volume = this_simulation->get_const_simulation_domain()->return_periodic_space_volume();
  double analytical_momenta_me = options.get_option("analytical_momenta_me",-1.0);
  double analytical_momenta_mh = options.get_option("analytical_momenta_mh",-1.0);

  double electron_mass = -1.0;
  double hole_mass = -1.0;
  //this has to be done here not in the core
  if(momentum_type!="")
  {
    std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
    Simulation* temp_simulation=this_simulation->find_simulation(temp_simulation_name);
    NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
    //if(electron_type)
    if(analytical_momenta_me==-1.0)
      temp_simulation->get_data("averaged_effective_mass_conduction_band",electron_mass);
    else
      electron_mass = analytical_momenta_me*NemoPhys::electron_mass;

    if(analytical_momenta_mh==-1.0)
      temp_simulation->get_data("averaged_effective_mass_valence_band",hole_mass);
    else
      hole_mass = analytical_momenta_mh*NemoPhys::electron_mass;

  }

  std::vector<double>* averaged_hole_factor_pointer = NULL;
  std::vector<double> averaged_hole_factor;
  if(electron_hole_model)
  {
    Hamilton_Constructor->get_data(std::string("averaged_hole_factor"),energy,averaged_hole_factor);
    averaged_hole_factor_pointer = &averaged_hole_factor;
  }

  if(!complex_energy)
  {
    double energy = read_energy_from_momentum(this_simulation,momentum_point,output_Propagator);

    //core_call
    core_lesser_equilibrium(energy,chemical_potential, temperature1, temp_retarded, temp_advanced, result, particle_type_is_Fermion,differentiate_over_energy,
        lead_electron_like, electron_hole_model, averaged_hole_factor_pointer, momentum_type, electron_mass, hole_mass, periodic_space_volume);
  }
  else
  {
    std::complex<double> energy=read_complex_energy_from_momentum(this_simulation, momentum_point,output_Propagator);
    double energy_real = energy.real();
    double energy_imag = energy.imag();

    NEMO_ASSERT(energy_imag!=-100,"imaginary energy component is the default argument. Did not expect this"); //Fabio: James is a careful guy
    
    //check if the energy is a pole
    //find the mesh constructor of the complex energy
    bool ispole;
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it = PropInterface->get_momentum_mesh_types().begin();
    std::string energy_name = std::string("");
    for (; momentum_name_it != PropInterface->get_momentum_mesh_types().end() && energy_name == std::string(""); ++momentum_name_it)
      if (momentum_name_it->second == NemoPhys::Complex_energy)
        energy_name = momentum_name_it->first;
    std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
    std::map<std::string, Simulation*>::const_iterator temp_cit = Mesh_Constructors.find(energy_name);
    NEMO_ASSERT(temp_cit != Mesh_Constructors.end(), "Propagation(\"" + this_simulation->get_name() + "\")::do_solve_lesser_equilibrium have not found constructor of mesh \"" + energy_name + "\"\n");
    Simulation* mesh_constructor = temp_cit->second;
    mesh_constructor->get_data(energy, ispole);

    InputOptions& energy_mesh_options = mesh_constructor->get_reference_to_options();
    bool non_rectangular = energy_mesh_options.get_option(std::string("non_rectangular"), false);

    std::complex<double> Glesser_weight(1.0, 0.0);
    if (!non_rectangular)
    {
      mesh_constructor->get_data("integration_weight", momentum_point[0], Glesser_weight);
    }
    else
    {
      std::vector<double> temp_kvector = PropagationUtilities::read_kvector_from_momentum(this_simulation, momentum_point, output_Propagator);
      NemoMeshPoint temp_momentum(0, temp_kvector);
      std::vector<NemoMeshPoint> temp_kmomentum(1, temp_momentum);
      std::vector<double> temp_energy(2, 0.0);
      temp_energy[0] = energy.real();
      temp_energy[1] = energy.imag();
      NemoMeshPoint temp_emomentum(0, temp_energy);
      mesh_constructor->get_data("integration_weight", temp_kmomentum, temp_emomentum, Glesser_weight);
    }

    //core call for complex energy
    core_lesser_equilibrium(energy_real,chemical_potential, temperature1, temp_retarded, temp_advanced, result, particle_type_is_Fermion,differentiate_over_energy,
            lead_electron_like, electron_hole_model, averaged_hole_factor_pointer, momentum_type, electron_mass, hole_mass, periodic_space_volume,energy_imag,ispole, Glesser_weight);
  }

  set_job_done_momentum_map(this_simulation,&(output_Propagator->get_name()),&momentum_point, true);
  if(retarded_Propagator==NULL)
    delete temp_retarded;
  if(advanced_Propagator==NULL)
    delete temp_advanced;
  PropagationUtilities::supress_noise(this_simulation,result);
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::core_lesser_equilibrium(double energy,double chemical_potential, double temperature, PetscMatrixParallelComplex* retarded_matrix,
    PetscMatrixParallelComplex* advanced_matrix, PetscMatrixParallelComplex*& result, bool particle_type_is_Fermion, bool differentiate_over_energy,
    bool lead_electron_like, bool electron_hole_model, std::vector<double>* averaged_hole_factor_pointer, std::string momentum_type,
    double electron_mass, double hole_mass, double periodic_space_volume, double imag_energy, bool ispole, std::complex<double> Glesser_weight)
{
  bool use_complex_energy = false;
  if(imag_energy != -100)//physically we know -100 eV is a huge negative energy
    use_complex_energy = true;
  double temperature_in_eV=temperature*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  bool use_analytical_momenta = false;
  if(momentum_type!="")
    use_analytical_momenta= true;


  if(!use_complex_energy)
  {
    delete result;
    result = new PetscMatrixParallelComplex(*retarded_matrix); //result = PR
    result->add_matrix(*advanced_matrix, DIFFERENT_NONZERO_PATTERN, std::complex<double>(-1.0, 0.0));   //result = PR-PA

    if(particle_type_is_Fermion)
    {
      if(!electron_hole_model)
      {
        if(!use_analytical_momenta)
        {
          if(lead_electron_like)
          {
            //electrons
            if (!differentiate_over_energy)
              *result *= std::complex<double>(-NemoMath::fermi_distribution(chemical_potential,temperature_in_eV,energy),
                  0.0);  //weight with Fermi distribution, result = -Fermi*(PR-PA)
            else
              *result *= std::complex<double>(-NemoMath::dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,energy),0.0);
          }
          else
          {
            //holes
            if (!differentiate_over_energy)
              *result *= std::complex<double>((1.0-NemoMath::fermi_distribution(chemical_potential,temperature_in_eV,energy)),
                  0.0);  //weight with 1-Fermi distribution, result = 1-Fermi*(PR-PA)
            else
              *result *= std::complex<double>(-NemoMath::dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,
                  energy),0.0);  //weight with 1-Fermi distribution, result = 1-Fermi*(PR-PA)
          }
        }
        else
        {
          //analytical momenta
          NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D","core lesser equilibrium called with unknown analytical_momenta \""+momentum_type+"\"\n");
          if(lead_electron_like)
          {
            //electron
            *result *= std::complex<double>(-get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, true,
                differentiate_over_energy, false, electron_mass, periodic_space_volume),0.0);
          }
          else //hole
          {
            *result *= std::complex<double>(get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, false,
                differentiate_over_energy, false, hole_mass, periodic_space_volume),0.0);
          }
        }
      }
      else //distinguishing electrons and holes via hole factor
      {
        //std::vector<double> temp_data;
        //Hamilton_Constructor->get_data(std::string("averaged_hole_factor"),energy,temp_data);
        NEMO_ASSERT(averaged_hole_factor_pointer!=NULL,"core_lesser_equilibrium electron hole model set with no averaged_hole_factor found");
        NEMO_ASSERT((*averaged_hole_factor_pointer)[0] ==(*averaged_hole_factor_pointer)[1],"unequal hole factors received\n");
        double hole_factor=(*averaged_hole_factor_pointer)[0];
        if(!use_analytical_momenta)
        {
          if(!differentiate_over_energy)
            *result*=std::complex<double>((hole_factor-NemoMath::fermi_distribution(chemical_potential,temperature_in_eV,energy)),0.0);
          else
            *result*=std::complex<double>(-NemoMath::dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,energy),0.0);
        }
        else //analytical integration + hole_factor
        {
          NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D"," core_lesser_equilibrium called with unknown analytical_momenta \""+momentum_type+"\"\n");
          PetscMatrixParallelComplex temp_matrix(*result);
          temp_matrix*=std::complex<double>(hole_factor*get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, false,
              differentiate_over_energy, false, electron_mass,periodic_space_volume),0.0);
          *result*=std::complex<double>(-(1.0-hole_factor)*get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, true,
              differentiate_over_energy, false, hole_mass, periodic_space_volume),0.0);
          result->add_matrix(temp_matrix,SAME_NONZERO_PATTERN);
        }
      }
    }
    else //for Bosons
    {
      //NEMO_ASSERT(!complex_energy,prefix+"complex energies for Bosons are not implemented yet\n");
      //if(options.get_option("analytical_kspace_1D",false) || options.get_option("analytical_kspace_2D",false))
      //  throw std::runtime_error("Propagation(\""+this_simulation->get_name()+"\")::analytical_kspace for Bosons is not implemented\n");
      //double energy=read_energy_from_momentum(this_simulation, momentum_point,output_Propagator);
      if (!differentiate_over_energy)
        *result *= std::complex<double>(-NemoMath::bose_distribution(chemical_potential,temperature_in_eV,energy),
            0.0);   //weight with Bose distribution, result = -Bose*(PR-PA)
      else
        *result *= std::complex<double>(-NemoMath::dbose_distribution_over_dE(chemical_potential,temperature_in_eV,energy),
            0.0);   //weight with Bose distribution, result = -Bose*(PR-PA)
    }

  }//end no complex energy
  else //complex energy
  {
    NEMO_ASSERT(particle_type_is_Fermion, "PropagationUtilities::core_lesser_equilibrium NYI for Bosons with complex energy");
    NEMO_ASSERT(!use_analytical_momenta,"PropagationUtilities::core_lesser_equilibrium NYI for analytical momenta with complex energy");
    NEMO_ASSERT(!differentiate_over_energy, "PropagationUtilities::core_lesser_equilibrium NYI for differentiate over energy with complex energy");

    std::complex<double> complex_energy=cplx(energy,imag_energy);
    delete result;
    result = new PetscMatrixParallelComplex(*retarded_matrix); //result = GR

    if (!ispole)
    {// energy is not a pole
      std::complex<double> fermi_factor = NemoMath::complex_fermi_distribution(chemical_potential, temperature_in_eV, complex_energy);
      *result *= Glesser_weight * fermi_factor;  // weight*GR*fL
      result->imaginary_part();   // imag(weight*GR*fL)
      *result *= std::complex<double>(0.0, 1.0);  //G< = i*imag(weight*GR*fL)
    }
    else
    {// energy is a pole
      result->real_part();  // real(GR)
      *result *= std::complex<double>(0.0, 4.0*NemoMath::pi*NemoPhys::temperature*NemoPhys::boltzmann_constant / NemoPhys::elementary_charge);  // G< = i*4*pi*k*T*real(GR)
    }
  }
}

void PropagationUtilities::solve_correlation_Green_nonlocalRGF(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point, std::set<const Propagator*>& lesser_self,
    PetscMatrixParallelComplex*& result, const Propagator* retarded_G, const Propagator* advanced_G)
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_correlation_Green_nonlocalRGF";
  std::string error_prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_correlation_Green_nonlocalRGF ";

  NemoUtils::tic(tic_toc_prefix);
  //call do_solve_lesser as before
  //solve_correlation_Green_exact(this_simulation, output_Propagator, momentum_point, result, lesser_self, retarded_G, advanced_G);

  InputOptions options = this_simulation->get_reference_to_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();

  NEMO_ASSERT(!PropInterface->complex_energy_used(),error_prefix + "NYI for complex energy");


  PetscMatrixParallelComplex* summed_lesser_self=NULL;
  PetscMatrixParallelComplex* temp_matrix=NULL;
  if(lesser_self.size()>0)
  {
    std::set<const Propagator*>::const_iterator lesser_it=lesser_self.begin();
    NEMO_ASSERT((*lesser_it)!=NULL,error_prefix+"self-energy pointer is NULL\n");
    std::string lesser_propagator_name=(*lesser_it)->get_name();

    Simulation* data_source=find_source_of_data(this_simulation, lesser_propagator_name);
    data_source->get_data(lesser_propagator_name,&momentum_point,temp_matrix,&(
        Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));
    //sum up all self-energies to a single matrix
    summed_lesser_self = new PetscMatrixParallelComplex(*temp_matrix);

    lesser_it++;
    for(; lesser_it!=lesser_self.end(); lesser_it++)
    {
      NEMO_ASSERT((*lesser_it)!=NULL,error_prefix+"self-energy pointer is NULL\n");
      lesser_propagator_name=(*lesser_it)->get_name();
      //ask the constructor of this Propagator to give the Matrix for this momentum (NOTE: maybe to be replaced with the solver defined in options...
      data_source=find_source_of_data(this_simulation,lesser_propagator_name);
      data_source->get_data(lesser_propagator_name,&momentum_point,temp_matrix,
          &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));

      NEMO_ASSERT(temp_matrix!=NULL,error_prefix+"received NULL for \""+lesser_propagator_name+"\" from \""+data_source->get_name()+"\"\n");
      cerr << "nonlocalg< summed self component " << lesser_propagator_name << " " << this_simulation->get_const_simulation_domain()->get_name() << " " << temp_matrix->get(0,0) << " \n";

      summed_lesser_self->add_matrix(*temp_matrix, DIFFERENT_NONZERO_PATTERN); //note: to speed up, we might want to use the same matrix pattern

      if(PropOptionInterface->get_debug_output())
      {
        std::string temp_string;
        const std::vector<NemoMeshPoint>* temp_pointer=&momentum_point;
        translate_momentum_vector(this_simulation,temp_pointer, temp_string);
        temp_string+="_dsl_"+this_simulation->get_output_suffix();//options.get_option("output_suffix",std::string(""));
        temp_matrix->save_to_matlab_file(lesser_propagator_name+temp_string+".m");
      }
    }
    if(PropOptionInterface->get_sigma_convert_dense())
      summed_lesser_self->matrix_convert_dense();
  }
  else
    throw std::runtime_error(error_prefix+"no lesser self-energies received\n");

  //summed lesser_self is the running sum of what is to be multiplied by Gr and Ga
  //get sigma<(i,i-1)
  std::string variable_name ="coupling_domain_Hamiltonian_constructor"; //"i-1"
  if(!options.check_option(variable_name))
    throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
  std::string coupling_H_constructor_name=options.get_option(variable_name,std::string(""));
  Simulation* coupling_H_constructor = this_simulation->find_simulation(coupling_H_constructor_name);
  const Domain* coupling_domain = coupling_H_constructor->get_const_simulation_domain();
  if(options.check_option("coupling_domain"))
  {
    std::string temp_domain_name=options.get_option("coupling_domain",std::string(""));
    coupling_domain=Domain::get_domain(temp_domain_name);
    NEMO_ASSERT(coupling_domain!=NULL,tic_toc_prefix+"have not found target_domain \""+temp_domain_name+"\"\n");
  }
  const DOFmapInterface& coupling_DOFmap = coupling_H_constructor->get_const_dof_map(coupling_domain);

  Simulation* scattering_sigmaR_solver = PropOptionInterface->get_scattering_sigmaR_solver();
  Simulation* scattering_sigmaL_solver = PropOptionInterface->get_scattering_sigmaL_solver();
  //NEMO_ASSERT(scattering_sigmaR_solver,tic_toc_prefix + "scattering_sigmaR_solver was NULL");
  PetscMatrixParallelComplex* off_sigmaR = NULL;
  PetscMatrixParallelComplex* off_sigmaL = NULL;
  if(scattering_sigmaR_solver!=NULL)
  {
    std::string temp_name;
    scattering_sigmaR_solver->get_data("writeable_Propagator",temp_name);
    scattering_sigmaR_solver->get_data(temp_name,&momentum_point, off_sigmaR,
        &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),&coupling_DOFmap);
  }
  if(scattering_sigmaL_solver!=NULL)
  {
    std::string temp_name;
    scattering_sigmaL_solver->get_data("writeable_Propagator",temp_name);
    scattering_sigmaL_solver->get_data(temp_name,&momentum_point, off_sigmaL,
        &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),&coupling_DOFmap);
  }


  //get gR(i-1,i-1)
  //get the retarded and advanced Green's functions
  if(retarded_G!=NULL)
  {
    //add t(i,i-1) = H(i,i-1) - sigmaR(i,i-1)
    if(off_sigmaR!=NULL)
    {
      //add sigma<(i,i-1)*gA(i-1,i-1)*t(i,i-1)' where t(i,i-1) includes both coupling H and sigmaR(i,i-1)
      GreensfunctionInterface* retarded_solver=dynamic_cast<GreensfunctionInterface*>(PropOptionInterface->get_exact_GR_solver());
      NEMO_ASSERT(retarded_solver!=NULL,error_prefix+"retarded_solver is NULL\n");
      PetscMatrixParallelComplex* temp_retarded_neighbor=NULL;
      Simulation* temp = PropOptionInterface->get_exact_GR_solver();
      PropagationOptionsInterface* temp_interface=dynamic_cast<PropagationOptionsInterface*>(temp);
      temp = temp_interface->get_combine_solver_for_storage();
      GreensfunctionInterface* temp_green_interface = dynamic_cast<GreensfunctionInterface*>(temp);
      if(PropOptionInterface->get_particle_type_is_Fermion())
      {
        temp_green_interface->get_Greensfunction(momentum_point,temp_retarded_neighbor,
            &coupling_DOFmap,&coupling_DOFmap,
            NemoPhys::Fermion_retarded_Green);
      }
      else
      {
        temp_green_interface->get_Greensfunction(momentum_point,temp_retarded_neighbor,
            &coupling_DOFmap,&coupling_DOFmap,
            NemoPhys::Boson_retarded_Green);
      }
      if(temp_retarded_neighbor->if_container())
        temp_retarded_neighbor->assemble();



      //get t(i,i-1) = H(i,i-1) - sigmaR(i,i-1)
      const Domain* target_domain=this_simulation->get_const_simulation_domain();
      const DOFmapInterface& target_DOFmap = Hamilton_Constructor->get_const_dof_map(target_domain);

      DOFmapInterface* large_coupling_DOFmap=NULL; //will store the DOFmap for the matrix of the target (i) domain and the coupling (i-1) domain


      PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;  //should become a matrix in the upper right corner
      PetscMatrixParallelComplex* temp_overlap = NULL;
      {
        std::vector<NemoMeshPoint> sorted_momentum;
        QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum,options,PropInterface->get_momentum_mesh_types(),Hamilton_Constructor);
        Hamilton_Constructor->get_data(std::string("Hamiltonian"), sorted_momentum, coupling_domain, coupling_Hamiltonian,large_coupling_DOFmap,target_domain);
        Hamilton_Constructor->get_data(std::string("overlap_matrix_coupling"),sorted_momentum,coupling_domain, temp_overlap, large_coupling_DOFmap,target_domain);
        delete large_coupling_DOFmap;

        if(temp_overlap!=NULL)
        {
          std::complex<double> energy = std::complex<double>(read_energy_from_momentum(this_simulation,momentum_point,output_Propagator),0.0);
          //cerr<<energy;
          *temp_overlap *= -energy; //-ES
          coupling_Hamiltonian->add_matrix(*temp_overlap, DIFFERENT_NONZERO_PATTERN, std::complex<double>(1.0, 0.0));
          //coupling_Hamiltonian->save_to_matlab_file("T_int.m");
        }
      }
      int start_own_large_rows;
      int end_own_large_rows_p1;
      coupling_Hamiltonian->get_ownership_range(start_own_large_rows,end_own_large_rows_p1);
      //NemoUtils::toc(tic_toc_HLLm1);

      //unsigned int number_of_couple_rows = coupling_Hamiltonian->get_num_rows();
      unsigned int number_of_couple_cols = coupling_Hamiltonian->get_num_cols();
      //unsigned int number_of_own_couple_rows = coupling_Hamiltonian->get_num_owned_rows();
      unsigned int number_of_target_rows = target_DOFmap.get_global_dof_number();
      unsigned int number_of_target_cols = number_of_target_rows;
      unsigned int number_of_own_target_rows = target_DOFmap.get_number_of_dofs();

      //extract sub block of coupling Hamiltonian to get H(i,i-1)
      //NemoUtils::tic(tic_toc_prefix + "get submatrix from coupling Hamiltonian");
      unsigned int number_of_sub_couple_rows = number_of_target_rows;
      unsigned int number_of_sub_couple_cols = number_of_couple_cols-number_of_target_cols;
      if(coupling_Hamiltonian->get_num_rows() <= number_of_target_rows)
      {
        number_of_target_rows = coupling_Hamiltonian->get_num_rows();
        number_of_sub_couple_rows = coupling_Hamiltonian->get_num_rows();//number_of_super_cols;
        number_of_sub_couple_cols = number_of_couple_cols;
        number_of_target_rows = number_of_sub_couple_rows;
        number_of_own_target_rows = number_of_sub_couple_rows;
        number_of_target_cols = 0;
      }
      std::vector<int> temp_rows(number_of_sub_couple_rows);
      std::vector<int> temp_cols(number_of_sub_couple_cols);
      for(unsigned int i=0; i<number_of_sub_couple_rows; i++)
        temp_rows[i]=i;
      for(unsigned int i=0; i<number_of_sub_couple_cols; i++)
        temp_cols[i]=i+number_of_target_cols;
      PetscMatrixParallelComplex* coupling = NULL;

      {
        coupling= new PetscMatrixParallelComplex(number_of_sub_couple_rows,number_of_sub_couple_cols,
            this_simulation->get_simulation_domain()->get_communicator());
        coupling->set_num_owned_rows(number_of_own_target_rows);
      }
      vector<int> rows_diagonal(number_of_sub_couple_rows,0);
      vector<int> rows_offdiagonal(number_of_sub_couple_rows,0);

      {
        for(unsigned int i=0; i<number_of_sub_couple_rows; i++)
        {
          rows_diagonal[i]=coupling_Hamiltonian->get_nz_diagonal(i);
          rows_offdiagonal[i]=coupling_Hamiltonian->get_nz_offdiagonal(i);
        }
        for(unsigned int i=0; i<number_of_sub_couple_rows; i++)
          coupling->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
        coupling_Hamiltonian->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,coupling);
        coupling->assemble();
      }

      delete coupling_Hamiltonian;
      coupling_Hamiltonian=NULL;


      coupling->add_matrix(*off_sigmaR, DIFFERENT_NONZERO_PATTERN, std::complex<double> (1.0,0.0));


      //add temp_matrix1 = t(i,i-1)*gR(i-1,i-1)*sigma<(i,i-1)
      //temp_retarded_neighbor
      PetscMatrixParallelComplex* temp_matrix1 = NULL;
      NEMO_ASSERT(off_sigmaL!=NULL,error_prefix + " off_sigmaL is NULL when off_sigmaR is not");

      PetscMatrixParallelComplex* temp_p1=NULL;
      PetscMatrixParallelComplex::mult(*coupling, *temp_retarded_neighbor, &temp_p1); //save temp_p1= t(i,i-1)*gR(i-1,i-1)for use in the almost c.c. term

      delete coupling;
      coupling = NULL;
      PetscMatrixParallelComplex::mult(*temp_p1, *off_sigmaL, &temp_matrix1); //t(i,i-1)*gR(i-1,i-1)*sigma<(i,i-1)

      summed_lesser_self->add_matrix(*temp_matrix1, DIFFERENT_NONZERO_PATTERN, std::complex<double> (1.0,0.0));
      delete temp_matrix1;
      temp_matrix1 = NULL;

      //add temp_matrix2 = sigma<(i,i-1)*gA(i-1,i-1)*t(i,i-1)^T
      PetscMatrixParallelComplex* temp_matrix2=NULL;
      PetscMatrixParallelComplex* temp_p1_transpose=new PetscMatrixParallelComplex(temp_p1->get_num_cols(),
          temp_p1->get_num_rows(),
          temp_p1->get_communicator());

      temp_p1->hermitian_transpose_matrix(*temp_p1_transpose,MAT_INITIAL_MATRIX);

      PetscMatrixParallelComplex::mult(*off_sigmaL, *temp_p1_transpose, &temp_matrix2);

      delete temp_p1_transpose;
      temp_p1_transpose = NULL;
      delete temp_p1;
      temp_p1 = NULL;
      summed_lesser_self->add_matrix(*temp_matrix2, DIFFERENT_NONZERO_PATTERN, std::complex<double> (1.0,0.0));
      delete temp_matrix2;
      temp_matrix2 = NULL;
    }

    if(PropOptionInterface->get_sigma_convert_dense())
      summed_lesser_self->matrix_convert_dense();
    GreensfunctionInterface* retarded_solver=dynamic_cast<GreensfunctionInterface*>(PropOptionInterface->get_exact_GR_solver());

    PetscMatrixParallelComplex* temp_retarded=NULL;
    if(PropOptionInterface->get_particle_type_is_Fermion())
      retarded_solver->get_Greensfunction(momentum_point,temp_retarded,
          &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),NULL,
          NemoPhys::Fermion_retarded_Green);
    else
      retarded_solver->get_Greensfunction(momentum_point,temp_retarded,
          &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),NULL,
          NemoPhys::Boson_retarded_Green);
    //retarded_solver->get_data(retarded_G->get_name(),&momentum_point,temp_retarded,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    if(temp_retarded->if_container())
      temp_retarded->assemble();

    if(!PropInterface->complex_energy_used())
    {
      //gR*summed_lesser_self
      if(advanced_G==NULL)
      {

        PetscMatrixParallelComplex* temp_p1 = NULL;
        PetscMatrixParallelComplex::mult(*temp_retarded, *summed_lesser_self, &temp_p1); //temp_p1=G^R*Sigma^<

        if(result!=NULL)
        {
          delete result;
          result=NULL;
        }
        PetscMatrixParallelComplex* temp_matrix2=new PetscMatrixParallelComplex(temp_retarded->get_num_cols(),
            temp_retarded->get_num_rows(),
            temp_retarded->get_communicator());

        temp_retarded->hermitian_transpose_matrix(*temp_matrix2,MAT_INITIAL_MATRIX);
        PetscMatrixParallelComplex::mult(*temp_p1,*temp_matrix2,&result); //result=G^R*Sigma^<*G^A

        delete temp_p1;
        temp_p1 = NULL;

        delete temp_matrix2;
        temp_matrix2 = NULL;
      }
    }
    else
    {
    }

  }

  ////deallocate the memory, once calculation is done
  if(summed_lesser_self!=NULL&&lesser_self.size()>0)
  {
    delete summed_lesser_self;
    summed_lesser_self=NULL;
  }

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::solve_correlation_Green_exact(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result,
    std::set<const Propagator*>& lesser_self,const Propagator* retarded_G, const Propagator* advanced_G)
{
  std::string tic_toc_prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_correlation_Green_exact ";
  NemoUtils::tic(tic_toc_prefix);

  std::string error_prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::solve_correlation_Green_exact ";

  InputOptions options = this_simulation->get_reference_to_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();

  double temperature1 = PropOptionInterface->get_temperature();
  double temperature_in_eV=temperature1*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;

  PetscMatrixParallelComplex* summed_lesser_self=NULL;
  PetscMatrixParallelComplex* temp_matrix=NULL;
  if(lesser_self.size()>0)
  {
    std::set<const Propagator*>::const_iterator lesser_it=lesser_self.begin();
    NEMO_ASSERT((*lesser_it)!=NULL,error_prefix+"self-energy pointer is NULL\n");
    std::string lesser_propagator_name=(*lesser_it)->get_name();

    Simulation* scatt_solver = dynamic_cast<PropagationOptionsBase*>(this_simulation)->get_scattering_sigmaL_solver();
    std::set<Simulation*> contact_solvers = dynamic_cast<PropagationOptionsBase*>(this_simulation)->get_contact_sigmaL_solvers();

    const DOFmapInterface& dofmap = Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain());

    std::set<Simulation*>::const_iterator contact_it;
    if (scatt_solver && (contact_solvers.size() > 0) )  //Both scattering and contact self energies exist
    {
      SelfenergyInterface* selfenergy_interface = dynamic_cast<SelfenergyInterface*>(scatt_solver);
      selfenergy_interface->get_Selfenergy(momentum_point, temp_matrix, &dofmap, &dofmap, NemoPhys::Fermion_lesser_self);
      //sum up all self-energies to a single matrix
      summed_lesser_self = new PetscMatrixParallelComplex(*temp_matrix);
      contact_it = contact_solvers.begin();
      Self_energy* contact_solver = dynamic_cast<Self_energy*>(*contact_it);
      contact_solver->get_Selfenergy(momentum_point, temp_matrix, &dofmap, &dofmap, NemoPhys::Fermion_lesser_self);
      summed_lesser_self->add_matrix(*temp_matrix, DIFFERENT_NONZERO_PATTERN);  // Add scattering and contact self energies
    }
    else  // Either scattering or contact self energies exist, not both
    {
      if (scatt_solver) // this_simulation->scattering_sigmaL_solver != NULL
      {
        SelfenergyInterface* selfenergy_interface = dynamic_cast<SelfenergyInterface*>(scatt_solver);
        selfenergy_interface->get_Selfenergy(momentum_point, temp_matrix, &dofmap, &dofmap, NemoPhys::Fermion_lesser_self);
      }

      if (contact_solvers.size() > 0) // this_simulation->contact_sigmaL_solvers is not empty
      {
        contact_it = contact_solvers.begin();
        Self_energy* contact_solver = dynamic_cast<Self_energy*>(*contact_it);
        contact_solver->get_Selfenergy(momentum_point, temp_matrix, &dofmap, &dofmap, NemoPhys::Fermion_lesser_self);
      }
      //sum up all self-energies to a single matrix
      summed_lesser_self = new PetscMatrixParallelComplex(*temp_matrix);
    }

    contact_it++;
    for(; contact_it!=contact_solvers.end(); contact_it++)  // Iterate through additional self energy solvers
    {
      if (contact_solvers.size() > 0) // this_simulation->contact_sigmaL_solvers is not empty
      {
        Self_energy* contact_solver = dynamic_cast<Self_energy*>(*contact_it);
        contact_solver->get_Selfenergy(momentum_point, temp_matrix, &dofmap, &dofmap, NemoPhys::Fermion_lesser_self);
      }

      NEMO_ASSERT(temp_matrix!=NULL,error_prefix+"received NULL for \""+lesser_propagator_name+"\"\n");
      summed_lesser_self->add_matrix(*temp_matrix, DIFFERENT_NONZERO_PATTERN); //note: to speed up, we might want to use the same matrix pattern

      if(PropOptionInterface->get_debug_output())
      {
        std::string temp_string;
        const std::vector<NemoMeshPoint>* temp_pointer=&momentum_point;
        translate_momentum_vector(this_simulation,temp_pointer, temp_string);
        temp_string+="_dsl_"+this_simulation->get_output_suffix();//options.get_option("output_suffix",std::string(""));
        temp_matrix->save_to_matlab_file(lesser_propagator_name+temp_string+".m");
      }
    }
    if(PropOptionInterface->get_sigma_convert_dense())
      summed_lesser_self->matrix_convert_dense();
  }
  else
    throw std::runtime_error(error_prefix+"no lesser self-energies received\n");

  //get the retarded and advanced Green's functions
  if(!PropInterface->complex_energy_used())
  {
    PetscMatrixParallelComplex* temp_retarded = NULL;
    PetscMatrixParallelComplex* temp_advanced = NULL;
    if(retarded_G!=NULL)
    {
      //Simulation* retarded_solver = pointer_to_Propagator_Constructors->find(retarded_G->get_name())->second;
      //GreensfunctionInterface* retarded_solver=find_source_of_Greensfunction(retarded_G->get_name());
      GreensfunctionInterface* retarded_solver=dynamic_cast<GreensfunctionInterface*>(PropOptionInterface->get_exact_GR_solver());
      NEMO_ASSERT(retarded_solver!=NULL,error_prefix+"retarded_solver is NULL\n");
      //Simulation* retarded_solver = find_source_of_data(retarded_G->get_name());

      if(PropOptionInterface->get_particle_type_is_Fermion())
        retarded_solver->get_Greensfunction(momentum_point,temp_retarded,
            &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),NULL,
            NemoPhys::Fermion_retarded_Green);
      else
        retarded_solver->get_Greensfunction(momentum_point,temp_retarded,
            &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),NULL,
            NemoPhys::Boson_retarded_Green);
      //retarded_solver->get_data(retarded_G->get_name(),&momentum_point,temp_retarded,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      if(temp_retarded->if_container())
        temp_retarded->assemble();
    }
    else if(advanced_G!=NULL)
    {
      //Simulation* advanced_solver = find_source_of_data(advanced_G->get_name());
      GreensfunctionInterface* advanced_solver=find_source_of_Greensfunction(this_simulation, advanced_G->get_name());
      if(PropOptionInterface->get_particle_type_is_Fermion())
        advanced_solver->get_Greensfunction(momentum_point,temp_advanced,
            &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),NULL,
            NemoPhys::Fermion_advanced_Green);
      else
        advanced_solver->get_Greensfunction(momentum_point,temp_advanced,
            &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())),NULL,
            NemoPhys::Boson_advanced_Green);
      //advanced_solver->get_data(advanced_G->get_name(),&momentum_point,temp_advanced,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      if(temp_advanced->if_container())
        temp_advanced->assemble();
    }

    //call core routine
    core_correlation_Green_exact(summed_lesser_self, result, temp_retarded, temp_advanced);

  }
  else //complex energy
  {
    PetscMatrixParallelComplex* temp_retarded = NULL;
    PetscMatrixParallelComplex* temp_advanced = NULL;
    if(retarded_G!=NULL)
    {
      //get GR
      GreensfunctionInterface* retarded_solver = dynamic_cast<GreensfunctionInterface*>(PropOptionInterface->get_exact_GR_solver());
      NEMO_ASSERT(retarded_solver != NULL, error_prefix + "retarded_solver is NULL\n");

      if (PropOptionInterface->get_particle_type_is_Fermion())
        retarded_solver->get_Greensfunction(momentum_point, temp_retarded,
          &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          NemoPhys::Fermion_retarded_Green);
      else
        retarded_solver->get_Greensfunction(momentum_point, temp_retarded,
          &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          NemoPhys::Boson_retarded_Green);
      if (temp_retarded->if_container())
        temp_retarded->assemble();

      std::complex<double> energy = read_complex_energy_from_momentum(this_simulation, momentum_point,output_Propagator);
      NEMO_ASSERT(options.check_option("drain_chemical_potential"), error_prefix + "drain chemical potential has do be specified for complex energy\n");
      double chemical_potential=options.get_option("drain_chemical_potential",0.0);
      bool isPole;
      //find the mesh constructor of the complex energy
      std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=PropInterface->get_momentum_mesh_types().begin();
      std::string energy_name=std::string("");
      for (; momentum_name_it!=PropInterface->get_momentum_mesh_types().end()&&energy_name==std::string(""); ++momentum_name_it)
        if(momentum_name_it->second==NemoPhys::Complex_energy)
          energy_name=momentum_name_it->first;
      std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
      std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(energy_name);
      NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),"Propagation(\""+this_simulation->get_name()+"\")::do_solve_lesser_equilibrium have not found constructor of mesh \""+energy_name+"\"\n");
      Simulation* mesh_constructor=temp_cit->second;
      mesh_constructor->get_data(energy, isPole);
      if(!isPole)
      {
        result = new PetscMatrixParallelComplex (*temp_retarded);
        *result *= (NemoMath::complex_fermi_distribution(chemical_potential,temperature_in_eV,energy)*std::complex<double>(-2.0,0.0));
      }
      else
      {
        result = new PetscMatrixParallelComplex (*temp_retarded);
        *result *= std::complex<double>(0.0,4.0*NemoMath::pi*NemoPhys::temperature*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge);
      }
    }
    else if(advanced_G!=NULL)
    {
      //get GA
      GreensfunctionInterface* advanced_solver = find_source_of_Greensfunction(this_simulation, advanced_G->get_name());
      if (PropOptionInterface->get_particle_type_is_Fermion())
        advanced_solver->get_Greensfunction(momentum_point, temp_advanced,
          &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          NemoPhys::Fermion_advanced_Green);
      else
        advanced_solver->get_Greensfunction(momentum_point, temp_advanced,
          &(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())), NULL,
          NemoPhys::Boson_advanced_Green);
      if (temp_advanced->if_container())
        temp_advanced->assemble();

      NEMO_ASSERT(options.check_option("drain_chemical_potential"), error_prefix + "drain chemical potential has do be specified for complex energy\n");
      double chemical_potential=options.get_option("drain_chemical_potential",0.0);
      temp_retarded=new PetscMatrixParallelComplex(temp_advanced->get_num_cols(),
          temp_advanced->get_num_rows(),
          temp_advanced->get_communicator()); //NULL;
      temp_advanced->hermitian_transpose_matrix(*temp_retarded,MAT_INITIAL_MATRIX);
      std::complex<double> energy = read_complex_energy_from_momentum(this_simulation, momentum_point,output_Propagator);
      bool isPole;
      //find the mesh constructor of the complex energy
      std::map<std::string, NemoPhys::Momentum_type> momentum_mesh_types = PropInterface->get_momentum_mesh_types();
      std::map<std::string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
      std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
      std::string energy_name=std::string("");
      for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
        if(momentum_name_it->second==NemoPhys::Complex_energy)
          energy_name=momentum_name_it->first;
      std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(energy_name);
      NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),"Propagation(\""+this_simulation->get_name()+"\")::do_solve_lesser_equilibrium have not found constructor of mesh \""+energy_name+"\"\n");
      Simulation* mesh_constructor=temp_cit->second;
      mesh_constructor->get_data(energy, isPole);
      if(!isPole)
      {
        result = new PetscMatrixParallelComplex (*temp_retarded);
        *result *= (NemoMath::complex_fermi_distribution(chemical_potential,temperature_in_eV,energy)*std::complex<double>(-2.0,0.0));
      }
      else
      {
        result = new PetscMatrixParallelComplex (*temp_retarded);
        *result *= std::complex<double>(0.0,4.0*NemoMath::pi*NemoPhys::temperature*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge);
      }
      delete temp_retarded;
      temp_retarded=NULL;
    }
    else
      throw std::invalid_argument(error_prefix+"at least either the retarded or the advanced Green's function must be given as input\n");
  }

  ////deallocate the memory, once calculation is done
  if(summed_lesser_self!=NULL&&lesser_self.size()>0)
  {
    delete summed_lesser_self;
    summed_lesser_self=NULL;
  }
  NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::core_correlation_Green_exact(PetscMatrixParallelComplex* summed_lesser_self,PetscMatrixParallelComplex*& result,
    PetscMatrixParallelComplex* retarded_green,PetscMatrixParallelComplex* advanced_green)
{
  NEMO_ASSERT(retarded_green!=NULL||advanced_green!=NULL,"PropagationUtilities core correlation green called with NULL GR and GA");
  NEMO_ASSERT(summed_lesser_self!=NULL,"PropagationUtilities core correlation green called with NULL correlation self");

  if(advanced_green == NULL)
  {
    advanced_green = new PetscMatrixParallelComplex(retarded_green->get_num_cols(),
        retarded_green->get_num_rows(),
        retarded_green->get_communicator());
    retarded_green->hermitian_transpose_matrix(*advanced_green,MAT_INITIAL_MATRIX);


    PetscMatrixParallelComplex* temp_p1=NULL;
    PetscMatrixParallelComplex::mult(*retarded_green, *summed_lesser_self, &temp_p1); //temp_p1=G^R*Sigma^<
    PetscMatrixParallelComplex::mult(*temp_p1,*advanced_green,&result); //result=G^R*Sigma^<*G^A
    delete temp_p1;
    temp_p1=NULL;
    delete advanced_green;
    advanced_green = NULL;

  }
  else if(retarded_green == NULL)
  {
    retarded_green =new PetscMatrixParallelComplex(advanced_green->get_num_cols(),
        advanced_green->get_num_rows(),
        advanced_green->get_communicator());
    advanced_green->hermitian_transpose_matrix(*retarded_green,MAT_INITIAL_MATRIX);
    PetscMatrixParallelComplex* temp_p1=NULL;
    PetscMatrixParallelComplex::mult(*retarded_green, *summed_lesser_self, &temp_p1); //temp_p1=G^R*Sigma^
    delete retarded_green;
    retarded_green = NULL;
    PetscMatrixParallelComplex::mult(*temp_p1,*advanced_green,&result); //result=G^R*Sigma^<*G^A
    delete temp_p1;
    temp_p1=NULL;

  }
  result->consider_as_full();
}



void PropagationUtilities::custom_matrix_product_RGF(Simulation* this_simulation, const int number_of_off_diagonals, PetscMatrixParallelComplex*& retarded_Green, PetscMatrixParallelComplex*& Gamma,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+this_simulation->get_name()+"\")::custom_matrix_product_RGF ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Greensolver(\""+this_simulation->get_name()+"\")::custom_matrix_product_RGF ";

  //0. start with sanity checks and deleting whatever had been in the result matrix
  NEMO_ASSERT(retarded_Green!=NULL, prefix+"have received NULL for the retarded Green's function\n");
  NEMO_ASSERT(Gamma!=NULL, prefix+"have received NULL for the Gamma matrix\n");

  if(result!=NULL)
    delete result;

  //1. perform loop over the requested number of off diagonals (only diagonal available for the moment)
  NEMO_ASSERT(number_of_off_diagonals==0,prefix+"called with more than the diagonal requested\n");

  //2. set up result matrix according to number_of_off_diagonals
  //2.1 create a new matrix
  result = new PetscMatrixParallelComplex(retarded_Green->get_num_rows(),retarded_Green->get_num_cols(),retarded_Green->get_communicator());
  //2.2 set the sparsity pattern
  result->set_num_owned_rows(retarded_Green->get_num_owned_rows());
  int start_own_rows;
  int end_own_rows;
  retarded_Green->get_ownership_range(start_own_rows,end_own_rows);
  for(int i=start_own_rows; i<end_own_rows; i++)
    result->set_num_nonzeros(i,1,0);
  result->allocate_memory();

  std::vector<std::complex<double> > diagonal_result(retarded_Green->get_num_owned_rows(),std::complex<double> (0.0,0.0));

  unsigned int number_of_gamma_rows = Gamma->get_num_rows();
  // figure out the start rows and end rows of nonzero block
  unsigned int start_row = 0;
  unsigned int end_row = 0;
  unsigned int nonzero_index = 0;
  for(unsigned int i=0; i<number_of_gamma_rows; i++)
  {
    nonzero_index = Gamma->get_nz_diagonal(i);
    if(nonzero_index>0)
    {
      start_row = i;
      break;
    }
  }
  nonzero_index = 0;
  for(unsigned int i=number_of_gamma_rows; i>0; i--)
  {
    nonzero_index = Gamma->get_nz_diagonal(i-1);
    if(nonzero_index>0)
    {
      end_row = i-1;
      break;
    }
  }
  unsigned int number_of_sub_rows = end_row+1-start_row;

  // get sub block of gamma
  std::vector<int> temp_rows(number_of_sub_rows);
  std::vector<int> temp_cols(number_of_sub_rows); //Yu: assuming gamma is square block
  for(unsigned int i=0; i<number_of_sub_rows; i++)
    temp_rows[i]=i+start_row;
  for(unsigned int i=0; i<number_of_sub_rows; i++)
    temp_cols[i]=i+start_row;
  PetscMatrixParallelComplex* sub_Gamma = NULL;
  sub_Gamma= new PetscMatrixParallelComplex(number_of_sub_rows,number_of_sub_rows,
      this_simulation->get_simulation_domain()->get_communicator() );
  //sub_Gamma->consider_as_full(); //Yu: dense block -- does not affect get_submatrix
  sub_Gamma->set_num_owned_rows(number_of_sub_rows);
  vector<int> rows_diagonal(number_of_sub_rows,0);
  vector<int> rows_offdiagonal(number_of_sub_rows,0);
  for(unsigned int i=0; i<number_of_sub_rows; i++)
  {
    rows_diagonal[i]=Gamma->get_nz_diagonal(i+start_row);
    rows_offdiagonal[i]=Gamma->get_nz_offdiagonal(i+start_row);
  }
  for(unsigned int i=0; i<number_of_sub_rows; i++)
    sub_Gamma->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
  NemoUtils::tic("extract sub_Gamma1");
  Gamma->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,sub_Gamma);
  sub_Gamma->assemble();
  NemoUtils::toc("extract sub_Gamma1");

  //turn sub_Gamma into a dense matrix
  //Yu: petsc does not provide a method for it, so we need to do it ourselves
  sub_Gamma->matrix_convert_dense();

  //================================
  //Yu: II extract submatrix of last block of GR
  ///-------------------------------
  unsigned int n_domains=number_of_gamma_rows/number_of_sub_rows;
  std::vector<int> temp_cols_G(number_of_sub_rows);
  for(unsigned int i=0; i<number_of_sub_rows; i++)
    temp_cols_G[i]=i+start_row;
  std::vector<int> temp_rows_G;

  PetscVectorNemo<std::complex<double> > sub_GR_vector(number_of_sub_rows,number_of_sub_rows,
      retarded_Green->get_communicator());
  PetscVectorNemo<std::complex<double> > temp_vector(number_of_sub_rows,number_of_sub_rows,
      retarded_Green->get_communicator());

  for(unsigned int id=0; id<n_domains; id++)
  {
    //extract last column block of GR
    unsigned int num_rows=0;
    if(id==n_domains-1)
      num_rows=number_of_gamma_rows-number_of_sub_rows*id;
    else
      num_rows=number_of_sub_rows;
    temp_rows_G.resize(num_rows);
    for(unsigned int i=0; i<num_rows; i++)
      temp_rows_G[i]=i+id*number_of_sub_rows;

    PetscMatrixParallelComplex* sub_GR = NULL;
    sub_GR= new PetscMatrixParallelComplex(num_rows,number_of_sub_rows,
                                           retarded_Green->get_communicator() );
    sub_GR->set_num_owned_rows(num_rows);
    for(unsigned int i=0; i<num_rows; i++)
      sub_GR->set_num_nonzeros_for_local_row(i,number_of_sub_rows,0);

    NemoUtils::tic("extract sub_GR1");
    retarded_Green->PetscMatrixParallelComplex::get_submatrix(temp_rows_G,temp_cols_G,MAT_INITIAL_MATRIX,sub_GR);
    sub_GR->assemble();
    NemoUtils::toc("extract sub_GR1");

    //turn sub_GR into a dense matrix
    //Yu: petsc does not provide a method for it, so we need to do it ourselves
    sub_GR->matrix_convert_dense();

    //matrix mult
    PetscMatrixParallelComplex* temp_matrix=NULL;
    stringstream ss;
    ss << sub_Gamma->get_num_rows();
    std::string tic_toc_mult = tic_toc_prefix+" GR x Gamma block size "+ss.str();
    NemoUtils::tic(tic_toc_mult);
    PetscMatrixParallelComplex::mult(*sub_GR,*sub_Gamma,&temp_matrix);
    NemoUtils::toc(tic_toc_mult);

    std::complex<double>* pointer_GR=NULL;
    std::complex<double>* pointer_temp=NULL;
    sub_GR->get_array(pointer_GR);
    temp_matrix->get_array(pointer_temp);
    //3. iterate over all matrix rows (i is result-row index)
    for(unsigned int i=0; i<num_rows; i++)
    {
      cplx* pointer_to_GR_vector=NULL;
      sub_GR_vector.get_array(pointer_to_GR_vector);
      for(unsigned int j=0; j<number_of_sub_rows; j++)
        pointer_to_GR_vector[j]=pointer_GR[j*num_rows+i];
      sub_GR_vector.store_array(pointer_to_GR_vector);

      cplx* pointer_to_temp_vector=NULL;
      temp_vector.get_array(pointer_to_temp_vector);
      for(unsigned int j=0; j<number_of_sub_rows; j++)
        pointer_to_temp_vector[j]=pointer_temp[j*num_rows+i];
      temp_vector.store_array(pointer_to_temp_vector);

      //dot_product
      std::complex<double> temp_result(0.0,0.0);
      PetscVectorNemo<std::complex<double> >::dot_product(temp_vector,sub_GR_vector,temp_result);

      //store the local result in the result matrix
      diagonal_result[i+id*number_of_sub_rows]=temp_result;
    }
    temp_matrix->store_array(pointer_temp);
    sub_GR->store_array(pointer_GR);
    delete temp_matrix;
    temp_matrix=NULL;
    delete sub_GR;
    sub_GR=NULL;
  }
  delete sub_Gamma;
  sub_Gamma=NULL;
  //----------------------
  //Yu: end of II
  //======================

  //4. store the diagonal in the result matrix
  PetscVectorNemo<std::complex<double> > petsc_vector_diagonal(retarded_Green->get_num_rows(),retarded_Green->get_num_rows(),retarded_Green->get_communicator());
  std::vector<int> indices(retarded_Green->get_num_rows(),0);
  for(unsigned int i=0; i<indices.size(); i++) indices[i]=i; //dense vector...
  petsc_vector_diagonal.set_values(indices,diagonal_result);
  result->matrix_diagonal_shift(petsc_vector_diagonal);
  //5. assemble the result matrix and exit
  result->assemble();
  NemoUtils::toc(tic_toc_prefix);
}

double PropagationUtilities::get_analytically_integrated_distribution(Simulation* this_simulation, const double energy, const double temperature_in_eV, const double chemical_potential,
    const std::string& momentum_type, const bool electron_type, const bool take_derivative_over_E, const bool take_derivative_over_mu)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+this_simulation->get_name()+"\")::get_analytically_integrated_distribution ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this_simulation->get_name()+"\")::get_analytically_integrated_distribution ";
  InputOptions options = this_simulation->get_reference_to_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();
  double effective_mass=0.0;
  std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
  Simulation* temp_simulation=this_simulation->find_simulation(temp_simulation_name);
  NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
  if(electron_type)
    temp_simulation->get_data("averaged_effective_mass_conduction_band",effective_mass);
  else
    temp_simulation->get_data("averaged_effective_mass_valence_band",effective_mass);

  const Domain* temp_domain=this_simulation->get_const_simulation_domain();
  double periodic_space_volume = temp_domain->return_periodic_space_volume();
  double result= get_analytically_integrated_distribution(energy,temperature_in_eV, chemical_potential,
                 momentum_type,electron_type, take_derivative_over_E, take_derivative_over_mu, effective_mass, periodic_space_volume);
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

double PropagationUtilities::get_analytically_integrated_distribution(const double energy, const double temperature_in_eV, const double chemical_potential,
    const std::string& momentum_type, const bool electron_type, const bool take_derivative_over_E, const bool take_derivative_over_mu,const double effective_mass, const double periodic_space_volume)
{
  //std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+this_simulation->get_name()+"\")::get_analytically_integrated_distribution ");
  std::string prefix = "PropagationUtilities::get_analytically_integrated_distribution ";
  NemoUtils::tic(prefix);

  //1. analyse the momentum type
  bool this_is_for_2D = true;
  if(momentum_type.find("1D")!=std::string::npos)
    this_is_for_2D = false;
  //2. average the effective mass in the domain of this solver
  //changed by Aryan as seem to have bug when using with QTBM::calculate_current
  //double me0=9.11E-31;
  //double effective_mass=0.05*me0;
  /***
  double effective_mass=0.0;
  std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
  Simulation* temp_simulation=find_simulation(temp_simulation_name);
  NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
  if(electron_type)
    temp_simulation->get_data("averaged_effective_mass_conduction_band",effective_mass);
  else
    temp_simulation->get_data("averaged_effective_mass_valence_band",effective_mass);
   ***/

  //3. perform the multiplications of Fermi integrals and prefactors
  NEMO_ASSERT(take_derivative_over_E==false || take_derivative_over_mu==false,prefix+"derivative can be taken on either E or mu, not both");
  double result = 1.0;
  double hbar=NemoPhys::planck_constant/NemoMath::pi/2.0;
  double kbT=NemoPhys::elementary_charge*temperature_in_eV;

  if(this_is_for_2D)
  {
    // calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
    double index_independent_prefactor=kbT/hbar/hbar/NemoMath::pi/2.0; //kB*T/(2*pi*hbar^2)
    if(electron_type)
    {
      // electrons
      double temp = (chemical_potential-energy)/temperature_in_eV; //(mu-E)/(kB*T)
      double fermi_integral=std::log(1.0+std::exp(temp)); //ln[1+exp((mu-E)/(kB*T))]
      if(take_derivative_over_E)
        fermi_integral=-std::exp(temp)/temperature_in_eV/(1.0+std::exp(temp));
      if(take_derivative_over_mu)
        fermi_integral=std::exp(temp)/temperature_in_eV/(1.0+std::exp(temp));
      result *= effective_mass*index_independent_prefactor*fermi_integral*NemoPhys::nm_in_m*NemoPhys::nm_in_m; //unit: 1/nm^2
    }
    else
    {
      //holes
      double temp = (energy-chemical_potential)/temperature_in_eV; //(E-mu)/(kB*T)
      double fermi_integral=std::log(1.0+std::exp(temp)); //ln[1+exp((E-mu)/(kB*T))]
      if(take_derivative_over_E)
        fermi_integral=std::exp(temp)/temperature_in_eV/(1.0+std::exp(temp));
      if(take_derivative_over_mu)
        fermi_integral=-std::exp(temp)/temperature_in_eV/(1.0+std::exp(temp));
      result *= effective_mass*index_independent_prefactor*fermi_integral*NemoPhys::nm_in_m*NemoPhys::nm_in_m; //unit: 1/nm^2
    }
    //if we want to get a density instead of the number of electrons (as needed for Poisson)
    //if (!this_simulation->get_reference_to_options().get_option("integration_to_get_density", true))
    {
      //const Domain* temp_domain=this_simulation->get_const_simulation_domain();
      result*=periodic_space_volume;
    }
  }
  else // 1D momentum
  {
    //throw std::runtime_error(prefix + "analytical_kspace_1D is not implemented\n");
    //to be multiplied by sqrt(me1z) to get 1D integrated (in kz) DOS
    double index_independent_prefactor=std::sqrt(kbT/NemoMath::pi/2.0/hbar/hbar); //1/hbar*sqrt(kB*T/(2*pi))
    //for debugging purpose
    //double fd_order=-0.5;
    double fermi_integral=0;
    double temp = 0;
    if(electron_type)
    {
      temp = (chemical_potential-energy)/temperature_in_eV; //(mu-E)/(kB*T)
      fermi_integral=NemoMath::fermi_int_m05(temp); //NemoMath::fermi_int(-0.5,temp);
      if(take_derivative_over_E) // dF-0.5/dE =dF-0.5/dtemp *dtemp/dE =F-1.5*(-1)/kB*T
      {
        fermi_integral=-NemoMath::fermi_int_m1_5(temp,temperature_in_eV)/temperature_in_eV ;// /kbT;  Junzhe on 2/27/15: should not devide by kT since the fermi_integral is unitless
        //fd_order=-1.5;
      }
      if(take_derivative_over_mu)
      {
        fermi_integral=NemoMath::fermi_int_m1_5(temp,temperature_in_eV)/temperature_in_eV ;// /kbT;
        //fd_order=-1.5;
      }
      result = std::sqrt(effective_mass)*index_independent_prefactor*fermi_integral*NemoPhys::nm_in_m; //unit: 1/nm
      //result = std::sqrt(effective_mass)*index_independent_prefactor*fermi_integral/NemoPhys::nm_in_m; //unit: 1/nm
    }
    else
    {
      //holes
      temp = (energy-chemical_potential)/temperature_in_eV; //(E-mu)/(kB*T)
      fermi_integral=NemoMath::fermi_int_m05(temp); //NemoMath::fermi_int(-0.5,temp);
      if(take_derivative_over_E) // dF-0.5/dE =dF-0.5/dtemp *dtemp/dE =F-1.5*(1)/kB*T
      {
        fermi_integral=NemoMath::fermi_int_m1_5(temp,temperature_in_eV)/temperature_in_eV; // /kbT;
        //fd_order=-1.5;
      }
      if(take_derivative_over_mu)
      {
        fermi_integral=-NemoMath::fermi_int_m1_5(temp,temperature_in_eV)/temperature_in_eV; // /kbT;
        //fd_order=-1.5;
      }
      result = std::sqrt(effective_mass)*index_independent_prefactor*fermi_integral*NemoPhys::nm_in_m; //unit: 1/nm
    }


    //if we want to get a density instead of the number of electrons (as needed for Poisson)
    //const Domain* temp_domain=this_simulation->get_const_simulation_domain();
    result*=periodic_space_volume;
    //result/=temp_domain->return_periodic_space_volume();
    /***
        NemoUtils::MsgLevel prev_level = msg.get_level();
        NemoUtils::msg.set_level(NemoUtils::MsgLevel(3));
        //msg <<"temp_domain->return_periodic_space_volume()*NemoPhys::nm_in_m = " << temp_domain->return_periodic_space_volume()*NemoPhys::nm_in_m << "\n";
        msg <<"energy chemical_potential temperature_in_eV = " << energy <<"\t" << chemical_potential <<"\t" << temperature_in_eV <<"\t" << "\n";
        msg <<"eta fd_order = " << temp <<"\t" << fd_order  << "\n";
        msg <<"electron_type fd(fd_order,eta) = " << electron_type <<"\t" << fermi_integral  << "\n";
        NemoUtils::msg.set_level(prev_level);
     ***/

  }

  NemoUtils::toc(prefix);
  return result;
}

void PropagationUtilities::supress_noise(Simulation* this_simulation,NemoMatrixInterface* in_out_matrix)
{
  PropagationOptionsInterface* temp_interface=get_PropagationOptionsInterface(this_simulation);
  if(temp_interface->get_use_matrix_0_threshold())
  {
    std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities for \""+this_simulation->get_name()+"\" supress_noise ");
    NemoUtils::tic(tic_toc_prefix);
    InputOptions input_options=this_simulation->get_options();
    double threshold= input_options.get_option("matrix_0_threshold",NemoMath::d_zero_tolerance);
    in_out_matrix->zero_small_values(threshold);
    NemoUtils::toc(tic_toc_prefix);
  }
}

void PropagationUtilities::supress_noise(NemoMatrixInterface* in_out_matrix, double threshold)
{
  if(threshold > 0)
  {
    std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities::supress_noise ");
    NemoUtils::tic(tic_toc_prefix);
    in_out_matrix->zero_small_values(threshold);
    NemoUtils::toc(tic_toc_prefix);
  }
}

void PropagationUtilities::symmetrize(Simulation* this_simulation,PetscMatrixParallelComplex*& input_matrix,NemoMath::symmetry_type& symm_type)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities for \""+this_simulation->get_name()+"\" symmetrize ");
  PropagationOptionsInterface* temp_interface=dynamic_cast<PropagationOptionsInterface*>(this_simulation);

  NEMO_ASSERT(temp_interface!=NULL,tic_toc_prefix+this_simulation->get_name()+" is not a PropagationOptionsInterface\n");
  if(symm_type==NemoMath::antihermitian)
  {
    PetscMatrixParallelComplex temp_result=PetscMatrixParallelComplex(input_matrix->get_num_cols(),
        input_matrix->get_num_rows(),
        input_matrix->get_communicator());
    input_matrix->hermitian_transpose_matrix(temp_result,MAT_INITIAL_MATRIX);
    input_matrix->add_matrix(temp_result, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));
    *input_matrix*=std::complex<double>(0.5,0.0);
  }
  else if(symm_type == NemoMath::symmetric)
  {
    PetscMatrixParallelComplex temp_result=PetscMatrixParallelComplex(input_matrix->get_num_cols(),
        input_matrix->get_num_rows(),
        input_matrix->get_communicator());
    input_matrix->transpose_matrix(temp_result,MAT_INITIAL_MATRIX);
    input_matrix->add_matrix(temp_result, DIFFERENT_NONZERO_PATTERN,std::complex<double> (1.0,0.0));
    *input_matrix*=std::complex<double>(0.5,0.0);

  }
  else
    throw std::invalid_argument("PropagationUtilies(\""+this_simulation->get_name()+"\")::symmetrize not implemented for this symmetry\n");
}

void PropagationUtilities::set_job_done_momentum_map(Simulation* this_simulation, 
                                          const std::string* Propagator_name, const std::vector<NemoMeshPoint>* momentum_point,
                                          const bool input_status)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities for \""+this_simulation->get_name()+"\" set_job_done_momentum_map ");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(NemoUtils::MsgLevel(4));
  PropagationOptionsInterface* temp_interface=get_PropagationOptionsInterface(this_simulation);
  bool input_status2 = temp_interface->get_always_ready() || input_status;
  if(Propagator_name==NULL)
  {
    PropagatorInterface* temp_Propagation=dynamic_cast<PropagatorInterface*>(this_simulation);
    NEMO_ASSERT(temp_Propagation!=NULL,tic_toc_prefix+this_simulation->get_name()+" is not a PropagatorInterface\n");
    Propagator* writeable_Propagator=NULL;
    temp_Propagation->get_Propagator(writeable_Propagator);
    if(writeable_Propagator!=NULL)
    {
      set_job_done_momentum_map(this_simulation,&(writeable_Propagator->get_name()),momentum_point,input_status2);
    }
  }
  else
  {
    std::map<std::vector<NemoMeshPoint>, bool> temp_map;
    std::map<std::vector<NemoMeshPoint>, bool>& job_done_momentum_map=temp_map;
    PropagatorInterface* temp_prop_interface=dynamic_cast<PropagatorInterface*>(this_simulation);
    NEMO_ASSERT(temp_prop_interface!=NULL,tic_toc_prefix+this_simulation->get_name()+" is not a PropagatorInterface\n");
    temp_prop_interface->get_job_done_momentum_map(job_done_momentum_map);
    {
      if(momentum_point!=NULL)
      {
        job_done_momentum_map[*momentum_point]=input_status2;
      }
      else
      {
        std::map<std::vector<NemoMeshPoint>, bool>::iterator it2=job_done_momentum_map.begin();
        for(; it2!=job_done_momentum_map.end(); it2++)
          it2->second=input_status2;
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

GreensfunctionInterface* PropagationUtilities::find_source_of_Greensfunction(Simulation* this_simulation,const std::string& inputname)
{
  GreensfunctionInterface* temp = dynamic_cast<GreensfunctionInterface*>(this_simulation->find_source_of_data(inputname));
  NEMO_ASSERT(temp!=NULL,"PropagationUtilities for \""+this_simulation->get_name()+"\" find_source_of_Greensfunction "+inputname+" is not Greensfunction compatible "+
              "(type:\""+find_source_of_data(this_simulation,inputname)->get_type()+"\"\n");
  return temp;
}

Simulation* PropagationUtilities::find_source_of_data(Simulation* this_simulation,const std::string& inputname)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::find_source_of_data2 ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::find_source_of_data: ";
  if(inputname.size()==0)
    NEMO_ASSERT(inputname.size()>0,prefix+"empty string received\n");
  Simulation* result=NULL;
  //1. if solver of inputname is given -> return this solver
  std::string variable = inputname+"_solver";
  const InputOptions& input_options=this_simulation->get_options();
  if(input_options.check_option(variable))
  {
    std::string solver_name=input_options.get_option(variable,std::string(""));
    result=this_simulation->find_simulation(solver_name);
    NEMO_ASSERT(result!=NULL,prefix+"simulation \""+solver_name+"\" not found\n");
  }
  else
  {
    //2. otherwise, return the constructor
    PropagatorInterface* temp_interface=get_PropagatorInterface(this_simulation);
    std::map<std::string,Simulation*>* pointer_to_Propagator_Constructors=temp_interface->list_Propagator_constructors();
    NEMO_ASSERT(pointer_to_Propagator_Constructors!=NULL,prefix+"received NULL for pointer_to_Propagator_Constructors\n");
    std::map<std::string,Simulation*>::const_iterator c_it=pointer_to_Propagator_Constructors->find(inputname);
    if(c_it==pointer_to_Propagator_Constructors->end())
    {
      std::map<std::string,Simulation*>::const_iterator c_it2=pointer_to_Propagator_Constructors->begin();
      std::string temp_name;
      NEMO_ASSERT(input_options.check_option(inputname+"_constructor"),prefix+"please define \""+inputname+"_constructor\"\n");
      temp_name=input_options.get_option(inputname+"_constructor",std::string(""));
      result=this_simulation->find_simulation(temp_name);
      NEMO_ASSERT(result!=NULL,prefix+"have not found simulation \""+temp_name
                  +"\" (constructor of \""+inputname+"\")\n");
    }
    else
      result=c_it->second;
  }
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

PropagationOptionsInterface* PropagationUtilities::get_PropagationOptionsInterface(Simulation* input_simulation)
{
  PropagationOptionsInterface* temp_interface=dynamic_cast<PropagationOptionsInterface*>(input_simulation);
  NEMO_ASSERT(temp_interface!=NULL,"PropagationUtilities "+input_simulation->get_name()+ " is not a PropagationOptionsInterface\n");
  return temp_interface;
}

PropagatorInterface* PropagationUtilities::get_PropagatorInterface(Simulation* input_simulation)
{
  PropagatorInterface* temp_interface=dynamic_cast<PropagatorInterface*>(input_simulation);
  NEMO_ASSERT(temp_interface!=NULL,"PropagationUtilities "+input_simulation->get_name()+ " is not a PropagatorInterface\n");
  return temp_interface;
}

void PropagationUtilities::translate_momentum_vector(Simulation* input_simulation, const std::vector<NemoMeshPoint>*& momentum_point, std::string& result, const bool for_filename)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+input_simulation->get_name()+"\")::translate_momentum_vector ");
  NemoUtils::tic(tic_toc_prefix);
  //construct the filename
  for (unsigned int i=0; i<momentum_point->size(); i++)
  {
    std::vector<double> temp_vector=(*momentum_point)[i].get_coords();
    std::ostringstream strs;
    for (unsigned int j=0; j<temp_vector.size(); j++)
    {
      if (temp_vector[j]>=0.0)
        if(for_filename)
          strs << fabs(temp_vector[j])<<"x";
        else
          strs << fabs(temp_vector[j])<<"\t";
      else if(for_filename)
        strs << "m"<<fabs(temp_vector[j])<<"x";
      else
        strs << "-"<<fabs(temp_vector[j])<<"\t";
    }
    std::string str = strs.str();
    //search and replace the "." with "c"
    std::string::iterator string_it=str.begin();
    for(; string_it!=str.end(); ++string_it)
    {
      if((*string_it)==*(std::string(".").c_str()))
        if(for_filename)
          str.replace(string_it,string_it+1,std::string("c").c_str());
    }
    result+=str;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::translate_momentum_vector(Simulation* input_simulation, const std::vector<double>* momentum_point, std::string& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+input_simulation->get_name()+"\")::translate_momentum_vector2 ");
  NemoUtils::tic(tic_toc_prefix);
  //construct the filename
  std::vector<double> temp_vector=*momentum_point;
  std::ostringstream strs;
  for (unsigned int j=0; j<temp_vector.size(); j++)
  {
    if (temp_vector[j]>=0.0)
      strs << fabs(temp_vector[j])<<"x";
    else
      strs << "m"<<fabs(temp_vector[j])<<"x";
  }
  std::string str = strs.str();
  //search and replace the "." with "c"
  std::string::iterator string_it=str.begin();
  for(; string_it!=str.end(); ++string_it)
  {
    if((*string_it)==*(std::string(".").c_str()))
      str.replace(string_it,string_it+1,std::string("c").c_str());
  }
  result+=str;
  NemoUtils::toc(tic_toc_prefix);
}

double PropagationUtilities::read_energy_from_momentum(Simulation* input_simulation, const std::vector<NemoMeshPoint>& momentum_point, const Propagator* input_Propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+input_simulation->get_name()+"\")::read_energy_from_momentum ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+input_simulation->get_name()+"\")::read_energy_from_momentum ";
  double energy;
  PropagatorInterface* PropInterface=get_PropagatorInterface(input_simulation);
  if(!PropInterface->complex_energy_used())
  {
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=PropInterface->get_momentum_mesh_types().begin();
    std::string energy_name=std::string("");
    for (; momentum_name_it!=PropInterface->get_momentum_mesh_types().end()&&energy_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Energy)
        energy_name=momentum_name_it->first;
    unsigned int energy_index=momentum_point.size()+1; //larger then the vector size to throw exception if energy index is not found
    NEMO_ASSERT(input_Propagator!=NULL,prefix+"received NULL for the input_Propagator\n");
    NEMO_ASSERT(energy_name.find("complex") == std::string::npos, prefix + "complex energy should use read_complex_energy_from_momentum()\n");
    for (unsigned int i=0; i<input_Propagator->momentum_mesh_names.size(); i++)
      if(input_Propagator->momentum_mesh_names[i]==energy_name)
        energy_index=i;
    NEMO_ASSERT(energy_index<momentum_point.size(),prefix+"have not found any energy in the momentum\n");
    const NemoMeshPoint& energy_point=momentum_point[energy_index];
    energy=energy_point.get_x();
  }
  else
  {
    std::complex<double> complex_energy;
    complex_energy = read_complex_energy_from_momentum(input_simulation, momentum_point, input_Propagator);
    NEMO_ASSERT(complex_energy.imag() == 0, tic_toc_prefix + " complex energy should use read_complex_energy_from_momentum()\n");
    energy = complex_energy.real();
  }
  NemoUtils::toc(tic_toc_prefix);
  return energy;
}

std::complex<double> PropagationUtilities::read_complex_energy_from_momentum(Simulation* input_simulation, const std::vector<NemoMeshPoint>& momentum_point, const Propagator* input_Propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+input_simulation->get_name()+"\")::read_complex_energy_from_momentum ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+input_simulation->get_name()+"\")::read_complex_energy_from_momentum ";
  std::complex<double> energy;
  PropagatorInterface* PropInterface=get_PropagatorInterface(input_simulation);
  if(PropInterface->complex_energy_used())
  {
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=PropInterface->get_momentum_mesh_types().begin();
    std::string energy_name=std::string("");
    for (; momentum_name_it!=PropInterface->get_momentum_mesh_types().end()&&energy_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Complex_energy)
        energy_name=momentum_name_it->first;
    unsigned int energy_index=momentum_point.size()+1; //larger then the vector size to throw exception if energy index is not found
    NEMO_ASSERT(input_Propagator!=NULL,prefix+"received NULL for the input_Propagator\n");
    for (unsigned int i=0; i<input_Propagator->momentum_mesh_names.size(); i++)
      if(input_Propagator->momentum_mesh_names[i]==energy_name)
        energy_index=i;
    NEMO_ASSERT(energy_index<momentum_point.size(),prefix+"have not found any complex energy in the momentum\n");
    const NemoMeshPoint& energy_point=momentum_point[energy_index];
    energy=std::complex<double>(energy_point.get_x(),energy_point.get_y());
  }
  else
    energy=std::complex<double> (read_energy_from_momentum(input_simulation,momentum_point,input_Propagator),0.0);
  NemoUtils::toc(tic_toc_prefix);
  return energy;
}

void PropagationUtilities::find_Hamiltonian_momenta(Simulation* input_simulation,const Propagator* input_Propagator, std::set<unsigned int>*& result)
{
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(input_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(input_simulation);
  const InputOptions& input_options=input_simulation->get_options();
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+input_simulation->get_name()+"\")::find_Hamiltonian_momenta ");
  NemoUtils::tic(tic_toc_prefix);
  std::string error_prefix="PropagationUtilities(\""+input_simulation->get_name()+"\")::find_Hamiltonian_momenta ";
  //find the names of the momenta that are constructed by the Hamiltonian_Constructor
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=PropInterface->get_momentum_mesh_types().begin();
  std::set<std::string> Hamilton_momentum_names;
  for (; momentum_name_it!=PropInterface->get_momentum_mesh_types().end(); ++momentum_name_it)
  {
    if(PropInterface->get_Mesh_Constructors().find(momentum_name_it->first)->second==PropOptionInterface->get_Hamilton_Constructor())
      Hamilton_momentum_names.insert(momentum_name_it->first);
  }

  //if Hamilton_momenta are defined in the input_deck, add them to the Hamilton_momentum_names list
  if(input_options.check_option("Hamilton_momenta"))
  {
    std::vector<std::string> list_of_extra_momenta;
    input_options.get_option("Hamilton_momenta",list_of_extra_momenta);
    for(unsigned int i=0; i<list_of_extra_momenta.size(); i++)
      Hamilton_momentum_names.insert(list_of_extra_momenta[i]);
  }

  //find the indices of the momenta that are constructed by the Hamiltonian_Constructor
  std::set<std::string>::const_iterator names_cit=Hamilton_momentum_names.begin();
  for (; names_cit!=Hamilton_momentum_names.end(); ++names_cit)
  {
    for (unsigned int i=0; i<input_Propagator->momentum_mesh_names.size(); i++)
    {
      if(input_Propagator->momentum_mesh_names[i].find(*names_cit)!=std::string::npos)
      {
        result->insert(i);
      }
    }
  }
  NEMO_ASSERT(result->size()==Hamilton_momentum_names.size(),
              error_prefix+"mismatch of found momenta constructed by \""
              +PropOptionInterface->get_Hamilton_Constructor()->get_name()+"\"\n");
  //so far only one Hamilton momentum allowed:
  NEMO_ASSERT(result->size()<=1,error_prefix+"only one Hamilton momentum allowed at the moment\n");
  //result should be NULL, if no momentum is constructed by the Hamilton_Constructor
  if(result->size()==0)
  {
    result = NULL;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::set_valley(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>&  momentum, Simulation* Hamilton_Constructor)
{
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::set_valley() ";
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  std::vector<std::string>::const_iterator it = output_Propagator->momentum_mesh_names.begin();
  for(unsigned int i = 0; it < output_Propagator->momentum_mesh_names.end(); it++,i++)
  {
    if (*it == "valley")
    {
      std::map<std::string, Simulation*> Mesh_Constructors=PropInterface->get_Mesh_Constructors();
      InputOptions& writeable_solver_options = Hamilton_Constructor->get_reference_to_options();
      vector<string> list_names;
      NEMO_ASSERT(Mesh_Constructors.find("valley") != Mesh_Constructors.end() ,prefix+"momentum id out of scope\n");
      Mesh_Constructors["valley"]->get_data("valleys", list_names);
      NEMO_ASSERT(i<momentum.size(),prefix+"momentum id out of scope\n");
      NEMO_ASSERT(momentum[i].get_idx()<(int)list_names.size(),prefix+"id_valley out of scope\n");
      string valley_str = list_names[momentum[i].get_idx()];
      writeable_solver_options.set_option("set_valley",valley_str);
      //broadcasting valley information to all constructors
      for(std::map<std::string,Simulation*>::iterator it = Mesh_Constructors.begin(); it!=Mesh_Constructors.end(); ++it)
      {
        Simulation* temp_simulation = it->second;
        if(it->first != "valley" && temp_simulation!=this_simulation)
        {
          InputOptions& writeable_solver_options2 = temp_simulation->get_reference_to_options();
          writeable_solver_options2.set_option("set_valley",valley_str);
        }
      }
    }
  }
}
void PropagationUtilities::extract_real_part(PetscMatrixParallelComplex* complex_matrix,
                                    PetscMatrixParallel<double>* real_matrix)
{
  unsigned int local_rows = complex_matrix->get_num_owned_rows();
  real_matrix->set_num_owned_rows(local_rows);

  int start_row, end_row;
  std::vector<int> columns;
  std::vector< std::complex <double > > data;
  complex_matrix->get_ownership_range(start_row, end_row);

  for (unsigned int i = 0; i < local_rows; i++)
  {

    const  int n_local_cols = complex_matrix->get_nz_diagonal(i);
    const  int  n_non_local_cols = complex_matrix->get_nz_offdiagonal(i);
    real_matrix->set_num_nonzeros_for_local_row(i, n_local_cols, n_non_local_cols);
  }
  real_matrix->allocate_memory();
  real_matrix->set_to_zero();
  for (unsigned int i=0; i<local_rows; i++)
  {
    complex_matrix->get_non_zeros_from_a_row(i+start_row, columns, data);
    int n1 = columns.size();
    std::vector<double> real_numbers(n1);
    for (int j=0; j < n1; j++)
      real_numbers[j] = std::real(data[j]);

    real_matrix->set(i+start_row,columns, real_numbers);
  }

  real_matrix->assemble();
}

void PropagationUtilities::transfer_matrix_get_energy(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum, std::complex<double>* energy)
{
  //obtain energy from momentum
  std::string tic_toc_name = this_simulation->get_name();
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::transfer_matrix_get_energy ";
  NemoUtils::tic(tic_toc_prefix);

  //PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  InputOptions options = this_simulation->get_reference_to_options();
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);

  //figure out which index of momentum is the energy
  bool complex_energy=false;
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=PropInterface->get_momentum_mesh_types().begin();
  std::string energy_name=std::string("");
  for (; momentum_name_it!=PropInterface->get_momentum_mesh_types().end()&&energy_name==std::string(""); ++momentum_name_it)
    if(momentum_name_it->second==NemoPhys::Energy)
      energy_name=momentum_name_it->first;
    else if (momentum_name_it->second==NemoPhys::Complex_energy)
    {
      energy_name=momentum_name_it->first;
      complex_energy=true;
    }
  unsigned int energy_index=0;
  for (unsigned int i=0; i<output_Propagator->momentum_mesh_names.size(); i++)
    if(output_Propagator->momentum_mesh_names[i].find(std::string("energy"))!=std::string::npos)
      energy_index=i;
  //-------------------------------------------------------
  NemoMeshPoint enery_point=momentum[energy_index];
  if(!complex_energy)
    *energy=std::complex<double> (enery_point.get_x(),0.0);
  else
    *energy=std::complex<double> (enery_point.get_x(),enery_point.get_y());
  msg<<"PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_leads() for energy"<<*energy<<std::endl;
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::transfer_matrix_get_self_energy(Simulation* this_simulation, PetscMatrixParallelComplex*& E_minus_H_matrix, PetscMatrixParallelComplex*& T_matrix,
    unsigned int* number_of_wave, unsigned int* /*number_of_vectors_size*/, PetscMatrixParallelComplex*& phase_factor,
    PetscMatrixParallelComplex*& Wave_matrix, PetscMatrixParallelComplex*& result_matrix,
    unsigned int number_of_sub_rows, unsigned int number_of_sub_cols)
{

  //PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  InputOptions options = this_simulation->get_reference_to_options();
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);

  //calculate self-energy with given modes 
  //result with subdomain size -- NEGF or QTBM that doesn't use smart dofmap
  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::transfer_matrix_get_self_energy ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_self_energy ";

  unsigned int number_of_device_rows = T_matrix->get_num_rows(); //coupling Hamiltonian matrix 
  unsigned int number_of_device_cols = number_of_device_rows;
  if(*number_of_wave==0) //if number of modes is 0, fake the result matrix with a number 0
  { 
    //result defined in subdomain size
    result_matrix = new PetscMatrixParallelComplex(number_of_sub_rows,number_of_sub_cols,
      this_simulation->get_simulation_domain()->get_communicator() );
    result_matrix->set_num_owned_rows(number_of_sub_rows);
    for(unsigned int i=0; i<number_of_sub_rows; i++)
      result_matrix->set_num_nonzeros(i,1,0);
    result_matrix->allocate_memory();
    result_matrix->set_to_zero();
    result_matrix->assemble();
  }
  else 
  {
    PetscMatrixParallelComplex WaveT(Wave_matrix->get_num_cols(),Wave_matrix->get_num_rows(),
      this_simulation->get_simulation_domain()->get_communicator() );
    PetscMatrixParallelComplex* GR_inv_temp1 = NULL;
    PetscMatrixParallelComplex* GR_inv1 = NULL;
    PetscMatrixParallelComplex* GR_inv_temp2 = NULL;
    PetscMatrixParallelComplex* GR_inv2 = NULL;

    std::string tic_toc_mult1 = tic_toc_prefix+": invg = VL'*(E-H)*VL+VL'*T*VL*expmikdelta";
    NemoUtils::tic(tic_toc_mult1);
    Wave_matrix->hermitian_transpose_matrix(WaveT,MAT_INITIAL_MATRIX);//VL'
    PetscMatrixParallelComplex::mult(*Wave_matrix,*phase_factor,&GR_inv2); //GR_inv2=VL*expmikdelta
    PetscMatrixParallelComplex::mult(*T_matrix,*GR_inv2,&GR_inv_temp2); //GR_inv_temp2=T*VL*expmikdelta //delete GR_inv2 after this line
    delete GR_inv2;
    PetscMatrixParallelComplex::mult(*E_minus_H_matrix,*Wave_matrix,&GR_inv_temp1); //GR_inv_temp1=(E-H)*VL
    GR_inv_temp1->add_matrix(*GR_inv_temp2,DIFFERENT_NONZERO_PATTERN); //GR_inv_temp1 = (E-H)*VL+T*VL*expmikdelta //delete GR_inv_temp2 after this line
    delete GR_inv_temp2;
    PetscMatrixParallelComplex::mult(WaveT,*GR_inv_temp1,
                                     &GR_inv1); //GR_inv1=GR_inv1=VL'*(E-H)*VL+VL'*T*VL*expmikdelta //delete GR_inv_temp1 after this line
    delete GR_inv_temp1;
    NemoUtils::toc(tic_toc_mult1);

    std::string tic_toc_VLmultT= tic_toc_prefix+": VL'*T01, T10*VL ";
    NemoUtils::tic(tic_toc_VLmultT);

    vector<int> T_rows_diagonal(number_of_sub_rows,0);
    vector<int> T_rows_offdiagonal(number_of_sub_rows,0);
    for(unsigned int i=0; i<number_of_sub_rows; i++)
    {
      T_rows_diagonal[i]=T_matrix->get_nz_diagonal(i);
      T_rows_offdiagonal[i]=T_matrix->get_nz_offdiagonal(i);
    }
    PetscMatrixParallelComplex* couple_temp = NULL;
    //get only the sub rows that are allocated -- this is important for efficient solution of the linear equation
    transfer_matrix_get_submatrix(this_simulation,number_of_sub_rows,number_of_device_cols,0,0,T_rows_diagonal,T_rows_offdiagonal,T_matrix,couple_temp);
    PetscMatrixParallelComplex* couplemultwave = NULL;
    PetscMatrixParallelComplex::mult(*couple_temp,*Wave_matrix,
                                     &couplemultwave); //T10*VL //Yu: petsc assumes sparse*dense = dense, even though it has tons of zeros!
    delete couple_temp;
    PetscMatrixParallelComplex wavemultcouple(couplemultwave->get_num_cols(),couplemultwave->get_num_rows(),
      this_simulation->get_simulation_domain()->get_communicator() );
    couplemultwave->hermitian_transpose_matrix(wavemultcouple,MAT_INITIAL_MATRIX);//(T10*VL)' = VL'*T01
    NemoUtils::toc(tic_toc_VLmultT);

    PetscMatrixParallelComplex* temp_self = NULL;

    if(*number_of_wave<=1) //if only 1 mode, it is a number, do the simple calculation
    {
      std::string tic_toc_mult2= tic_toc_prefix+": sigma = (T10*VL)*inv(GR_inv1)*V:'*T01 ";
      NemoUtils::tic(tic_toc_mult2);
      //obtain GR = inv(GR_inv1)
      PetscMatrixParallelComplex* GR_matrix = NULL;
      std::complex<double> GR_matrix_1(0.0,0.0);
      if(*number_of_wave==1)
        GR_matrix_1 = 1.0/(GR_inv1->get(0,0));
      GR_matrix = new PetscMatrixParallelComplex(1,1,this_simulation->get_simulation_domain()->get_communicator() );
      GR_matrix->set_num_owned_rows(1);
      GR_matrix->set_num_nonzeros(0,1,0);
      GR_matrix->allocate_memory();
      GR_matrix->set_to_zero();
      GR_matrix->set(0,0,GR_matrix_1);
      GR_matrix->assemble();

      PetscMatrixParallelComplex* temp1= NULL;
      PetscMatrixParallelComplex::mult(*couplemultwave,*GR_matrix,&temp1);
      PetscMatrixParallelComplex::mult(*temp1,wavemultcouple,&temp_self); //self-energy
      NemoUtils::toc(tic_toc_mult2);
      delete GR_matrix;
      delete temp1;
    }
    else
    {
      // ------------------------------------
      // solve a linear equation as OMEN did
      // ------------------------------------
      //set up the Linear solver

      PetscMatrixParallelComplex temp_solution(GR_inv1->get_num_rows(),wavemultcouple.get_num_cols(),
          this_simulation->get_simulation_domain()->get_communicator() );
      temp_solution.consider_as_full(); //Yu: petsc require solution as dense
      temp_solution.allocate_memory();

      //Yu: set A, B, X -- petsc require B and X as dense!
      {
        LinearSolverPetscComplex solver(*GR_inv1,wavemultcouple,&temp_solution);

        //prepare input options for the linear solver
        InputOptions options_for_linear_solver;
        string value("petsc"); //default petsc preconditioner
        string key("solver");
        options_for_linear_solver[key] = value;
        solver.set_options(options_for_linear_solver);

        std::string tic_toc_prefix_linear= tic_toc_prefix+": solve_linear_equation GR_inv1*X=VL'*T01";
        NemoUtils::tic(tic_toc_prefix_linear);
        solver.solve(); //GR_inv1*X=VL'*T01 solve X
        NemoUtils::toc(tic_toc_prefix_linear);
      }
      PetscMatrixParallelComplex* temp_G=NULL;
      temp_G=&temp_solution;

      //delete wavemultcouple_temp;
      std::string tic_toc_mult2= tic_toc_prefix+": sigma = T10*VL*X";
      NemoUtils::tic(tic_toc_mult2);
      PetscMatrixParallelComplex::mult(*couplemultwave,*temp_G,&temp_self); //sigma = T10*VL*X
      NemoUtils::toc(tic_toc_mult2);
    }

    //need to copy result to a sparse matrix, for following usage
    //the problem is petsc doesn't allow adding dense matrix to sparse matrix directly!
    std::string tic_toc_copy1= tic_toc_prefix+": copy result";
    NemoUtils::tic(tic_toc_copy1);
    vector<int> result_diagonal(number_of_sub_rows,0); //result stored in subdomain
    vector<int> result_offdiagonal(number_of_sub_rows,0);
    for(unsigned int i=0; i<number_of_sub_rows; i++)
    {
      {
        result_diagonal[i]=temp_self->get_nz_diagonal(i);
        result_offdiagonal[i]=temp_self->get_nz_offdiagonal(i);
      }
    }
    //initialize result matrix as sparse matrix
    transfer_matrix_initialize_temp_matrix(this_simulation,number_of_sub_rows,number_of_sub_rows,result_diagonal,result_offdiagonal,result_matrix);
    const std::complex<double>* pointer_to_data= NULL;
    vector<cplx> data_vector;
    vector<int> col_index;
    int n_nonzeros=0;
    const int* n_col_nums=NULL;
    for(unsigned int i=0; i<number_of_sub_rows; i++) //copy row-wise, this is most efficient way for copying elements to sparse matrix 
    {
      {
        temp_self->get_row(i,&n_nonzeros,n_col_nums,pointer_to_data);
        col_index.resize(n_nonzeros,0);
        data_vector.resize(n_nonzeros,cplx(0.0,0.0));
        for(int j=0; j<n_nonzeros; j++)
        {
          col_index[j]=n_col_nums[j];
          data_vector[j]=pointer_to_data[j];
        }
        result_matrix->set(i,col_index,data_vector);
        temp_self->store_row(i,&n_nonzeros,n_col_nums,pointer_to_data);
      }
    }
    result_matrix->assemble();
    NemoUtils::toc(tic_toc_copy1);
    delete GR_inv1;
    delete couplemultwave;
    delete temp_self;
  }
  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::transfer_matrix_get_new_self_energy(Simulation* this_simulation, PetscMatrixParallelComplex*& E_minus_H_matrix, PetscMatrixParallelComplex*& T_matrix,
    unsigned int* number_of_wave, unsigned int* /*number_of_vectors_size*/, PetscMatrixParallelComplex*& phase_factor,
    PetscMatrixParallelComplex*& Wave_matrix, PetscMatrixParallelComplex*& result_matrix,
    PetscMatrixParallelComplex*& coupling_device)
{
  //solve self-energy, result store with device size -- OMEN-QTBM or RGF
  //PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  InputOptions options = this_simulation->get_reference_to_options();
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);

  //solve self-energy, result store with device size -- OMEN-QTBM or RGF
  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::transfer_matrix_get_self_energy ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::transfer_matrix_get_self_energy ";

  unsigned int number_of_device_rows = coupling_device->get_num_rows(); //coupling of contact/device
  unsigned int number_of_device_cols = number_of_device_rows;
  if(*number_of_wave==0) //number of modes is 0, fake result with number 0
  {
    result_matrix = new PetscMatrixParallelComplex(number_of_device_rows,number_of_device_cols,
        this_simulation->get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
    result_matrix->set_num_owned_rows(number_of_device_rows);
    for(unsigned int i=0; i<number_of_device_rows; i++)
      result_matrix->set_num_nonzeros(i,1,0);
    result_matrix->allocate_memory();
    result_matrix->set_to_zero();
    result_matrix->assemble();
  }
  else
  {
    PetscMatrixParallelComplex Wave = (*Wave_matrix);
    PetscMatrixParallelComplex WaveT(Wave.get_num_cols(),Wave.get_num_rows(),
      this_simulation->get_simulation_domain()->get_communicator());
    PetscMatrixParallelComplex* GR_inv_temp1 = NULL;
    PetscMatrixParallelComplex* GR_inv1 = NULL;
    PetscMatrixParallelComplex* GR_inv_temp2 = NULL;
    PetscMatrixParallelComplex* GR_inv2 = NULL;

    std::string tic_toc_mult1 = tic_toc_prefix+": invg = VL'*(E-H)*VL+VL'*T*VL*expmikdelta";
    NemoUtils::tic(tic_toc_mult1);
    (&Wave)->hermitian_transpose_matrix(WaveT,MAT_INITIAL_MATRIX);//VL'
    PetscMatrixParallelComplex::mult(Wave,*phase_factor,&GR_inv2); //GR_inv2=VL*expmikdelta
    PetscMatrixParallelComplex::mult(*T_matrix,*GR_inv2,&GR_inv_temp2); //GR_inv_temp2=T*VL*expmikdelta
    PetscMatrixParallelComplex::mult(*E_minus_H_matrix,Wave,&GR_inv_temp1); //GR_inv_temp1=(E-H)*VL
    GR_inv_temp1->add_matrix(*GR_inv_temp2,DIFFERENT_NONZERO_PATTERN); //GR_inv_temp1 = (E-H)*VL+T*VL*expmikdelta
    PetscMatrixParallelComplex::mult(WaveT,*GR_inv_temp1,&GR_inv1); //GR_inv1=GR_inv1=VL'*(E-H)*VL+VL'*T*VL*expmikdelta
    NemoUtils::toc(tic_toc_mult1);

    PetscMatrixParallelComplex* GR_matrix_inv = NULL;
    //get the left upper nonzeros of GR_inv1
    vector<int> GR_rows_diagonal(*number_of_wave,0);
    vector<int> GR_rows_offdiagonal(*number_of_wave,0);
    for(unsigned int i=0; i<*number_of_wave; i++)
    {
      GR_rows_diagonal[i]=GR_inv1->get_nz_diagonal(i);
      GR_rows_offdiagonal[i]=GR_inv1->get_nz_offdiagonal(i);
    }
    transfer_matrix_get_full_submatrix(this_simulation,*number_of_wave,*number_of_wave,0,0,GR_rows_diagonal,GR_rows_offdiagonal,GR_inv1,GR_matrix_inv);

    std::string tic_toc_VLmultT= tic_toc_prefix+": VL'*T01, T10*VL ";
    NemoUtils::tic(tic_toc_VLmultT);
    unsigned int start_row = 0;
    unsigned int end_row = 0;
    unsigned int nonzero_index = 0;
    //extract the nonzero rows from coupling matrix -- this is important for efficient solution of the linear equation
    for(unsigned int i=0; i<number_of_device_rows; i++)
    {
      nonzero_index = coupling_device->get_nz_diagonal(i);
      if(nonzero_index>0)
      {
        start_row = i; //starting row of nonzeros
        break;
      }
    }
    nonzero_index = 0;
    for(unsigned int i=number_of_device_rows; i>0; i--)
    {
      nonzero_index = coupling_device->get_nz_diagonal(i-1);
      if(nonzero_index>0)
      {
        end_row = i-1; //end row of nonzeros
        break;
      }
    }
    unsigned int number_of_sub_rows = end_row+1-start_row; //operations only to those nonzero rows; this is important for efficiency
    vector<int> T_rows_diagonal(number_of_sub_rows,0);
    vector<int> T_rows_offdiagonal(number_of_sub_rows,0);
    for(unsigned int i=0; i<number_of_sub_rows; i++)
    {
      T_rows_diagonal[i]=coupling_device->get_nz_diagonal(i+start_row);
      T_rows_offdiagonal[i]=coupling_device->get_nz_offdiagonal(i+start_row);
    }
    PetscMatrixParallelComplex* couple_temp = NULL;
    unsigned int number_of_contact_cols = E_minus_H_matrix->get_num_cols();
    //extract the nonzero rows and put into couple_temp
    transfer_matrix_get_submatrix(this_simulation,number_of_sub_rows,number_of_contact_cols,start_row,0,T_rows_diagonal,T_rows_offdiagonal,coupling_device,couple_temp);
    PetscMatrixParallelComplex small_coupling_device(*couple_temp);
    delete couple_temp;

    PetscMatrixParallelComplex* couplemultwave = NULL;
    PetscMatrixParallelComplex::mult(small_coupling_device,Wave,
                                     &couplemultwave); //T10*VL //Yu: petsc assumes sparse*dense = dense, even though it has tons of zeros!
    PetscMatrixParallelComplex wavemultcouple(couplemultwave->get_num_cols(),couplemultwave->get_num_rows(),
        this_simulation->get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
    couplemultwave->hermitian_transpose_matrix(wavemultcouple,MAT_INITIAL_MATRIX);//(T10*VL)' = VL'*T01
    NemoUtils::toc(tic_toc_VLmultT);

    PetscMatrixParallelComplex* temp_self = NULL;

    if(*number_of_wave<=1) //if only 1 mode, use simple calculation
    {
      std::string tic_toc_mult2= tic_toc_prefix+": sigma = (T10*VL)*inv(GR_inv1)*V:'*T01 ";
      NemoUtils::tic(tic_toc_mult2);
      //obtain GR = inv(GR_inv1)
      PetscMatrixParallelComplex* GR_matrix = NULL;
      std::complex<double> GR_matrix_1(0.0,0.0);
      if(*number_of_wave==1)
        GR_matrix_1 = 1.0/(GR_matrix_inv->get(0,0));
      GR_matrix = new PetscMatrixParallelComplex(1,1,
        this_simulation->get_simulation_domain()->get_communicator());
      GR_matrix->set_num_owned_rows(1);
      GR_matrix->set_num_nonzeros(0,1,0);
      GR_matrix->allocate_memory();
      GR_matrix->set_to_zero();
      GR_matrix->set(0,0,GR_matrix_1);
      GR_matrix->assemble();

      PetscMatrixParallelComplex* temp1= NULL;
      PetscMatrixParallelComplex::mult(*couplemultwave,*GR_matrix,&temp1);
      PetscMatrixParallelComplex::mult(*temp1,wavemultcouple,&temp_self);
      NemoUtils::toc(tic_toc_mult2);
      delete GR_matrix;
      delete temp1;
    }
    else
    {
      // ------------------------------------
      // solve a linear equation as OMEN did
      // ------------------------------------
      //set up the Linear solver
      PetscMatrixParallelComplex temp_solution(GR_matrix_inv->get_num_rows(),wavemultcouple.get_num_cols(),
          this_simulation->get_simulation_domain()->get_communicator() );
      temp_solution.consider_as_full(); //Yu: petsc require solution as dense
      temp_solution.allocate_memory();

      //Yu: set A, B, X -- petsc require B and X as dense!
      LinearSolverPetscComplex solver(*GR_matrix_inv,wavemultcouple,&temp_solution);

      //prepare input options for the linear solver
      InputOptions options_for_linear_solver;
      string value("petsc");
      string key("solver");
      options_for_linear_solver[key] = value;
      solver.set_options(options_for_linear_solver);

      std::string tic_toc_prefix_linear= tic_toc_prefix+": solve_linear_equation GR_inv1*X=VL'*T01";
      NemoUtils::tic(tic_toc_prefix_linear);
      solver.solve(); // GR_inv1*X=VL'*T01 solve X
      NemoUtils::toc(tic_toc_prefix_linear);

      PetscMatrixParallelComplex* temp_G=NULL;
      temp_G=&temp_solution;
      std::string tic_toc_mult2= tic_toc_prefix+": sigma = T10*VL*X";
      NemoUtils::tic(tic_toc_mult2);
      PetscMatrixParallelComplex::mult(*couplemultwave,*temp_G,&temp_self); //sigma = T10*VL*X
      NemoUtils::toc(tic_toc_mult2);
    }
    
    //copy result to sparse matrix since petsc doesn't allow adding dense matrix to sparse matrix!
    std::string tic_toc_copy1= tic_toc_prefix+": copy result";
    NemoUtils::tic(tic_toc_copy1);
    vector<int> result_diagonal(number_of_device_rows,0);
    vector<int> result_offdiagonal(number_of_device_rows,0);
    for(unsigned int i=start_row; i<=end_row; i++)
    {
      result_diagonal[i]=temp_self->get_nz_diagonal(i-start_row);
      result_offdiagonal[i]=temp_self->get_nz_offdiagonal(i-start_row);
    }
    //the result matrix is defined in device dimension for QTBM requirement
    transfer_matrix_initialize_temp_matrix(this_simulation,number_of_device_rows,number_of_device_rows,result_diagonal,result_offdiagonal,
                                           result_matrix);
    const std::complex<double>* pointer_to_data= NULL;
    vector<cplx> data_vector;
    vector<int> col_index;
    int n_nonzeros=0;
    const int* n_col_nums=NULL;
    for(unsigned int i=start_row; i<=end_row; i++) //row-wise, only for those nonzero rows
    {
      temp_self->get_row(i-start_row,&n_nonzeros,n_col_nums,pointer_to_data);
      col_index.resize(n_nonzeros,0);
      data_vector.resize(n_nonzeros,cplx(0.0,0.0));
      for(int j=0; j<n_nonzeros; j++)
      {
        col_index[j]=n_col_nums[j]+start_row;
        data_vector[j]=pointer_to_data[j];
      }
      result_matrix->set(i,col_index,data_vector);
      temp_self->store_row(i-start_row,&n_nonzeros,n_col_nums,pointer_to_data);
    }
    result_matrix->assemble();
    NemoUtils::toc(tic_toc_copy1);
    delete GR_inv_temp1;
    delete GR_inv1;
    delete GR_inv_temp2;
    delete GR_inv2;
    delete GR_matrix_inv;
    delete couplemultwave;
    delete temp_self;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::exact_inversion(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& inverse_Green, PetscMatrixParallelComplex*& result)
{
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  delete result;
  result = NULL;
  std::string invert_solver_option;
  std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;
  if (PropOptionInterface->get_offload_solver_initialized() && PropOptionInterface->get_offload_solver()->offload)
  {
    c_it_momentum_offload_info = PropOptionInterface->get_offload_solver()->offloading_momentum_map.find(momentum_point);
    NEMO_ASSERT(c_it_momentum_offload_info != PropOptionInterface->get_offload_solver()->offloading_momentum_map.end(),
        this_simulation->get_name() + "have not found momentum in the offloading_momentum_map\n");

    if(c_it_momentum_offload_info->second.offload_to == ResourceUtils::MIC)
    {
      invert_solver_option = "MIC_offload";
    }
    else if (c_it_momentum_offload_info->second.offload_to == ResourceUtils::GPU)
    {
      invert_solver_option = std::string("magma");
    }
    else if (c_it_momentum_offload_info->second.offload_to == ResourceUtils::CPU)
    {
      invert_solver_option = std::string("lapack");
    }
  }
  else
  {
    invert_solver_option = this_simulation->get_options().get_option("invert_solver",std::string("petsc"));
  }
  std::string tic_toc_prefix = "PropagationUtilities::exact_inversion ";
  delete result;
  result = NULL;

  NEMO_ASSERT(inverse_Green!=NULL,tic_toc_prefix+"received NULL for the inverseGR matrix\n");
  if(invert_solver_option==std::string("lapack"))
  {
    NemoUtils::tic(tic_toc_prefix+"invertLapack");
    //use invertLapack
    inverse_Green->matrix_convert_dense();
    PetscMatrixParallelComplex::invertLapack(*inverse_Green, &result);
    NemoUtils::toc(tic_toc_prefix+"invertLapack");
  }
  else if (invert_solver_option==std::string("MIC_offload"))
  {
    NemoUtils::tic(tic_toc_prefix+"invertLapack");
    //use invertLapack
    inverse_Green->matrix_convert_dense();

    if (PropOptionInterface->get_offload_solver_initialized() 
      && PropOptionInterface->get_offload_solver()->offload)
    {
      NEMO_ASSERT(c_it_momentum_offload_info != PropOptionInterface->get_offload_solver()->offloading_momentum_map.end(),
          tic_toc_prefix + "have not found momentum in the offloading_momentum_map\n");

      if(c_it_momentum_offload_info->second.offload_to == ResourceUtils::MIC)
      {
        int coprocessor_num = c_it_momentum_offload_info->second.coprocessor_num; // = options.get_option("offload_to", std::string(""));
        PetscMatrixParallelComplex::invertLapack(*inverse_Green, &result, NemoMICUtils::get_mic_offload_type(coprocessor_num));
      }
      else
        throw std::runtime_error(tic_toc_prefix + " called using nonexistent offload_to option\n");
    }
    else
    {
      throw std::runtime_error(tic_toc_prefix + " offload solver not initialized\n");
    }
    NemoUtils::toc(tic_toc_prefix+"invertLapack");
  }
  else  if(invert_solver_option==std::string("magma"))
  {
#ifdef MAGMA_ENABLE
    inverse_Green->matrix_convert_dense();
    PetscMatrixParallelComplex::invertMAGMA(*inverse_Green, &result);
#else
    NEMO_EXCEPTION("Cannot use MAGMA library without compiling and linking with MAGMA!\n");
#endif

  }
  else  if(invert_solver_option==std::string("magmamic"))
  {
#ifdef MAGMAMIC_ENABLE
    inverse_Green->matrix_convert_dense();
    PetscMatrixParallelComplex::invertMAGMAMIC(*inverse_Green, &result);
#else
    NEMO_EXCEPTION("Cannot use MAGMAMIC library without compiling and linking with MAGMA!\n");
#endif

  }
  else
  {
    PetscMatrixParallelComplex::invert(*inverse_Green, &result);
  }
  NEMO_ASSERT(result!=NULL, "PropagationUtilities::do_solve_retarded: inversion failed\n");
  result->consider_as_full();
}

void PropagationUtilities::calculate_slab_resolved_current(Simulation* this_simulation, std::string propagator_name,
    Simulation* EOM_Constructor, Simulation* source_of_G, vector<double>& slab_resolved_current,
    vector< map<double, double > >& slab_energy_resolved_current)
{
  InputOptions options = this_simulation->get_reference_to_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);


  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::calculate_slab_resolved_current ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::calculate_slab_resolved_currents ";

  bool energy_current = options.get_option("energy_current",false);
  bool energy_resolved_current = options.get_option("energy_resolved_current",false);


  //1. get the subdomains
  std::vector<string> subdomain_names;
  EOM_Constructor->get_data("subdomains",subdomain_names);

  slab_resolved_current.resize(subdomain_names.size()-1);
  if(energy_resolved_current)
    slab_energy_resolved_current.resize(subdomain_names.size()-1);
  //std::vector<std::map<std::vector<NemoMeshPoint>,std::vector<double> > > diag_HG(subdomain_names.size()-1);

  BlockSchroedingerModule* temp_module=dynamic_cast<BlockSchroedingerModule*>(EOM_Constructor);
  NEMO_ASSERT(temp_module!=NULL, prefix+ "Hamilton_Constructor is not a BlockSchroedingerModule\n");
  //NEMO_ASSERT(eom_interface!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::get_device_Hamiltonian "+Hamilton_Constructor->get_name()+" is not an EOMMatrixInterface\n");
  Propagator* temp_propagator=NULL;
  this_simulation->get_data(propagator_name,temp_propagator);
  //2. iterate over the copied PropagatorMap of this Propagator
  Propagator::PropagatorMap& temp_map = temp_propagator->propagator_map;
  Propagator::PropagatorMap::iterator momentum_it=temp_map.begin();

  EOMMatrixInterface* eom_interface = dynamic_cast<EOMMatrixInterface*>(EOM_Constructor);
  //NEMO_ASSERT(eom_interface!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::get_device_Hamiltonian "+Hamilton_Constructor->get_name()+" is not an EOMMatrixInterface\n");
  /*std::vector<NemoMeshPoint> temp_vector;
  std::vector<NemoMeshPoint> sorted_momentum_point;
  QuantumNumberUtils::sort_quantum_number(momentum_it->first,sorted_momentum_point,options,PropInterface->get_momentum_mesh_types(),EOM_Constructor);
  PetscMatrixParallelComplex* Hamiltonian = NULL;
  eom_interface->get_EOMMatrix(temp_vector, NULL, NULL, PropOptionInterface->get_avoid_copying_hamiltonian(), Hamiltonian);
*/

  //delete Hamiltonian;
  //Hamiltonian = NULL;
  //2. loop through diagonal_indices
  for (unsigned int i = 0; i < subdomain_names.size()-1; ++i)
  {

//    Domain* neighbor_domain = Domain::get_domain(subdomain_names[i+1]);

    Propagator* temp_propagator=NULL;
    this_simulation->get_data(propagator_name,temp_propagator);
    //2. iterate over the copied PropagatorMap of this Propagator
    Propagator::PropagatorMap& temp_map = temp_propagator->propagator_map;
    Propagator::PropagatorMap::iterator momentum_it=temp_map.begin();

    std::map<std::vector<NemoMeshPoint>,std::vector<double> > diag_HG;

   // PetscMatrixParallelComplex* Matrix = NULL;
    //2.3 loop through all momenta
    for(; momentum_it!=temp_map.end(); ++momentum_it)
    {

      std::vector<NemoMeshPoint> sorted_momentum_point;
      QuantumNumberUtils::sort_quantum_number(momentum_it->first,sorted_momentum_point,options,PropInterface->get_momentum_mesh_types(),EOM_Constructor);
      PetscMatrixParallelComplex* Hamiltonian = NULL;
      DOFmapInterface* temp_pointer=NULL;
      eom_interface->get_EOMMatrix(sorted_momentum_point, NULL, NULL, PropOptionInterface->get_avoid_copying_hamiltonian(), Hamiltonian,temp_pointer);

      PetscMatrixParallelComplexContainer* Hamiltonian_container=dynamic_cast<PetscMatrixParallelComplexContainer*>(Hamiltonian);
      const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& H_blocks=Hamiltonian_container->get_const_container_blocks();

      std::vector<std::pair<std::vector<int>,std::vector<int> > > diagonal_indices;
      std::vector<std::pair<std::vector<int>,std::vector<int> > > offdiagonal_indices;
      temp_module->get_ordered_Block_keys(diagonal_indices, offdiagonal_indices);

      //2.3.1 call get_data to get G(E, k) and extract for this subdomain
      PetscMatrixParallelComplex* G_matrix = NULL;
      source_of_G->get_data(propagator_name, &momentum_it->first, G_matrix);
      //PetscMatrixParallelComplexContainer* GL_container=dynamic_cast<PetscMatrixParallelComplexContainer*>(G_matrix);
      //NEMO_ASSERT(GL_container!=NULL, prefix + " GL is not a container ");
      //const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& GL_blocks=GL_container->get_const_container_blocks();

      //G_matrix->assemble();
      //G_matrix->save_to_matlab_file(this_simulation->get_name()+propagator_name+".m");
      PetscMatrixParallelComplex* G_matrix_ij = new PetscMatrixParallelComplex(offdiagonal_indices[i].first.size(),offdiagonal_indices[i].second.size(),
            this_simulation->get_simulation_domain()->get_communicator());
        //G_matrix_ij->set_num_owned_rows(offdiagonal_indices[i].first.size());
        //for(unsigned int ii=0; ii<offdiagonal_indices[i].first.size(); ii++)
        //  G_matrix_ij->set_num_nonzeros_for_local_row(ii,offdiagonal_indices[i].second.size(),0);
        G_matrix_ij->consider_as_full();
        G_matrix_ij->allocate_memory();
        G_matrix_ij->set_to_zero();

      //G_matrix_ij->assemble();
      //G_matrix_ij->save_to_matlab_file("Gij"+subdomain_names[i]+".m");
      //2.3.2 get full coupling_H H(i,i+1)
      PetscMatrixParallelComplex* coupling_Hamiltonian = NULL;
      std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator off_block_cit=H_blocks.find(offdiagonal_indices[i]);
      coupling_Hamiltonian = off_block_cit->second;
      //full_coupling_Hamiltonian->save_to_matlab_file("coupling_"+subdomain_names[i]+".m");

      G_matrix->get_submatrix(offdiagonal_indices[i].first,offdiagonal_indices[i].second, MAT_INITIAL_MATRIX, G_matrix_ij);

      //2.3.4 multiply H*G<(i,i,E,k)
      PetscMatrixParallelComplex* HG = NULL;
      PetscMatrixParallelComplex transpose_coupling(coupling_Hamiltonian->get_num_cols(),coupling_Hamiltonian->get_num_rows(),
            coupling_Hamiltonian->get_communicator());
      coupling_Hamiltonian->matrix_convert_dense();
        coupling_Hamiltonian->hermitian_transpose_matrix(transpose_coupling, MAT_INITIAL_MATRIX); //t'
      PetscMatrixParallelComplex::mult(transpose_coupling, *G_matrix_ij, &HG); //t(i,i-1)*gR(i-1,i-1)*sigma<(i,i-1)

      //2.3.5 extract diagonal and store for integrate_vector
      std::vector<cplx> temp_diag_HG;
      HG->get_diagonal(&temp_diag_HG);
      std::vector<double> temp_diag_HG_real(temp_diag_HG.size());
      for(unsigned int HG_idx = 0; HG_idx < temp_diag_HG.size(); ++HG_idx)
        temp_diag_HG_real[HG_idx] = temp_diag_HG[HG_idx].real();
      //std::map<std::vector<NemoMeshPoint>,std::vector<double> >::iterator map_it = diag_HG[i].find(momentum_it->first);
      diag_HG[momentum_it->first] = temp_diag_HG_real;
      delete HG;
      HG = NULL;
      //delete coupling_Hamiltonian;
      //coupling_Hamiltonian = NULL;
      //if(!use_transpose)
      {
        delete G_matrix_ij;
        G_matrix_ij = NULL;
      }
      delete Hamiltonian;
      Hamiltonian = NULL;
    }
    //2.4 call integrate_vector and sum
    //void Propagation::integrate_vector(const std::map<std::vector<NemoMeshPoint>,std::vector<double> >& input_data,std::vector<double>& result)
    std::vector<double> temp_current;
    PropagationUtilities::integrate_vector(this_simulation,diag_HG,temp_current,energy_current);
    unsigned int vector_size = (unsigned int )temp_current.size();
//    int temp_double = 0;
    for (unsigned int idx = 0 ; idx < vector_size; ++idx)
      slab_resolved_current[i] += temp_current[idx];
    //2.5 if requested call integrate_vector for energy resolved
    //std::vector<std::complex<double> >*>*& diag_HG_pointer;
    std::map<double,std::vector<double > > temp_energy_resolved_current;
    std::map<double, std::vector<double > >* temp_energy_resolved_current_pointer = &temp_energy_resolved_current;
    std::map<std::vector<NemoMeshPoint>, std::vector<double > >* diag_HG_pointer = &diag_HG;
    if(energy_resolved_current)
      PropagationUtilities::integrate_vector_for_energy_resolved(this_simulation, diag_HG_pointer, temp_energy_resolved_current_pointer, energy_current);



    //2.6 multiply by prefactor and store into result vector
    double prefactor = 1.0;
    std::string momentum_name = std::string("");
    for (unsigned int ii = 0; ii < temp_propagator->momentum_mesh_names.size(); ii++)
    {
      if (temp_propagator->momentum_mesh_names[ii].find("energy") == std::string::npos)
      {
        momentum_name = temp_propagator->momentum_mesh_names[ii];

        if(momentum_name.find("momentum_1D")!=std::string::npos)
          prefactor/=2.0*NemoMath::pi;
        if(momentum_name.find("momentum_2D")!=std::string::npos)
          prefactor/=4.0*NemoMath::pi*NemoMath::pi;
        else if(momentum_name.find("momentum_3D")!=std::string::npos)
          prefactor/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
      }
    }
    slab_resolved_current[i] *= 2*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::h*prefactor;
    if(energy_resolved_current)
    {
      std::map<double,std::vector<double > >::iterator map_it = temp_energy_resolved_current.begin();
      std::map<double, double > temp_energy_resolved_current_real;
      for (; map_it != temp_energy_resolved_current.end(); ++map_it)
      {
    	  unsigned int  size = (unsigned int ) map_it->second.size();
        double temp_current = 0.0;
        for(unsigned int idx = 0; idx < size; idx++)
          temp_current += map_it->second[idx]*2*NemoPhys::elementary_charge/NemoPhys::h*prefactor;
        slab_energy_resolved_current[i].insert(std::make_pair<double,double>(map_it->first,temp_current));
      }
    }
    //2.7 if requested also store the energy resolved vector

  }
  NemoUtils::toc(tic_toc_prefix);


}

void PropagationUtilities::integrate_submatrix(Simulation* this_simulation, const std::set<std::pair<int,int> >& set_of_row_col_indices,
                                      const std::set<std::vector<NemoMeshPoint> >* local_integration_range,
                                      Simulation* source_of_data, NemoPhys::Propagator_type propagator_type, const MPI_Comm& integration_comm, const int root_rank,
                                      std::vector<int>& group,
                                      std::vector<std::complex<double> >*& result, const std::set<std::string>& exclude_integration,
                                      bool is_diagonal_only, const std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>* >* pointer_to_weight_function,
                                      std::map<std::set<std::vector<NemoMeshPoint> > , std::vector<std::complex<double> > >* local_integral_data,
                                      bool save_local_sigma_v2, bool call_prefactor, const std::vector<NemoMeshPoint> * initial_momentum )
{
  PropagatorInterface* PropInterface = get_PropagatorInterface(this_simulation);
  std::string prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_submatrix: ";

  NemoUtils::tic(prefix);

  //1. we assume that data_name is the name of a propagator; use get_data to get a pointer to the Propagator
  Propagator* Propagator_pointer = NULL;
  PropInterface->get_Propagator(Propagator_pointer);

  const Domain* const_simulation_domain = this_simulation->get_const_simulation_domain();

  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();

  Propagation* Propagation_pointer = NULL;
  Propagation_pointer = dynamic_cast<Propagation*>(this_simulation);
  NEMO_ASSERT(Propagation_pointer, prefix + "Simulation " + this_simulation->get_name() + " could not be cast to type Propagation");

  std::map<std::string, Simulation*> Mesh_Constructors = Propagation_pointer->get_Mesh_Constructors();

  std::vector<std::complex<double> > summed_submatrix(set_of_row_col_indices.size(),std::complex<double>(0.0,0.0));

  bool use_old_data=false;
  if(local_integral_data!=NULL)
  {
    std::map<std::set<std::vector<NemoMeshPoint> > ,std::vector<std::complex<double> > >::const_iterator save_cit=local_integral_data->find(
          *local_integration_range);
    if(save_cit!=local_integral_data->end())
    {
      NEMO_ASSERT(save_cit->second.size()==summed_submatrix.size(),prefix+"found inconsistent precalculated data\n");
      summed_submatrix=save_cit->second;
      use_old_data=true;

    }
  }
  //2. loop over all local momenta that we integrate over
  if(local_integration_range!=NULL&&!use_old_data)
  {
    std::set<std::vector<NemoMeshPoint> >::iterator integration_it=local_integration_range->begin();
    for(; integration_it!=local_integration_range->end(); integration_it++)
    {
      //2.1 get the respective source matrix for this momentum
      PetscMatrixParallelComplex* source_matrix = NULL;
      GreensfunctionInterface* green_source=dynamic_cast<GreensfunctionInterface*>(source_of_data);
      NEMO_ASSERT(green_source!=NULL, prefix + source_of_data->get_name() + "is not a GreensfunctionInterface\n");
      green_source->get_Greensfunction((const std::vector<NemoMeshPoint>)(*integration_it), source_matrix, &(Hamilton_Constructor->get_const_dof_map(const_simulation_domain)),
          &(Hamilton_Constructor->get_const_dof_map(const_simulation_domain)), propagator_type);

      //2.2 translate the submatrix into vectors
      //2.2.1 get the matrix elements due to the set_of_row_col_indices
      if(source_matrix->if_container()&&!is_diagonal_only)
        source_matrix->assemble();
      //2.3 translate the submatrix into vectors
      std::vector<std::complex<double> > submatrix(set_of_row_col_indices.size(),std::complex<double>(0.0,0.0));
      if (is_diagonal_only)
        source_matrix->get_diagonal(&submatrix);
      else if(pointer_to_weight_function!=NULL)
      {
        if(pointer_to_weight_function->size()==1)
        {
          //2.3.1 get the matrix elements due to the set_of_row_col_indices
          std::set<std::pair<int, int> >::const_iterator index_cit = set_of_row_col_indices.begin();
          int counter = 0;

          for (; index_cit != set_of_row_col_indices.end(); index_cit++)
          {
            submatrix[counter] =  source_matrix->get((*index_cit).first,(*index_cit).second);
            counter++;
          }
        }
        else
        {
          NEMO_ASSERT(set_of_row_col_indices.size()==pointer_to_weight_function->size(),prefix+"inconsistent size of weight function and matrix-index map\n");
          //get the required submatrix and multiply each element according to pointer_to_weight_function
          std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>* >::const_iterator weight_cit=pointer_to_weight_function->begin();
          int counter = 0;
          for(;weight_cit!=pointer_to_weight_function->end();++weight_cit)
          {
            std::pair<int,int> this_row_and_col=weight_cit->first;
            std::map<std::vector<NemoMeshPoint>,double>::const_iterator for_this_momentum_cit=weight_cit->second->find(*integration_it);
            NEMO_ASSERT(for_this_momentum_cit!=weight_cit->second->end(),prefix+"have not found the momentum in the weight-map\n");
            submatrix[counter] =  source_matrix->get(this_row_and_col.first,this_row_and_col.second)*for_this_momentum_cit->second;
            counter++;
          }
        }
      }
      else
      {
        std::set<std::pair<int, int> >::const_iterator index_cit = set_of_row_col_indices.begin();
        int counter = 0;

        for (; index_cit != set_of_row_col_indices.end(); index_cit++)
        {

          submatrix[counter] =  source_matrix->get((*index_cit).first,(*index_cit).second);
          counter++;
        }
      }
      if(call_prefactor == true)
      {
        // int counter = 0;
         //2.3.1 get the matrix elements due to the set_of_row_col_indices
         std::set<std::pair<int, int> >::const_iterator index_cit = set_of_row_col_indices.begin();
         for (unsigned int counter =0; counter<submatrix.size(); counter++,index_cit++)
         {
           submatrix[counter] *= Propagation_pointer->polar_optical_prefactor(*initial_momentum,*integration_it,(*index_cit).first,(*index_cit).second);
         }
      }
      /*
            double tolerance = 1E-10; //this is the tolerance for whether to keep a matrix element or not

            //filter out small matrix elements
            int num_elements = submatrix.size();
            for(int i = 0; i < num_elements; i++)
            {
              cplx tmp = submatrix[i];
              if( (std::abs(tmp.real()) < tolerance) && (std::abs(tmp.imag()) < tolerance) )
                submatrix[i] = cplx(0.0,0.0);
              else if ( (std::abs(tmp.real()) < tolerance) && (std::abs(tmp.imag()) > tolerance))
                submatrix[i] = cplx(0.0,tmp.imag());
              else if ( (std::abs(tmp.real()) > tolerance) && (std::abs(tmp.imag()) < tolerance))
                submatrix[i] = cplx(tmp.real(),0.0);
              else
                submatrix[i] = cplx(tmp.real(), tmp.imag());
            }
       */
      //2.4 multiply the vectors with the integration weight
      double integration_weight = 1.0;
      //get the constructor of this meshpoint
      for(unsigned int i=0; i<(*integration_it).size(); i++)
      {
        std::string momentum_mesh_name=Propagator_pointer->momentum_mesh_names[i];
        std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
        NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),"Propagation::(\""+this_simulation->get_name()
                    +"\")::integrate_diagonal: have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
        //Simulation* mesh_constructor=temp_cit->second;
        //ask the constructor for the weighting of this point
        //if momentum_mesh_name is not in the set of meshes to exclude from integration
        std::set<std::string>::const_iterator exclude_cit=exclude_integration.find(momentum_mesh_name);
        double temp_double=1.0;
        if(exclude_cit==exclude_integration.end())
          temp_double=Propagation_pointer->find_integration_weight(momentum_mesh_name,(*integration_it),Propagator_pointer);

        //if the integrand weight function exists, multiply with the appropriate value
        /*if(pointer_to_weight_function!=NULL)
        {
          std::map<std::vector<NemoMeshPoint>, double>::const_iterator weight_cit=pointer_to_weight_function->find(*integration_it);
          if(weight_cit!=pointer_to_weight_function->end())
            temp_double*=weight_cit->second;
        }*/
        integration_weight*=temp_double;

      }
      if(pointer_to_weight_function!=NULL)
      {
        if(pointer_to_weight_function->size()==1)
        {
          const std::map<std::vector<NemoMeshPoint>, double>* pointer_to_local_weight_function=(pointer_to_weight_function->begin())->second;
          std::map<std::vector<NemoMeshPoint>, double>::const_iterator weight_cit=pointer_to_local_weight_function->find(*integration_it);
          NEMO_ASSERT(weight_cit!=pointer_to_local_weight_function->end(),"Propagation::(\""+this_simulation->get_name()
                      +"\")::integrate_diagonal: have not found specific momentum\n");
          integration_weight*=weight_cit->second;
        }
        //else: multiplication of the weights is done for this scenario above already
      }
      //2.5.integrate over the locally stored submatrices
      for(unsigned int i=0; i<summed_submatrix.size(); i++)
      {
        summed_submatrix[i]+=integration_weight*submatrix[i];      }
    }
    if(save_local_sigma_v2 == true)
    {
      pair<set<std::vector<NemoMeshPoint> > ,std::vector<std::complex<double> > > local_int_entry(*local_integration_range,summed_submatrix);
      local_integral_data->insert(local_int_entry );
    }
  }// end of 2.
  //3. MPI_Reduce the summed_submatrix
  result->clear();
  //3.1 prepare memory if this MPI-process is the root
  int my_integration_rank;
  MPI_Comm_rank(integration_comm,&my_integration_rank);

  InputOptions options = this_simulation->get_reference_to_options();
  std::string type_of_MPI_reduce=options.get_option("type_of_MPI_reduce",std::string("synchronous_split"));
  if(type_of_MPI_reduce=="asynchronous_split_less")
  {
    Propagation_pointer->MPI_NEMO_Reduce(&summed_submatrix,result,summed_submatrix.size(), MPI_DOUBLE_COMPLEX,integration_comm,root_rank,  group);
  }
  else if(type_of_MPI_reduce=="asynchronous_split_less_tree")
  {
    Propagation_pointer->MPI_NEMO_Reduce_Tree(&summed_submatrix, result, summed_submatrix.size(), MPI_DOUBLE_COMPLEX,integration_comm,root_rank,  group);
  }
  else if(type_of_MPI_reduce=="asynchronous_split_less_nonblocking")
  {
    Propagation_pointer->MPI_NEMO_Reduce_nonblocking(&summed_submatrix, result, summed_submatrix.size(), MPI_DOUBLE_COMPLEX,integration_comm,root_rank,  group);
  }
  else
  {
    if(my_integration_rank==root_rank)
    {
      result->resize(summed_submatrix.size(),std::complex<double>(0.0,0.0));
    }
    MPI_Reduce(&(summed_submatrix[0]),&((*result)[0]),summed_submatrix.size(), MPI_DOUBLE_COMPLEX,MPI_SUM,root_rank, integration_comm);
  }

  NemoUtils::toc(prefix);
}

void PropagationUtilities::integrate_vector(Simulation* this_simulation, const std::map<std::vector<NemoMeshPoint>,std::vector<double> >& input_data,
                                            std::vector<double>& result, bool multiply_by_energy, Simulation* H_constructor,
                                            bool density_by_hole_factor, vector<int>* row_indices, bool is_hole)
{
  InputOptions options = this_simulation->get_reference_to_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);


  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_vector";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_vector ";

  result.clear();
  result.resize(input_data.begin()->second.size(),0.0);

  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }


  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");

  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);

  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
  //1.loop over incoming data
  std::map<std::vector<NemoMeshPoint>,std::vector<double> >::const_iterator c_it=input_data.begin();
  for(;c_it!=input_data.end();++c_it)
  {
    NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");
    double integration_weight = 1.0;
    for(unsigned int i=0;i<c_it->first.size();++i)
    {
      std::string momentum_mesh_name=writeable_Propagator->momentum_mesh_names[i];
      std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
      NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
      //2.get the integration weight and sum up locally
      double temp_double=PropagationUtilities::find_integration_weight(this_simulation, momentum_mesh_name, c_it->first, writeable_Propagator);
      integration_weight *= temp_double;

      if(multiply_by_energy)
      {
        if(momentum_mesh_name.find("energy")!=std::string::npos)
          integration_weight *= read_energy_from_momentum(this_simulation,c_it->first,writeable_Propagator);
      }

    }

    if(density_by_hole_factor)
    {
      std::map<int,int> index_to_atom_id_map;

      std::map<unsigned int, double> threshold_map;
      std::map<unsigned int, double> threshold_map_right;
      double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,c_it->first,writeable_Propagator);
      H_constructor->get_data(std::string("hole_factor_dof"), energy, threshold_map, threshold_map_right);
      for(unsigned int i=0; i<result.size(); i++)
      {
        //cerr << " i " << i << " row index " << " " << (*row_indices)[i] << " \n";
        std::map<unsigned int, double>::iterator thresh_it = threshold_map.find((*row_indices)[i]);
        NEMO_ASSERT(thresh_it!=threshold_map.end(),prefix+" could not find atom id in threshold map \n");
        double hole_factor = thresh_it->second;
        if(!is_hole)//input_type == NemoPhys::Fermion_lesser_Green)
          result[i]+=c_it->second[i]*integration_weight*(1-hole_factor);
        else //if (input_type == NemoPhys::Fermion_greater_Green)
          result[i]+=-c_it->second[i]*integration_weight*hole_factor;
        //else
        //  throw std::invalid_argument(prefix+"called with unknown input type \n");
      }
    }
    else
    {
      for(unsigned int i=0;i<result.size();i++)
        result[i]+=integration_weight*c_it->second[i];
    }
  }
  //3.communicate the result to get the global result
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  MPI_Barrier(topcomm);
  std::vector<double> temp_result=result;
  if(!PropOptionInterface->get_solve_on_single_replica())
    MPI_Allreduce(&(temp_result[0]),&(result[0]),result.size(), MPI_DOUBLE, MPI_SUM ,topcomm);
  else
    MPI_Reduce(&(temp_result[0]),&(result[0]),result.size(), MPI_DOUBLE, MPI_SUM, 0, topcomm);

  NemoUtils::toc(tic_toc_prefix);
}


void PropagationUtilities::integrate_vector_for_energy_resolved(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>,
                                                            std::vector<double > >* input_data,
                                                            std::map<double,std::vector<double> >* result,
                                                            bool multiply_by_energy)
{
  InputOptions options = this_simulation->get_reference_to_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);


  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_vector";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_vector ";



  std::vector<double > temp_result(input_data->size(),0.0);

  int data_size = input_data->begin()->second.size();

  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }

  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");

  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();

  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  //get unique energies
  //1.1 if energy resolved data is wanted, prepare the required container and the all_energies
  std::vector<double> all_energies;
  std::map<double, int> translation_map_energy_index;
  std::vector<vector<double> > all_kvectors;
  std::map<vector<double>, set<double> > all_energies_per_kvector;
  std::map<vector<double>, int > translation_map_kvector_index;

  std::set<std::vector<NemoMeshPoint> > all_momenta;
  std::set<std::vector<NemoMeshPoint> >* pointer_to_all_momenta =&all_momenta;

  //if(get_energy_resolved_data)
  {
    //1.1 find all energies that exist - on all MPI processes
    if(pointer_to_all_momenta->size()==0)
      Parallelizer->get_data("all_momenta",pointer_to_all_momenta);
    NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"received empty all_momenta\n");
    std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
    std::set<double> temp_all_energies;
    std::set<std::complex<double>,Compare_complex_numbers > temp_all_complex_energies;
    std::string tic_toc_prefix1b = NEMOUTILS_PREFIX(tic_toc_prefix+" 1b.");
    NemoUtils::tic(tic_toc_prefix1b);
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      {
        double temp_energy = PropagationUtilities::read_energy_from_momentum(this_simulation,*c_it, writeable_Propagator);
        temp_all_energies.insert(temp_energy);
      }
    }
    NemoUtils::toc(tic_toc_prefix1b);
    //1.2 save the ordered energies into a vector and save the index-mapping
    all_energies.resize(temp_all_energies.size(),0.0);
    std::string tic_toc_prefix1c = NEMOUTILS_PREFIX(tic_toc_prefix+" 1c.");
    NemoUtils::tic(tic_toc_prefix1c);
    int counter = 0;
    {
      std::set<double>::iterator e_it;
      for(e_it=temp_all_energies.begin(); e_it!=temp_all_energies.end(); e_it++)
      {
        double temp_energy = *e_it;
        translation_map_energy_index[temp_energy]=counter;
        all_energies[counter]=temp_energy;
        counter++;
      }
    }
    NemoUtils::toc(tic_toc_prefix1c);
  }

  //1.loop over incoming data
  std::map<std::vector<NemoMeshPoint>,std::vector<double > >::const_iterator c_it=input_data->begin();

  for(; c_it!=input_data->end(); ++c_it)
  {
    double energy_resolved_integration_weight=1.0;
    for(unsigned int i=0;i<c_it->first.size();++i)
    {

      if(writeable_Propagator->momentum_mesh_names[i].find("energy")==std::string::npos)
      {

        std::string momentum_mesh_name=writeable_Propagator->momentum_mesh_names[i];
        std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
        NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
        //2.get the integration weight and sum up locally
        double temp_double=find_integration_weight(this_simulation,momentum_mesh_name, c_it->first, writeable_Propagator);

        energy_resolved_integration_weight*=temp_double;
      }
    }

    //do the integration and store in result
    {
      double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,c_it->first,writeable_Propagator);
      //std::map<double,std::vector<std::complex<double> > >* energy_resolved_data=Propagator_pointer->get_energy_resolved_integrated_diagonal();
      //allocate the energy_resolved_data map
      if(result->size()!=all_energies.size())
      {
        result->clear();
        for(unsigned int i=0; i<all_energies.size(); i++)
          (*result)[all_energies[i]]=std::vector<double>(data_size);
      }
      std::map<double,std::vector<double > >::iterator edata_it=result->find(energy);
      if(edata_it==result->end())
      {
        (*result)[energy]=std::vector<double >(data_size);
        edata_it=result->find(energy);
      }
      NEMO_ASSERT(edata_it!=result->end(),prefix+"have not found energy entry in the energy resolved data map\n");
      if(!multiply_by_energy)
        for(unsigned int i=0; i<(unsigned int)data_size; i++)
          (edata_it->second)[i]+=(c_it->second)[i]*energy_resolved_integration_weight;
      else
        for(unsigned int i=0; i<(unsigned int )data_size; i++)
          (edata_it->second)[i]+=(c_it->second)[i]*energy_resolved_integration_weight*energy;
    }

  }

  //std::map<double,std::vector<std::complex<double> > >* energy_resolved_data=Propagator_pointer->get_energy_resolved_integrated_diagonal();
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  std::map<double,std::vector<double> >::iterator edata_it=result->begin();
  //perform the sum over MPI data for a single energy
  for(; edata_it!=result->end(); ++edata_it)
  {
    std::vector<double>  send_vector=edata_it->second;
    if(!PropOptionInterface->get_solve_on_single_replica())
      MPI_Allreduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE, MPI_SUM ,topcomm);
    else
      MPI_Reduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE, MPI_SUM , 0, topcomm);
  }
  //clean up the rest
  if(options.get_option("clean_up_other_ranks",false))
  {
    int rank;
    MPI_Comm_rank(topcomm,&rank);
    if(rank!=0)
      result->clear();
  }

  NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::integrate_vector_for_energy_resolved_per_k(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>,
    std::vector<double > >* input_data,
    std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >* result, bool solve_on_single_replica)
{
  InputOptions options = this_simulation->get_reference_to_options();
  //PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);

  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_vector_for_k_resolved";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_vector_for_k_resolved ";

  std::vector<double > temp_result(input_data->size(),0.0);

  //int data_size = input_data->begin()->second.size();

  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }

  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");

  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();

  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  //get unique energies
  //1.1 if k resolved resolved data is wanted, prepare the required container and the all_energies
  std::vector<double> all_energies;
  std::map<double, int> translation_map_energy_index;
  std::vector<vector<double> > all_kvectors;
  std::map<vector<double>, set<double> > all_energies_per_kvector;
  std::map<vector<double>, int > translation_map_kvector_index;

  std::set<std::vector<NemoMeshPoint> > all_momenta;
  std::set<std::vector<NemoMeshPoint> >* pointer_to_all_momenta =&all_momenta;

  //if(get_energy_resolved_nonrectangular_data)
  { //TODO @Fabio complex energy
    //find all energies that exist - on all MPI processes
    if(pointer_to_all_momenta->size()==0)
      Parallelizer->get_data("all_momenta",pointer_to_all_momenta);
    NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"received empty all_momenta\n");
    std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();

    std::set<std::complex<double>,Compare_complex_numbers > temp_all_complex_energies;
    //loop through all momenta
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      double temp_energy=0.0;
      temp_energy = PropagationUtilities::read_energy_from_momentum(this_simulation,*c_it,writeable_Propagator);
      std::vector<double> momentum_point=PropagationUtilities::read_kvector_from_momentum(this_simulation,*c_it, writeable_Propagator);
      std::map<vector<double>, set<double> >::iterator k_it = all_energies_per_kvector.find(momentum_point);
      if(k_it!=all_energies_per_kvector.end())
        k_it->second.insert(temp_energy);
      else
      {
        std::set<double> temp_set;
        temp_set.insert(temp_energy);
        all_energies_per_kvector[momentum_point] = temp_set;
      }
    }
  }

  //1.loop over incoming data
  std::map<std::vector<NemoMeshPoint>,std::vector<double > >::const_iterator c_it=input_data->begin();

  for(; c_it!=input_data->end(); ++c_it)
  {
    double energy_resolved_integration_weight=1.0;
    for(unsigned int i=0;i<c_it->first.size();++i)
    {

      if(writeable_Propagator->momentum_mesh_names[i].find("energy")==std::string::npos)
      {

        std::string momentum_mesh_name=writeable_Propagator->momentum_mesh_names[i];
        std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
        NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
        //2.get the integration weight and sum up locally
        double temp_double=find_integration_weight(this_simulation,momentum_mesh_name, c_it->first, writeable_Propagator);

          energy_resolved_integration_weight*=temp_double;

      }
    }

    //do the integration and store in result
    {

      double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,c_it->first,writeable_Propagator);
      vector<double> k_vector = PropagationUtilities::read_kvector_from_momentum(this_simulation,c_it->first,writeable_Propagator);
      //std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >* energy_per_k_data =
      //    Propagator_pointer->get_energy_resolved_per_k_integrated_diagonal();

      if(result->size()!=all_energies_per_kvector.size())
      {
        result->clear();
        std::map<std::vector<double>, std::set<double> >::iterator k_it = all_energies_per_kvector.begin();
        //for(unsigned int i=0; i<all_energies_per_kvector.size(); i++)
        for(; k_it != all_energies_per_kvector.end(); ++k_it)
        {
          std::map<double, std::vector<std::complex<double> > > temp_map;
          (*result)[k_it->first]=temp_map;//std::vector<std::complex<double> >(temp_result.size(),0.0);
        }
      }

      std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >::iterator k_data_it = result->begin();
      for(; k_data_it!=result->end(); k_data_it++)
      {
        std::map<vector<double>,std::set<double> >::iterator k_it = all_energies_per_kvector.find(k_data_it->first);
        if(k_data_it->second.size()!=k_it->second.size())
        {
          std::set<double>::iterator set_it = k_it->second.begin();
          for(; set_it != k_it->second.end(); set_it++)
            (k_data_it->second)[*set_it] = std::vector<std::complex<double> >(1,0.0);
        }
      }

      //find specific k
      k_data_it = result->find(k_vector);
      std::map<vector<double>,std::set<double> >::iterator k_it = all_energies_per_kvector.find(k_vector);
      {
        //std::map<double,std::vector<std::complex<double> > > temp_e_data = k_data_it->second;
        std::map<double,std::vector<std::complex<double> > >::iterator edata_it = k_data_it->second.begin();//temp_e_data.begin();

        edata_it = k_data_it->second.find(energy);
        NEMO_ASSERT(edata_it!=k_data_it->second.end(),prefix+"have not found energy entry in the energy resolved data map\n");
        std::vector<std::complex<double> > temp_vector;
        cplx average_input_scalar(0.0,0.0);
        for(unsigned int i=0; i<temp_result.size(); i++)
          average_input_scalar+=(c_it->second)[i]*energy_resolved_integration_weight;

        average_input_scalar/=temp_result.size();
        average_input_scalar = std::abs(average_input_scalar);
        (edata_it->second)[0]=average_input_scalar;
      }
    }

  }


  //std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >* r=
  //    Propagator_pointer->get_energy_resolved_per_k_integrated_diagonal();
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >::iterator k_it=result->begin();
   //perform the sum over MPI data for a single energy
  for(; k_it != result->end(); ++k_it)
  {
    std::map<double,std::vector<std::complex<double> > >::iterator edata_it= k_it->second.begin();
    for(; edata_it!=k_it->second.end(); ++edata_it)
    {
      std::vector<std::complex<double> > send_vector=edata_it->second;
      if(!solve_on_single_replica)
        MPI_Allreduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
      else
        MPI_Reduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM , 0, topcomm);
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::integrate_vector_for_energy_resolved_per_k(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>, std::vector<double> >* input_data,
    std::map<vector<double>, std::map<double, std::vector<double> > >* result, bool solve_on_single_replica)
{
  InputOptions options = this_simulation->get_reference_to_options();
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_averaged_scalar_for_energy_resolved_per_k";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_averaged_scalar_for_energy_resolved_per_k ";

  result->clear();

  //Get the parallelization
  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  int myrank;
  MPI_Comm_rank(topcomm, &myrank);

  //Get access to propagator
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  //Get access to k, E mesh information
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
  int energy_index = -1;
  int momentum_index = -1;
  for(unsigned int ii = 0; ii<(writeable_Propagator->momentum_mesh_names).size(); ii++)
  {
    if((writeable_Propagator->momentum_mesh_names)[ii].find("energy") != std::string::npos)
      energy_index = ii;
    else if((writeable_Propagator->momentum_mesh_names)[ii].find("momentum") != std::string::npos)
      momentum_index = ii;
    else
      throw std::runtime_error(tic_toc_prefix+" unknown momentum_mesh_name: "+(writeable_Propagator->momentum_mesh_names)[ii]+".\n");
  }
  NEMO_ASSERT(energy_index >= 0 && momentum_index >= 0, tic_toc_prefix + " could not find energy or momentum mesh. \n" );
  // k mesh
  std::string k_mesh_name = writeable_Propagator->momentum_mesh_names[momentum_index];
  std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(k_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+k_mesh_name+"\"\n");
  Simulation* k_mesh_constructor = temp_cit->second;
  // E mesh
  std::string E_mesh_name = writeable_Propagator->momentum_mesh_names[energy_index];
  temp_cit=Mesh_Constructors.find(E_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+E_mesh_name+"\"\n");
  //Simulation* E_mesh_constructor = temp_cit->second;

  //Get all k points  
  vector<NemoMeshPoint*> all_k_points;
  NemoMesh* k_mesh;
  k_mesh_constructor->get_data("k_space",k_mesh);
  all_k_points = k_mesh->get_mesh_points();
  NEMO_ASSERT(all_k_points.size() != 0, tic_toc_prefix + " k mesh contains zero points! \n");

  //Set up result container
  for(unsigned int ii = 0; ii<all_k_points.size(); ii++)
  {
    std::vector<double> k_point_vec(3);
    k_point_vec[0] = all_k_points[ii]->get_x();
    k_point_vec[1] = all_k_points[ii]->get_y();
    k_point_vec[2] = all_k_points[ii]->get_z();
    std::map<double, std::vector<double> > place_holder;
    (*result)[k_point_vec] = place_holder;
  }

  //Get all-momenta
  std::set<std::vector<NemoMeshPoint> >* all_momenta;
  Parallelizer->get_data("all_momenta", all_momenta);
  NEMO_ASSERT(all_momenta->size()>0, tic_toc_prefix + " received empty all_momenta\n");

  //Prepare for data to be reduced, and set up result container
  unsigned int counter = 0;
  std::set<std::vector<NemoMeshPoint> >::iterator it_all_momenta;
  std::map<std::vector<NemoMeshPoint>, std::vector<double> >::iterator it_input;
  unsigned int vector_size = (input_data->begin()->second).size();
  std::map<vector<double>, std::map<double, std::vector<double> > >::iterator it_result;
  for(it_all_momenta = all_momenta->begin(); it_all_momenta != all_momenta->end(); it_all_momenta++)
  {
    std::vector<double> temp_k_vec(3,0.0);
    double energy = (*it_all_momenta)[energy_index].get_x();
    NemoMeshPoint k_point = (*it_all_momenta)[momentum_index];
    temp_k_vec[0] = k_point.get_x();
    temp_k_vec[1] = k_point.get_y();
    temp_k_vec[2] = k_point.get_z();
    it_result = result->find(temp_k_vec);    
    NEMO_ASSERT(it_result != result->end(), tic_toc_prefix + " unable to find k point in result container. \n");
    std::vector<double> data_vector(vector_size,0.0);
    it_input = input_data->find(*it_all_momenta);
    if(it_input != input_data->end())
    {
      data_vector = it_input->second;
      counter++;
    }
    std::vector<double> send_data_vector = data_vector;
    if(solve_on_single_replica)
      MPI_Reduce(&(send_data_vector[0]), &(data_vector[0]), data_vector.size(), MPI_DOUBLE, MPI_SUM, 0, topcomm);
    else
      MPI_Allreduce(&(send_data_vector[0]), &(data_vector[0]), data_vector.size(), MPI_DOUBLE, MPI_SUM, topcomm);
    (it_result->second)[energy] = data_vector;
  }
  //Sanity check the data
  NEMO_ASSERT(counter == input_data->size(), prefix + "incomplete storage of input data\n");
  counter = 0;
  for(it_result = result->begin(); it_result != result->end(); it_result++)
    counter += (it_result->second).size();
  NEMO_ASSERT(counter == all_momenta->size(), prefix + "incomplete storage of all E,k points in the result container. \n");
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::integrate_scalar_for_energy_resolved_per_k(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>, double>* input_data,
    std::map<vector<double>, std::map<double, double> >* result, bool solve_on_single_replica)
{
  InputOptions options = this_simulation->get_reference_to_options();
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_scalar_for_energy_resolved_per_k";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_scalar_for_energy_resolved_per_k ";

  result->clear();

  //Get the parallelization
  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  int myrank;
  MPI_Comm_rank(topcomm, &myrank);

  //Get access to propagator
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  //Get access to k, E mesh information
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
  int energy_index = -1;
  int momentum_index = -1;
  for(unsigned int ii = 0; ii<(writeable_Propagator->momentum_mesh_names).size(); ii++)
  {
    if((writeable_Propagator->momentum_mesh_names)[ii].find("energy") != std::string::npos)
      energy_index = ii;
    else if((writeable_Propagator->momentum_mesh_names)[ii].find("momentum") != std::string::npos)
      momentum_index = ii;
    else
      throw std::runtime_error(tic_toc_prefix+" unknown momentum_mesh_name: "+(writeable_Propagator->momentum_mesh_names)[ii]+".\n");
  }
  NEMO_ASSERT(energy_index >= 0 && momentum_index >= 0, tic_toc_prefix + " could not find energy or momentum mesh. \n" );
  // k mesh
  std::string k_mesh_name = writeable_Propagator->momentum_mesh_names[momentum_index];
  std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(k_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+k_mesh_name+"\"\n");
  Simulation* k_mesh_constructor = temp_cit->second;
  // E mesh
  //std::string E_mesh_name = writeable_Propagator->momentum_mesh_names[energy_index];
  //temp_cit=Mesh_Constructors.find(E_mesh_name);
  //NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+E_mesh_name+"\"\n");
  //Simulation* E_mesh_constructor = temp_cit->second;

  //Get all k points  
  vector<NemoMeshPoint*> all_k_points;
  NemoMesh* k_mesh;
  k_mesh_constructor->get_data("k_space",k_mesh);
  all_k_points = k_mesh->get_mesh_points();
  NEMO_ASSERT(all_k_points.size() != 0, tic_toc_prefix + " k mesh contains zero points! \n");

  //Set up result container
  for(unsigned int ii = 0; ii<all_k_points.size(); ii++)
  {
    std::vector<double> k_point_vec(3);
    k_point_vec[0] = all_k_points[ii]->get_x();
    k_point_vec[1] = all_k_points[ii]->get_y();
    k_point_vec[2] = all_k_points[ii]->get_z();
    std::map<double, double> place_holder;
    (*result)[k_point_vec] = place_holder;
  }

  //Get all-momenta
  std::set<std::vector<NemoMeshPoint> >* all_momenta;
  Parallelizer->get_data("all_momenta", all_momenta);
  NEMO_ASSERT(all_momenta->size()>0, tic_toc_prefix + " received empty all_momenta\n");

  //Prepare for data to be reduced, and set up result container
  unsigned int counter = 0;
  unsigned int index = 0;
  std::vector<double> data_vector(all_momenta->size(),0.0);
  std::set<std::vector<NemoMeshPoint> >::iterator it_all_momenta;
  std::map<std::vector<NemoMeshPoint>, double>::iterator it_input;
  std::map<vector<double>, std::map<double, double> >::iterator it_result;
  for(it_all_momenta = all_momenta->begin(); it_all_momenta != all_momenta->end(); it_all_momenta++)
  {
    std::vector<double> temp_k_vec(3,0.0);
    double energy = (*it_all_momenta)[energy_index].get_x();
    NemoMeshPoint k_point = (*it_all_momenta)[momentum_index];
    temp_k_vec[0] = k_point.get_x();
    temp_k_vec[1] = k_point.get_y();
    temp_k_vec[2] = k_point.get_z();
    it_result = result->find(temp_k_vec);
    NEMO_ASSERT(it_result != result->end(), tic_toc_prefix + " unable to find k point in result container. \n");
    (it_result->second)[energy] = 0.0;

    it_input = input_data->find(*it_all_momenta);
    if(it_input != input_data->end())
    {
      data_vector[index] += it_input->second;
      counter++;
    }
    index++;
  }
  NEMO_ASSERT(counter == input_data->size(), prefix + "incomplete storage of input data\n");

  //Reduce incoming data upfront
  std::vector<double> send_data_vector = data_vector;
  if(solve_on_single_replica)
    MPI_Reduce(&(send_data_vector[0]), &(data_vector[0]), data_vector.size(), MPI_DOUBLE, MPI_SUM, 0, topcomm);
  else
    MPI_Allreduce(&(send_data_vector[0]), &(data_vector[0]), data_vector.size(), MPI_DOUBLE, MPI_SUM, topcomm);

  if(!solve_on_single_replica || myrank == 0)
  {
    index = 0;
    for(it_all_momenta = all_momenta->begin(); it_all_momenta != all_momenta->end(); it_all_momenta++)
    {
      std::vector<double> temp_k_vec(3,0.0);
      double energy = (*it_all_momenta)[energy_index].get_x();
      NemoMeshPoint k_point = (*it_all_momenta)[momentum_index];
      temp_k_vec[0] = k_point.get_x();
      temp_k_vec[1] = k_point.get_y();
      temp_k_vec[2] = k_point.get_z();
      it_result = result->find(temp_k_vec);
      NEMO_ASSERT(it_result != result->end(), tic_toc_prefix + " unable to find k point in result container. \n");
      (it_result->second)[energy] = data_vector[index];
      index++;
    }
  }
  MPI_Barrier(topcomm);
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::integrate_scalar(Simulation* this_simulation, const std::map<std::vector<NemoMeshPoint>,
    double >& input_data, double& result, bool solve_on_single_replica)
{
  InputOptions options = this_simulation->get_reference_to_options();
  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+tic_toc_name+"\")::integrate_scalar ");
  NemoUtils::tic(tic_toc_prefix);

  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);

  //Get the parallelization
   std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
   Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
   std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
   std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
   std::vector<std::string> Mesh_tree_names;
   {
     //get the Mesh_tree_names:
     Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
     //get the Mesh_tree_downtop
     Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
     //get the Mesh_tree_topdown
     Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
   }

  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");

  //Get access to propagator
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();

  //1.loop over incoming data
  result=0;
  std::map<std::vector<NemoMeshPoint>,double>::const_iterator c_it=input_data.begin();
  for(;c_it!=input_data.end();++c_it)
  {
    NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");
    double integration_weight = 1.0;
    for(unsigned int i=0;i<c_it->first.size();++i)
    {
      std::string momentum_mesh_name=writeable_Propagator->momentum_mesh_names[i];
      std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
      NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
      //2.get the integration weight and sum up locally
      double temp_double=find_integration_weight(this_simulation, momentum_mesh_name, c_it->first, writeable_Propagator);
      integration_weight *= temp_double;
    }
    result+=integration_weight*c_it->second;
  }
  //3.communicate the result to get the global result
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  MPI_Barrier(topcomm);
  double temp_result=result;
  if(!solve_on_single_replica)
    MPI_Allreduce(&temp_result,&result,1, MPI_DOUBLE, MPI_SUM ,topcomm);
  else
    MPI_Reduce(&temp_result,&result,1, MPI_DOUBLE, MPI_SUM, 0, topcomm);

  NemoUtils::toc(tic_toc_prefix);
}



void PropagationUtilities::integrate_averaged_scalar_for_energy_resolved_per_k(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>, std::vector<double> >* input_data,
    std::map<vector<double>, std::map<double, double> >* result, bool solve_on_single_replica)
{
  InputOptions options = this_simulation->get_reference_to_options();
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_averaged_scalar_for_energy_resolved_per_k";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_averaged_scalar_for_energy_resolved_per_k ";

  result->clear();

  //Get the parallelization
  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  int myrank;
  MPI_Comm_rank(topcomm, &myrank);

  //Get access to propagator
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  //Get access to k, E mesh information
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();
  int energy_index = -1;
  int momentum_index = -1;
  for(unsigned int ii = 0; ii<(writeable_Propagator->momentum_mesh_names).size(); ii++)
  {
    if((writeable_Propagator->momentum_mesh_names)[ii].find("energy") != std::string::npos)
      energy_index = ii;
    else if((writeable_Propagator->momentum_mesh_names)[ii].find("momentum") != std::string::npos)
      momentum_index = ii;
    else
      throw std::runtime_error(tic_toc_prefix+" unknown momentum_mesh_name: "+(writeable_Propagator->momentum_mesh_names)[ii]+".\n");
  }
  NEMO_ASSERT(energy_index >= 0 && momentum_index >= 0, tic_toc_prefix + " could not find energy or momentum mesh. \n" );
  // k mesh
  std::string k_mesh_name = writeable_Propagator->momentum_mesh_names[momentum_index];
  std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(k_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+k_mesh_name+"\"\n");
  Simulation* k_mesh_constructor = temp_cit->second;
  // E mesh
  std::string E_mesh_name = writeable_Propagator->momentum_mesh_names[energy_index];
  temp_cit=Mesh_Constructors.find(E_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+E_mesh_name+"\"\n");
  //Simulation* E_mesh_constructor = temp_cit->second;

  //Get all k points  
  vector<NemoMeshPoint*> all_k_points;
  NemoMesh* k_mesh;
  k_mesh_constructor->get_data("k_space",k_mesh);
  all_k_points = k_mesh->get_mesh_points();
  NEMO_ASSERT(all_k_points.size() != 0, tic_toc_prefix + " k mesh contains zero points! \n");

  //Set up result container
  for(unsigned int ii = 0; ii<all_k_points.size(); ii++)
  {
    std::vector<double> k_point_vec(3);
    k_point_vec[0] = all_k_points[ii]->get_x();
    k_point_vec[1] = all_k_points[ii]->get_y();
    k_point_vec[2] = all_k_points[ii]->get_z();
    std::map<double, double> place_holder;
    (*result)[k_point_vec] = place_holder;
  }

  //Get all-momenta
  std::set<std::vector<NemoMeshPoint> >* all_momenta;
  Parallelizer->get_data("all_momenta", all_momenta);
  NEMO_ASSERT(all_momenta->size()>0, tic_toc_prefix + " received empty all_momenta\n");

  //Prepare for data to be reduced, and set up result container
  unsigned int counter = 0;
  unsigned int index = 0;
  std::vector<double> data_vector(all_momenta->size(),0.0);
  std::set<std::vector<NemoMeshPoint> >::iterator it_all_momenta;
  std::map<std::vector<NemoMeshPoint>, std::vector<double> >::iterator it_input;
  std::map<vector<double>, std::map<double, double> >::iterator it_result;
  for(it_all_momenta = all_momenta->begin(); it_all_momenta != all_momenta->end(); it_all_momenta++)
  {
    std::vector<double> temp_k_vec(3,0.0);
    double energy = (*it_all_momenta)[energy_index].get_x();
    NemoMeshPoint k_point = (*it_all_momenta)[momentum_index];
    temp_k_vec[0] = k_point.get_x();
    temp_k_vec[1] = k_point.get_y();
    temp_k_vec[2] = k_point.get_z();
    it_result = result->find(temp_k_vec);    
    NEMO_ASSERT(it_result != result->end(), tic_toc_prefix + " unable to find k point in result container. \n");
    (it_result->second)[energy] = 0.0;

    it_input = input_data->find(*it_all_momenta);
    if(it_input != input_data->end())
    {
      double averaged_result = 0.0;
      std::vector<double > temp_vec = it_input->second;
      for(unsigned int ii = 0; ii < temp_vec.size(); ii++)
        averaged_result += temp_vec[ii];
      averaged_result /= temp_vec.size();
      data_vector[index] += averaged_result;
      counter++;
    }
    index++;
  }
  NEMO_ASSERT(counter == input_data->size(), prefix + "incomplete storage of input data\n");

  //Reduce incoming data upfront
  std::vector<double> send_data_vector = data_vector;
  if(solve_on_single_replica)
    MPI_Reduce(&(send_data_vector[0]), &(data_vector[0]), data_vector.size(), MPI_DOUBLE, MPI_SUM, 0, topcomm);
  else
    MPI_Allreduce(&(send_data_vector[0]), &(data_vector[0]), data_vector.size(), MPI_DOUBLE, MPI_SUM, topcomm);

  if(!solve_on_single_replica || myrank == 0)
  {
    index = 0;
    for(it_all_momenta = all_momenta->begin(); it_all_momenta != all_momenta->end(); it_all_momenta++)
    {
      std::vector<double> temp_k_vec(3,0.0);
      double energy = (*it_all_momenta)[energy_index].get_x();
      NemoMeshPoint k_point = (*it_all_momenta)[momentum_index];
      temp_k_vec[0] = k_point.get_x();
      temp_k_vec[1] = k_point.get_y();
      temp_k_vec[2] = k_point.get_z();
      it_result = result->find(temp_k_vec);   
      NEMO_ASSERT(it_result != result->end(), tic_toc_prefix + " unable to find k point in result container. \n");
      (it_result->second)[energy] = data_vector[index];
      index++;
    }
  }
  MPI_Barrier(topcomm);
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::integrate_scalar_for_k_resolved(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>, double>* input_data,
    std::map<vector<double>, double>* result, bool multiply_by_k, bool solve_on_single_replica)
{
  InputOptions options = this_simulation->get_reference_to_options();
  //PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);

  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_scalar_for_energy_resolved_per_k";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_scalar_for_energy_resolved_per_k ";

  result->clear();

  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }

  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");

  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();

  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  int energy_index = -1;
  int momentum_index = -1;
  //get unique k-points
  for(unsigned int ii = 0; ii<(writeable_Propagator->momentum_mesh_names).size(); ii++)
  {
    if((writeable_Propagator->momentum_mesh_names)[ii].find("energy") != std::string::npos)
      energy_index = ii;
    else if((writeable_Propagator->momentum_mesh_names)[ii].find("momentum") != std::string::npos)
      momentum_index = ii;
    else
      throw std::runtime_error(tic_toc_prefix+" unknown momentum_mesh_name: "+(writeable_Propagator->momentum_mesh_names)[ii]+".\n");
  }
  NEMO_ASSERT(energy_index >= 0 && momentum_index >= 0, tic_toc_prefix + " could not find energy or momentum mesh. \n" );

  std::string k_mesh_name = writeable_Propagator->momentum_mesh_names[momentum_index];
  std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(k_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+k_mesh_name+"\"\n");
  
  vector<NemoMeshPoint*> all_k_points;
  NemoMesh* k_mesh;
  Simulation* k_mesh_constructor = temp_cit->second;
  k_mesh_constructor->get_data("k_space",k_mesh);
  all_k_points = k_mesh->get_mesh_points();
  NEMO_ASSERT(all_k_points.size() != 0, tic_toc_prefix + " k mesh contains zero points! \n");

  for(unsigned int ii = 0; ii<all_k_points.size(); ii++) //Set up result container
  {
    std::vector<double> k_point_vec(3);
    k_point_vec[0] = all_k_points[ii]->get_x();
    k_point_vec[1] = all_k_points[ii]->get_y();
    k_point_vec[2] = all_k_points[ii]->get_z();
    (*result)[k_point_vec] = 0.0;
  }

  std::string E_mesh_name = writeable_Propagator->momentum_mesh_names[energy_index];
  temp_cit=Mesh_Constructors.find(E_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+E_mesh_name+"\"\n");
  //Simulation* E_mesh_constructor = temp_cit->second;

  std::map<std::vector<NemoMeshPoint>, double>::iterator it_data = input_data->begin();
  std::map<vector<double>, double>::iterator it_result = result->begin();
  for(; it_data != input_data->end(); it_data++) //loop over input data, do the integration.
  {
    std::vector<double> temp_k_pt(3);
    temp_k_pt[0] = ((it_data->first)[momentum_index]).get_x();
    temp_k_pt[1] = ((it_data->first)[momentum_index]).get_y();
    temp_k_pt[2] = ((it_data->first)[momentum_index]).get_z();
    it_result = result->find(temp_k_pt);
    NEMO_ASSERT(it_result != result->end(), tic_toc_prefix + " cannot find k point in result container. \n");
    //Find the energy integration weight
    double energy_integration_weight=1.0;
    energy_integration_weight = find_integration_weight(this_simulation, E_mesh_name, it_data->first, writeable_Propagator);
    it_result->second += it_data->second*energy_integration_weight;
  }

  if(multiply_by_k)
  {    
    for(it_result = result->begin(); it_result != result->end(); it_result++)
    {
      double k_norm = NemoMath::vector_norm(it_result->first);
      it_result->second *= k_norm;
    }
  }

  //Reduce results from all MPI-processes
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  for(it_result = result->begin(); it_result != result->end(); it_result++)
  {
    double send_number = it_result->second;
    if(!solve_on_single_replica)
      MPI_Allreduce(&send_number, &(it_result->second), 1, MPI_DOUBLE, MPI_SUM, topcomm);
    else
      MPI_Reduce(&send_number, &(it_result->second), 1, MPI_DOUBLE, MPI_SUM, 0, topcomm);
  }

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::integrate_averaged_scalar_for_k_resolved(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>, std::vector<double> >* input_data,
    std::map<vector<double>, double>* result, bool multiply_by_k, bool solve_on_single_replica)
{
  InputOptions options = this_simulation->get_reference_to_options();
  //PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);

  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_averaged_scalar_for_k_resolved";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_averaged_scalar_for_k_resolved ";

  result->clear();

  std::string who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
  Simulation* Parallelizer = this_simulation->find_simulation(who_is_parallelizing);
  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }

  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");

  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();

  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  int energy_index = -1;
  int momentum_index = -1;
  //get unique k-points
  for(unsigned int ii = 0; ii<(writeable_Propagator->momentum_mesh_names).size(); ii++)
  {
    if((writeable_Propagator->momentum_mesh_names)[ii].find("energy") != std::string::npos)
      energy_index = ii;
    else if((writeable_Propagator->momentum_mesh_names)[ii].find("momentum") != std::string::npos)
      momentum_index = ii;
    else
      throw std::runtime_error(tic_toc_prefix+" unknown momentum_mesh_name: "+(writeable_Propagator->momentum_mesh_names)[ii]+".\n");
  }
  NEMO_ASSERT(energy_index >= 0 && momentum_index >= 0, tic_toc_prefix + " could not find energy or momentum mesh. \n" );

  std::string k_mesh_name = writeable_Propagator->momentum_mesh_names[momentum_index];
  std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(k_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+k_mesh_name+"\"\n");
  
  vector<NemoMeshPoint*> all_k_points;
  NemoMesh* k_mesh;
  Simulation* k_mesh_constructor = temp_cit->second;
  k_mesh_constructor->get_data("k_space",k_mesh);
  all_k_points = k_mesh->get_mesh_points();
  NEMO_ASSERT(all_k_points.size() != 0, tic_toc_prefix + " k mesh contains zero points! \n");

  for(unsigned int ii = 0; ii<all_k_points.size(); ii++) //Set up result container
  {
    std::vector<double> k_point_vec(3);
    k_point_vec[0] = all_k_points[ii]->get_x();
    k_point_vec[1] = all_k_points[ii]->get_y();
    k_point_vec[2] = all_k_points[ii]->get_z();
    (*result)[k_point_vec] = 0.0;
  }

  std::string E_mesh_name = writeable_Propagator->momentum_mesh_names[energy_index];
  temp_cit=Mesh_Constructors.find(E_mesh_name);
  NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),tic_toc_prefix+"have not found constructor of mesh \""+E_mesh_name+"\"\n");
  //Simulation* E_mesh_constructor = temp_cit->second;

  std::map<std::vector<NemoMeshPoint>, std::vector<double> >::iterator it_data = input_data->begin();
  std::map<vector<double>, double>::iterator it_result = result->begin();
  for(; it_data != input_data->end(); it_data++) //loop over input data, do the integration.
  {
    std::vector<double> temp_k_pt(3);
    temp_k_pt[0] = ((it_data->first)[momentum_index]).get_x();
    temp_k_pt[1] = ((it_data->first)[momentum_index]).get_y();
    temp_k_pt[2] = ((it_data->first)[momentum_index]).get_z();
    it_result = result->find(temp_k_pt);
    if(it_result == result->end())
    {
      cout<<"target k: "<<temp_k_pt[0]<<"\t"<<temp_k_pt[1]<<"\t"<<temp_k_pt[2]<<endl;
      std::map<vector<double>, double>::iterator it2 = result->begin();
      cout<<"points in all k: "<<endl;
      for(; it2 != result->end(); it2++)
      {
        cout<<"avail k: "<<it2->first[0]<<"\t"<<it2->first[1]<<"\t"<<it2->first[2]<<endl;
        cout<<"diff: "<<it2->first[0]-temp_k_pt[0]<<"\t"<<it2->first[1]-temp_k_pt[1]<<"\t"<<it2->first[2]-temp_k_pt[2]<<endl;
      }

    }
    NEMO_ASSERT(it_result != result->end(), tic_toc_prefix + " cannot find k point in result container. \n");
    //Find the energy integration weight
    double energy_integration_weight=1.0;
    energy_integration_weight = find_integration_weight(this_simulation, E_mesh_name, it_data->first, writeable_Propagator);
    std::vector<double> data_vec = it_data->second;
    double raw_result = 0.0;
    for(unsigned int ii = 0; ii < data_vec.size(); ii++) //average input vector into a number
      raw_result += data_vec[ii];
    raw_result /= data_vec.size();
    it_result->second += raw_result * energy_integration_weight;
  }

  if(multiply_by_k)
  {    
    for(it_result = result->begin(); it_result != result->end(); it_result++)
    {
      double k_norm = NemoMath::vector_norm(it_result->first);
      it_result->second *= k_norm;
    }
  }

  //Reduce results from all MPI-processes
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  for(it_result = result->begin(); it_result != result->end(); it_result++)
  {
    double send_number = it_result->second;
    if(!solve_on_single_replica)
      MPI_Allreduce(&send_number, &(it_result->second), 1, MPI_DOUBLE, MPI_SUM, topcomm);
    else
      MPI_Reduce(&send_number, &(it_result->second), 1, MPI_DOUBLE, MPI_SUM, 0, topcomm);
  }

  NemoUtils::toc(tic_toc_prefix);
}

double PropagationUtilities::find_integration_weight(Simulation* this_simulation, const std::string& mesh_name, const std::vector<NemoMeshPoint>& momentum_point, const Propagator* input_Propagator)
{
  InputOptions options = this_simulation->get_reference_to_options();
//  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);


  std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::find_integration_weight";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::find_integration_weight ";

  //1. find the index of the momentum_point that corresponds to the mesh_name
  unsigned int mesh_index=input_Propagator->momentum_mesh_names.size();
  bool not_found=true;
  for(unsigned int i=0; i<input_Propagator->momentum_mesh_names.size()&&not_found; i++)
    if(input_Propagator->momentum_mesh_names[i]==mesh_name)
    {
      mesh_index=i;
      not_found=false;
    }
  std::map<string, Simulation* > Mesh_Constructors = PropInterface->get_Mesh_Constructors();

  NEMO_ASSERT(mesh_index<input_Propagator->momentum_mesh_names.size(),prefix+"have not found the mesh \""+mesh_name+"\"\n");
  //2. get the mesh_constructor corresponding to mesh_name
  std::map<std::string, Simulation*>::const_iterator c_it=Mesh_Constructors.find(mesh_name);
  NEMO_ASSERT(c_it!=Mesh_Constructors.end(),prefix+"have not found the constructor of mesh \""+mesh_name+"\"\n");
  NEMO_ASSERT(c_it->second!=NULL,prefix+"constructor of mesh \""+mesh_name+"\" is NULL\n");
  //3. get the integration weight via get_data to the solver of 2.
  double result;
  InputOptions& mesh_options=c_it->second->get_reference_to_options();
  if(!mesh_options.get_option(std::string("non_rectangular"),false))
    c_it->second->get_data("integration_weight",momentum_point[mesh_index],result);
  else
  {
    if(mesh_name.find("energy")!=std::string::npos)
    {
      //get the k-point from the momentum
      std::vector<double> temp_vector=PropagationUtilities::read_kvector_from_momentum(this_simulation, momentum_point, input_Propagator);
      NemoMeshPoint temp_momentum(0,temp_vector);
      std::vector<NemoMeshPoint> temp_vector_momentum(1,temp_momentum);
      c_it->second->get_data("integration_weight",temp_vector_momentum,momentum_point[mesh_index],result);
      NEMO_ASSERT(!PropInterface->complex_energy_used(),prefix+"complex_energy not supported here, yet\n");
    }
    else
      c_it->second->get_data("integration_weight",momentum_point,momentum_point[mesh_index],result);
  }
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

std::vector<double> PropagationUtilities::read_kvector_from_momentum(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum_point, const Propagator* input_Propagator,
    const NemoPhys::Momentum_type* input_momentum_type)
{
  InputOptions options = this_simulation->get_reference_to_options();
//     PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
     PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);


     std::string tic_toc_name = options.get_option("tic_toc_name",this_simulation->get_name());
     std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::read_kvector_from_momentum";
     NemoUtils::tic(tic_toc_prefix);
     std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::read_kvector_from_momentum ";

  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=PropInterface->get_momentum_mesh_types().begin();
  std::string k_vector_name=std::string("");
  for (; momentum_name_it!=PropInterface->get_momentum_mesh_types().end()&&k_vector_name==std::string(""); ++momentum_name_it)
    if(input_momentum_type==NULL)
    {
      if(momentum_name_it->second==NemoPhys::Momentum_2D||momentum_name_it->second==NemoPhys::Momentum_1D||momentum_name_it->second==NemoPhys::Momentum_3D)
        k_vector_name=momentum_name_it->first;
    }
    else
    {
      if(momentum_name_it->second==*input_momentum_type)
        k_vector_name=momentum_name_it->first;
    }

  unsigned int k_vector_index=momentum_point.size()+1; //larger then the vector size to throw exception if energy index is not found
  for (unsigned int i=0; i<input_Propagator->momentum_mesh_names.size(); i++)
    if(input_Propagator->momentum_mesh_names[i]==k_vector_name)
      k_vector_index=i;
  NEMO_ASSERT(k_vector_index<momentum_point.size(),prefix+"have not found any k-vector in the momentum\n");
  const NemoMeshPoint& k_vector_point=momentum_point[k_vector_index];
  std::vector<double> result=k_vector_point.get_coords();
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

void PropagationUtilities::do_solve_H_lesser(Simulation* this_simulation, Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix; //= NEMOUTILS_PREFIX("Propagation(\""+get_name()+"\")::do_solve_H_lesser ");//M.P. wrong but compiles
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this_simulation->get_name()+"\")::do_solve_H_lesser ";

  InputOptions options = this_simulation->get_reference_to_options();
  //  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();

  delete result;
  result=NULL;
  //PetscMatrixParallelComplex* temp_result=NULL;
  //1. solve the first summand
  //1.1 get the readable Propagator matrices of the first summand
  //1.1.1 get the forward retarded Green's function gR of the i+1 domain
  //1.1.1.1 get the name of the gR, its solver and the dofmap of it
  NEMO_ASSERT(options.check_option("half_way_retarded_green"),prefix+"please define \"half_way_retarded_green\"\n");
  std::string half_way_gRname=options.get_option("half_way_retarded_green",std::string(""));
  Simulation* source_of_half_way_gR = PropagationUtilities::find_source_of_data(this_simulation, half_way_gRname);
  //1.1.1.2 get the DOFmap this gR is solved on
  //get the Hamilton_constructor of source_of_half_way_gR
  Simulation* half_way_hamilton_constructor=NULL;
  source_of_half_way_gR->get_data("Hamilton_constructor",half_way_hamilton_constructor);
  NEMO_ASSERT(half_way_hamilton_constructor!=NULL,prefix+"have received NULL for the \"half_way_hamilton_constructor\"\n");
  //1.1.1.3 get the dofmap corresponding to the half_way domain
  Simulation* half_way_DOF_solver=half_way_hamilton_constructor;
  if(options.check_option("half_way_DOFmap_solver"))
  {
    std::string half_way_dof_solver_name=options.get_option("half_way_DOFmap_solver",std::string(""));
    half_way_DOF_solver=this_simulation->find_simulation(half_way_dof_solver_name);
    NEMO_ASSERT(half_way_DOF_solver!=NULL,prefix+"have not found solver \"half_way_dof_solver_name\"\n");
  }
  std::string variable_name=output_Propagator->get_name()+std::string("_lead_domain");
  std::string neighbor_domain_name;
  if (options.check_option(variable_name))
    neighbor_domain_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(prefix+"define \""+variable_name+"\"\n");
  const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);

  const DOFmapInterface& half_way_DOFmap=half_way_DOF_solver->get_const_dof_map(neighbor_domain);///variable_name,row_domain);

  //1.1.1.4 get access to the actual matrix
  PetscMatrixParallelComplex* half_way_gR_matrix=NULL;
  source_of_half_way_gR->get_data(half_way_gRname,&momentum_point,half_way_gR_matrix,&half_way_DOFmap);
  //1.1.2 get the exact lesser Green's function G< of the i domain
  //1.1.2.1 get the name of the G< and its solver
  NEMO_ASSERT(options.check_option("exact_lesser_green"),prefix+"please define \"exact_lesser_green\"\n");
  std::string exact_GLname=options.get_option("exact_lesser_green",std::string(""));
  Simulation* source_of_exact_GL = PropagationUtilities::find_source_of_data(this_simulation, exact_GLname);
  //1.1.2.2 get the dofmap of the exact G< section needed here
  NEMO_ASSERT(options.check_option("exact_DOFmap_solver"),prefix+"please define \"exact_DOFmap_solver\"\n");
  Simulation* exact_DOfmap_solver=this_simulation->find_simulation(options.get_option("exact_DOFmap_solver",std::string("")));
  NEMO_ASSERT(exact_DOfmap_solver!=NULL,prefix+"have not found \""+options.get_option("exact_DOFmap_solver",std::string(""))+"\"\n");
  //const DOFmap& exact_DOFmap=exact_DOfmap_solver->get_const_dof_map(get_const_simulation_domain());
  //1.1.2.3 get access to the actual matrix
  PetscMatrixParallelComplex* exact_GL_matrix=NULL;
  source_of_exact_GL->get_data(exact_GLname,&momentum_point,exact_GL_matrix,&this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()));
  //1.2 get the coupling Hamiltonian T=H_i,i+1
  PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;
  PetscMatrixParallelComplex* temp_coupling_Hamiltonian=NULL;
  DOFmapInterface& coupling_DOFmap = half_way_DOF_solver->get_dof_map(neighbor_domain);//NULL; //new DOFmap;
  //1.2.1 find the momentum that the Hamilton constructor can read out of momentum_point
  //std::set<unsigned int> Hamilton_momentum_indices;
  //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
  //find_Hamiltonian_momenta(output_Propagator,pointer_to_Hamilton_momentum_indices);
  //set_valley(output_Propagator, momentum_point, Hamilton_Constructor);
  //std::vector<double> temp_vector(3,0.0);
  //NemoMeshPoint temp_momentum(0,temp_vector);
  //if(pointer_to_Hamilton_momentum_indices!=NULL)
  //  temp_momentum=momentum_point[*(Hamilton_momentum_indices.begin())];
  //1.2.2 get the coupling Hamiltonian from Hamilton_Constructor as a big matrix

  std::vector<NemoMeshPoint> sorted_momentum;
  QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum,options,PropInterface->get_momentum_mesh_types(),Hamilton_Constructor);
  DOFmapInterface* temp_pointer=&coupling_DOFmap;
  DOFmapInterface* coupling_DOFmap_pointer=&coupling_DOFmap;
  Hamilton_Constructor->get_data(std::string("Hamiltonian"),sorted_momentum,neighbor_domain,temp_coupling_Hamiltonian,
      coupling_DOFmap_pointer,this_simulation->get_const_simulation_domain());

  if(coupling_DOFmap_pointer!=temp_pointer)
    delete coupling_DOFmap_pointer;
  //1.2.3 get the relevant submatrix of the coupling Hamiltonian and store in coupling_Hamiltonian
  //1.2.3.1 find the relevant rows (the local rows of exact_GL_matrix)
  int start_rows;
  int end_rows;
  exact_GL_matrix->get_ownership_range(start_rows,end_rows);
  std::vector<int> rows(end_rows-start_rows,0);
  for(int i=start_rows; i<end_rows; i++)
    rows[i]=i;
  //1.2.3.2 find the relevant columns (all columns after the maximum rows of the gR matrix)
  int num_cols=half_way_gR_matrix->get_num_cols();
  std::vector<int> columns(num_cols,0);
  int offset = exact_GL_matrix->get_num_rows();
  //if coupling Hamiltonian came back as dense.
  if(num_cols >= (int)(temp_coupling_Hamiltonian->get_num_rows()) || !temp_coupling_Hamiltonian->check_if_sparse())
    offset = 0;
  for(int i=0; i<num_cols; i++)
    columns[i]=offset+i;
  //1.2.3.3 store the submatrix
  int num_rows=exact_GL_matrix->get_num_rows();
  coupling_Hamiltonian = new PetscMatrixParallelComplex(num_rows, num_cols, this_simulation->get_simulation_domain()->get_communicator());
  coupling_Hamiltonian->set_num_owned_rows(end_rows-start_rows);

  std::vector<int> rows_diagonal(num_rows,0);
  std::vector<int> rows_offdiagonal(num_rows,0);
  for (int i = start_rows; i<end_rows; i++)
  {
    rows_diagonal[i] = temp_coupling_Hamiltonian->get_nz_diagonal(i);
    rows_offdiagonal[i] = temp_coupling_Hamiltonian->get_nz_offdiagonal(i);
  }
  for (int i = start_rows; i<end_rows; i++)
    coupling_Hamiltonian->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);

  temp_coupling_Hamiltonian->get_submatrix(rows,columns,MAT_INITIAL_MATRIX,coupling_Hamiltonian);
  delete temp_coupling_Hamiltonian;

  //2.1.1 get the forward lesser Green's function g< of the i+1 domain
  //2.1.1.1 get the name of the gR and its solver
  NEMO_ASSERT(options.check_option("half_way_lesser_green"),prefix+"please define \"half_way_lesser_green\"\n");
  std::string half_way_gLname=options.get_option("half_way_lesser_green",std::string(""));
  Simulation* source_of_half_way_gL = PropagationUtilities::find_source_of_data(this_simulation, half_way_gLname);
  //2.1.1.2 get the DOFmap this gL is solved on - same as in 1.1.1.2
  //get the dofmap corresponding to the half_way domain - same as in 1.1.1.2
  //2.1.1.3 get access to the actual matrix
  PetscMatrixParallelComplex* half_way_gL_matrix=NULL;
  source_of_half_way_gL->get_data(half_way_gLname,&momentum_point,half_way_gL_matrix,&half_way_DOFmap);
  //2.1.2 get the exact retarded Green's function GR of the i domain
  //2.1.2.1 get the name of the GR and its solver
  NEMO_ASSERT(options.check_option("exact_retarded_green"),prefix+"please define \"exact_retarded_green\"\n");
  std::string exact_GRname=options.get_option("exact_retarded_green",std::string(""));
  Simulation* source_of_exact_GR = PropagationUtilities::find_source_of_data(this_simulation,exact_GRname);
  //2.1.2.2 the dofmap of the exact GR agrees with this dofmap...
  //2.1.2.3 get access to the actual matrix
  PetscMatrixParallelComplex* exact_GR_matrix=NULL;
  source_of_exact_GR->get_data(exact_GRname,&momentum_point,exact_GR_matrix,&this_simulation->get_const_dof_map(this_simulation->get_const_simulation_domain()));

  //call core function
  bool diagonal_result = options.get_option("diagonal_Greens_function", true);
  core_H_lesser(coupling_Hamiltonian, half_way_gR_matrix, exact_GL_matrix, half_way_gL_matrix, exact_GR_matrix, result, diagonal_result);
  delete coupling_Hamiltonian;
  coupling_Hamiltonian=NULL;

/*
  //1.4 solve -G< T gA T'
  PetscMatrixParallelComplex* temp_matrix=new PetscMatrixParallelComplex(*half_way_gR_matrix);
  //1.4.1 get gA
  half_way_gR_matrix->hermitian_transpose_matrix(*temp_matrix,MAT_REUSE_MATRIX);
  //1.4.2 multiply gA with T
  PetscMatrixParallelComplex* temp_matrix2=NULL;
  PetscMatrixParallelComplex::mult(*coupling_Hamiltonian,*temp_matrix,&temp_matrix2);
  delete temp_matrix;
  temp_matrix=new PetscMatrixParallelComplex(coupling_Hamiltonian->get_num_cols(),coupling_Hamiltonian->get_num_rows(),coupling_Hamiltonian->get_communicator());
  //1.4.3 multiply temp_matrix2=T*gA with T'
  coupling_Hamiltonian->hermitian_transpose_matrix(*temp_matrix,MAT_INITIAL_MATRIX);
  PetscMatrixParallelComplex* temp_matrix3=NULL;
  PetscMatrixParallelComplex::mult(*temp_matrix2,*temp_matrix,&temp_matrix3);
  //1.4.4 multiply temp_matrix3=T*gA*T' with G<
  PetscMatrixParallelComplex::mult(*exact_GL_matrix,*temp_matrix3,&temp_result);
  //1.4.5 multiply result with -1
  temp_result*=std::complex<double>(-1.0,0.0);
  //multiply by i to offset multiplication of -i by calculate_density function
  *temp_result*=std::complex<double>(0.0,1.0);
  temp_result*=std::complex<double>(0.0,-1.0);
  //temp_result->save_to_matlab_file("temp_result_HG1.m");
  delete temp_matrix2;
  temp_matrix2=NULL;
  delete temp_matrix3;
  temp_matrix3=NULL;

  //2. solve the second summand and add to the result of 1.
  //2.1 get the readable Propagator matrices of the second summand
  //2.1.1 get the forward lesser Green's function g< of the i+1 domain
  //2.1.1.1 get the name of the gR and its solver

  //2.2 solve -GR T g< T'
  //2.2.1 multiply g< and T'
  PetscMatrixParallelComplex::mult(*half_way_gL_matrix,*temp_matrix,&temp_matrix2);
  //2.2.2 multiply temp_matrix2=g< T' with T
  PetscMatrixParallelComplex::mult(*coupling_Hamiltonian,*temp_matrix2,&temp_matrix3);
  delete temp_matrix2;
  temp_matrix2=NULL;
  delete temp_matrix;
  temp_matrix=NULL;
  //delete coupling_Hamiltonian;
  //coupling_Hamiltonian=NULL;
  //2.2.3 multiply temp_matrix3=T g< T' with GR
  PetscMatrixParallelComplex::mult(*exact_GR_matrix,*temp_matrix3,&temp_matrix);
  delete temp_matrix3;
  temp_matrix3=NULL;
  //2.3 substract temp_matrix from result of 1.
  //temp_result->add_matrix(*temp_matrix,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));
  //multiply by i to offset multiplication of -i by calculate_density function
  temp_result->add_matrix(*temp_matrix,DIFFERENT_NONZERO_PATTERN,std::complex<double>(0.0,1.0));
  //temp_matrix->save_to_matlab_file("temp_result_HG2.m");
  delete temp_matrix;
  temp_matrix=NULL;
  //3. store the diagonal result in result
  temp_result->get_diagonal_matrix(result);
  result->assemble();

  //Take care of prefactors except for the 1/((2*pi)^d) factor. This will come from calculate_density.
  *result*=std::complex<double>(2*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar,0);

  delete temp_result;
*/
  //set_job_done_momentum_map(this_simulation, &output_Propagator->get_name(),&momentum_point,true);
  NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::core_H_lesser(PetscMatrixParallelComplex* coupling_Hamiltonian, PetscMatrixParallelComplex* half_way_gR_matrix,
    PetscMatrixParallelComplex* exact_GL_matrix, PetscMatrixParallelComplex* half_way_gL_matrix,
    PetscMatrixParallelComplex* exact_GR_matrix, PetscMatrixParallelComplex*& result, bool diagonal_result)
{
  PetscMatrixParallelComplex* temp_result=NULL;

  //1.4 solve -G< T gA T'
  PetscMatrixParallelComplex* temp_matrix=new PetscMatrixParallelComplex(*half_way_gR_matrix);
  //1.4.1 get gA
  half_way_gR_matrix->hermitian_transpose_matrix(*temp_matrix,MAT_REUSE_MATRIX);
  //1.4.2 multiply gA with T
  PetscMatrixParallelComplex* temp_matrix2=NULL;
  PetscMatrixParallelComplex::mult(*coupling_Hamiltonian,*temp_matrix,&temp_matrix2);
  delete temp_matrix;
  temp_matrix=new PetscMatrixParallelComplex(coupling_Hamiltonian->get_num_cols(),coupling_Hamiltonian->get_num_rows(),coupling_Hamiltonian->get_communicator());
  //1.4.3 multiply temp_matrix2=T*gA with T'
  coupling_Hamiltonian->hermitian_transpose_matrix(*temp_matrix,MAT_INITIAL_MATRIX);
  PetscMatrixParallelComplex* temp_matrix3=NULL;
  PetscMatrixParallelComplex::mult(*temp_matrix2,*temp_matrix,&temp_matrix3);
  //1.4.4 multiply temp_matrix3=T*gA*T' with G<
  PetscMatrixParallelComplex::mult(*exact_GL_matrix,*temp_matrix3,&temp_result);
  //1.4.5 multiply result with -1
  //*temp_result*=std::complex<double>(-1.0,0.0);
  //multiply by i to offset multiplication of -i by calculate_density function
  *temp_result*=std::complex<double>(0.0,1.0);
  //*temp_result*=std::complex<double>(0.0,-1.0);
  //temp_result->save_to_matlab_file("temp_result_HG1.m");
  delete temp_matrix2;
  temp_matrix2=NULL;
  delete temp_matrix3;
  temp_matrix3=NULL;

  //2. solve the second summand and add to the result of 1.
  //2.1 get the readable Propagator matrices of the second summand
  //2.1.1 get the forward lesser Green's function g< of the i+1 domain
  //2.1.1.1 get the name of the gR and its solver

  //2.2 solve -GR T g< T'
  //2.2.1 multiply g< and T'
  PetscMatrixParallelComplex::mult(*half_way_gL_matrix,*temp_matrix,&temp_matrix2);
  //2.2.2 multiply temp_matrix2=g< T' with T
  PetscMatrixParallelComplex::mult(*coupling_Hamiltonian,*temp_matrix2,&temp_matrix3);
  delete temp_matrix2;
  temp_matrix2=NULL;
  delete temp_matrix;
  temp_matrix=NULL;
  //delete coupling_Hamiltonian;
  //coupling_Hamiltonian=NULL;
  //2.2.3 multiply temp_matrix3=T g< T' with GR
  PetscMatrixParallelComplex::mult(*exact_GR_matrix,*temp_matrix3,&temp_matrix);
  delete temp_matrix3;
  temp_matrix3=NULL;
  //2.3 substract temp_matrix from result of 1.
  //temp_result->add_matrix(*temp_matrix,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));
  //multiply by i to offset multiplication of -i by calculate_density function
  temp_result->add_matrix(*temp_matrix,DIFFERENT_NONZERO_PATTERN,std::complex<double>(0.0,1.0));
  //temp_matrix->save_to_matlab_file("temp_result_HG2.m");
  delete temp_matrix;
  temp_matrix=NULL;

  if (diagonal_result)
  {
    //3. store the diagonal result in result
    temp_result->get_diagonal_matrix(result);
    result->assemble();
    delete temp_result;
  }
  else  // Result is block-diagonal
  {
    result = temp_result;
    result->assemble();
  }

  //Take care of prefactors except for the 1/((2*pi)^d) factor. This will come from calculate_density.
  *result*=std::complex<double>(2*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar,0);

}

void PropagationUtilities::generate_homogeneous_energy_mesh(double Emin, double Emax, unsigned int num_of_points, double exponent, std::map<double, double>& energy_and_integration_weight, std::vector<double>* energy_mesh, Espace* espace)
{
  std::string tic_toc_prefix = "PropagationUtilities::generate_homogeneous_energy_mesh ";
  NemoUtils::tic(tic_toc_prefix);

  energy_and_integration_weight.clear();
  std::vector<double> integration_weight(num_of_points, 0.0);
  std::vector<double> temp_energy_vector(num_of_points, 0.0);
  for (unsigned int i = 0; i < num_of_points; i++)
  {
    double temp_energy = Emin + (Emax - Emin)*std::pow(i*1.0 / (1.0*(num_of_points - 1)), exponent);
    std::vector<double> energy_vector_for_mesh(1, temp_energy);
    temp_energy_vector[i] = temp_energy;
    if(espace!=NULL)
      espace->add_point(energy_vector_for_mesh);
    if (energy_mesh != NULL)
      (*energy_mesh)[i] = temp_energy;
  }
  for (unsigned int i = 1; i < num_of_points - 1; i++)
  {
    integration_weight[i] = (temp_energy_vector[i + 1] - temp_energy_vector[i - 1]) / 2.0;
  }
  integration_weight[0] = (temp_energy_vector[1] - temp_energy_vector[0]) / 2.0;
  integration_weight[num_of_points - 1] = (temp_energy_vector[num_of_points - 1] - temp_energy_vector[num_of_points - 2]) / 2.0;

  //fill the map
  for (unsigned int i = 0; i < num_of_points; i++)
  {
    energy_and_integration_weight[temp_energy_vector[i]] = integration_weight[i];
  }

  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::get_contact_sigmaR_nonlocal(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum , EOMMatrixInterface* eom_interface, unsigned int i,
    unsigned int number_of_offdiagonal, std::vector<string> subdomain_names,
    std::vector<std::pair<std::vector<int>,std::vector<int> > > diagonal_indices,
    PetscMatrixParallelComplexContainer*& L_block_container, PetscMatrixParallelComplexContainer* gR_container,
    PetscMatrixParallelComplex*& Sigma_contact, Simulation* scattering_sigmaR_solver)
{
  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this_simulation->get_name() + "\")::get_contact_sigmaR_nonlocal: ";
  delete Sigma_contact;
  Sigma_contact = NULL;

  const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& gR_blocks=gR_container->get_const_container_blocks();

  //result wil be stored in container for later use
  int row_idx = i;
  int start_idx = i - 1 - number_of_offdiagonal;
  if(start_idx<0)
    start_idx = 0;
  int end_idx = i - 1;

  for(int k = start_idx; k <= end_idx; ++k)
  {
    PetscMatrixParallelComplex* sigmaR = NULL;
    if(scattering_sigmaR_solver != NULL)
    {
      //find sigmaR( row_idx,k )
      std::string row_name = subdomain_names[row_idx];
      std::string col_name = subdomain_names[k];
      PropagationUtilities::extract_sigmaR_submatrix(this_simulation, momentum, row_name, col_name, eom_interface, scattering_sigmaR_solver,sigmaR);
      sigmaR->matrix_convert_dense();
    }

    PetscMatrixParallelComplex* L_row_k = NULL;
    PropagationUtilities::solve_L(this_simulation, momentum, eom_interface, number_of_offdiagonal, subdomain_names, row_idx, k/*col_idx*/,
        diagonal_indices, sigmaR, L_row_k, L_block_container, gR_container, scattering_sigmaR_solver);

    //get gR
    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator gR_cit = gR_blocks.find(diagonal_indices[k]);
    PetscMatrixParallelComplex* gR_k_k = gR_cit->second;

    PetscMatrixParallelComplex* temp_matrix = NULL;
    PetscMatrixParallelComplex::mult(*L_row_k, *gR_k_k, &temp_matrix);

    //transpose L_row_k
    PetscMatrixParallelComplex L_k_row(L_row_k->get_num_cols(),L_row_k->get_num_rows(),L_row_k->get_communicator());
    L_row_k->transpose_matrix(L_k_row,MAT_INITIAL_MATRIX);
    delete L_row_k;
    L_row_k = NULL;
    PetscMatrixParallelComplex* temp_matrix2 = NULL;
    PetscMatrixParallelComplex::mult(*temp_matrix,L_k_row,&temp_matrix2);

    if(Sigma_contact == NULL)
      Sigma_contact = new PetscMatrixParallelComplex(*temp_matrix2);
    else
      Sigma_contact->add_matrix(*temp_matrix2, DIFFERENT_NONZERO_PATTERN, cplx(1.0,0.0));
    delete temp_matrix;
    temp_matrix = NULL;
    delete temp_matrix2;
    temp_matrix2 = NULL;
  }

}

void PropagationUtilities::get_contact_sigmaR_from_GR(Simulation* this_simulation, Propagator*& Propagator, const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& Hamiltonian, 
                                                      PetscMatrixParallelComplex*& GR, PetscMatrixParallelComplex*& Sigma1, PetscMatrixParallelComplex*& Sigma_contact, PetscMatrixParallelComplex* SigmaS)
{
  //get contact sigma from G = [E-H-SigmaL-SigmaR-SigmaS]^-1
  std::string tic_toc_prefix = "PropagationUtilities::(\"" + this_simulation->get_name() + "\")::get_contact_sigmaR_from_GR: ";
  delete Sigma_contact;
  Sigma_contact = NULL;

  PetscMatrixParallelComplex* temp_inverse_GR = NULL;
  //GR^-1
  exact_inversion(this_simulation, momentum, GR, temp_inverse_GR);
  *temp_inverse_GR *= std::complex<double>(-1.0, 0.0); //- GR^-1
  std::complex<double> energy = read_complex_energy_from_momentum(this_simulation, momentum, Propagator);
  temp_inverse_GR->matrix_diagonal_shift(energy); //-GR^-1 + E
  temp_inverse_GR->add_matrix(*Hamiltonian, DIFFERENT_NONZERO_PATTERN, std::complex<double>(-1.0, 0.0)); //-GR^-1 + E - H
  temp_inverse_GR->add_matrix(*Sigma1, DIFFERENT_NONZERO_PATTERN, std::complex<double>(-1.0, 0.0)); //-GR^-1 + E - H - Sigma1
  if(SigmaS != NULL)
    temp_inverse_GR->add_matrix(*SigmaS, DIFFERENT_NONZERO_PATTERN, std::complex<double>(-1.0, 0.0)); //-GR^-1 + E - H - Sigma1 - SigmaS

  Sigma_contact = new PetscMatrixParallelComplex(*temp_inverse_GR);
  delete temp_inverse_GR;
  temp_inverse_GR = NULL;
}

void PropagationUtilities::solve_L(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum, EOMMatrixInterface* eom_interface,
    unsigned int number_of_offdiagonal, std::vector<string> subdomain_names,
    unsigned int row_idx, unsigned int col_idx, std::vector<std::pair<std::vector<int>,std::vector<int> > > diagonal_indices,
    PetscMatrixParallelComplex* sigmaR, PetscMatrixParallelComplex*& L_block, PetscMatrixParallelComplexContainer*& L_blocks_container,
    PetscMatrixParallelComplexContainer*& gR_container, Simulation* scattering_sigmaR_solver)
{
  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this_simulation->get_name() + "\")::solve_diagonal_gL: ";

  const InputOptions& options=this_simulation->get_options();
  PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);


  msg.threshold(4) << " row_idx " << row_idx << " col_idx " << col_idx << " \n";

  //get gR blocks
  //Propagator::PropagatorMap& result_prop_map=writeable_Propagator->propagator_map;
  //Propagator::PropagatorMap::iterator prop_it=result_prop_map.find(momentum);
  //NEMO_ASSERT(prop_it!=result_prop_map.end(),tic_toc_prefix + " could not find gR container");
  //PetscMatrixParallelComplexContainer* gR_container=dynamic_cast<PetscMatrixParallelComplexContainer*> (prop_it->second);
  const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& gR_blocks=gR_container->get_const_container_blocks();

  //get coupling H
  //including sigmaR
  //row is coupling domain
  Domain* lead_subdomain = Domain::get_domain(subdomain_names[col_idx]);
  //column is target domain
  Domain* target_subdomain = Domain::get_domain(subdomain_names[row_idx]);
  int num_target_rows = diagonal_indices[row_idx].first.size();
  int num_target_cols = num_target_rows;

  std::vector<NemoMeshPoint> temp_vector;
  QuantumNumberUtils::sort_quantum_number(momentum,temp_vector,options,PropInterface->get_momentum_mesh_types(),this_simulation);

  //find all leads
  const vector< const Domain*>& leads = target_subdomain->get_all_leads();
  int num_leads = leads.size();

  for(unsigned int idx = 0; (int)idx < num_leads; ++idx)
  {
    if (leads[idx] == lead_subdomain)
    {
      PetscMatrixParallelComplex* full_couplingH = NULL;
      DOFmapInterface* temp_pointer=NULL;
      eom_interface->get_EOMMatrix(temp_vector, target_subdomain, lead_subdomain,true,full_couplingH,temp_pointer);

      //PetscMatrixParallelComplex* coupling = NULL;
      PropagationUtilities::extract_coupling_Hamiltonian_RGF(this_simulation, num_target_rows, num_target_cols,  full_couplingH,
          L_block, sigmaR);
    }
  }

  if(L_block==NULL && sigmaR != NULL)
    L_block = new PetscMatrixParallelComplex(*sigmaR);


  if(L_block==NULL &&sigmaR==NULL )
  {
    //for testing make L_block matrix that is diagonal 0
    //PetscMatrixParallelComplex fake_0_matrix(diagonal_indices[0].first.size(), diagonal_indices[0].second.size(),
    //    get_const_simulation_domain()->get_communicator());
    L_block = new PetscMatrixParallelComplex(diagonal_indices[row_idx].first.size(), diagonal_indices[col_idx].second.size(),
        this_simulation->get_const_simulation_domain()->get_communicator());
    int start_row = 0;
    int end_row = diagonal_indices[row_idx].first.size(); //fix this
    L_block->set_num_owned_rows(end_row - start_row);
    for (int i = start_row; i < end_row; i++)
      L_block->set_num_nonzeros_for_local_row(i, 1, 0);
    L_block->consider_as_full();
    L_block->allocate_memory();
    L_block->set_to_zero();
    L_block->assemble();
  }



  int start_idx = row_idx/*col_idx*/ - 1 - number_of_offdiagonal;
  //check bounds
  if(start_idx < 0)
    start_idx = 0;
  int end_idx = col_idx - 1;//1;
  //check bounds
  //if(end_idx < 0)
  //  end_idx = 0;

  for(int k = start_idx; k <= end_idx && k>=0; k++)
  {
    //find L(row_idx,k)
    const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& L_blocks=L_blocks_container->get_const_container_blocks();

    std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_idx].first,diagonal_indices[k].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator Lblock_cit=L_blocks.find(temp_pair);
    //NEMO_ASSERT(Lblock_cit != L_blocks.end(),
    //      tic_toc_prefix + " have not found L block \n");
    PetscMatrixParallelComplex* L_row_k = NULL;
    if(Lblock_cit!=L_blocks.end())
      L_row_k = Lblock_cit->second;
    else
    {  //solve it
      PetscMatrixParallelComplex* sigmaR_row_k = NULL;
      if(scattering_sigmaR_solver != NULL)
      {
        //find sigmaR( row_idx,k )
        std::string row_name = subdomain_names[row_idx];
        std::string col_name = subdomain_names[k];
        PropagationUtilities::extract_sigmaR_submatrix(this_simulation, momentum, row_name, col_name, eom_interface, scattering_sigmaR_solver, sigmaR_row_k);
        sigmaR_row_k->matrix_convert_dense();
      }

      PropagationUtilities::solve_L(this_simulation, momentum, eom_interface, number_of_offdiagonal, subdomain_names, row_idx, k, diagonal_indices,
          sigmaR_row_k, L_row_k, L_blocks_container, gR_container, scattering_sigmaR_solver);
      delete sigmaR_row_k;
      sigmaR_row_k = NULL;
    }

    //find gR(k,k)
    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator gR_cit = gR_blocks.find(diagonal_indices[k]);
    PetscMatrixParallelComplex* gR_k_k = gR_cit->second;
    //find L(col_idx,k) -> transpose
    std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[col_idx].first,diagonal_indices[k].second);
    Lblock_cit=L_blocks.find(temp_pair2);
    //NEMO_ASSERT(Lblock_cit != L_blocks.end(),
    //     tic_toc_prefix + " have not found L block for transpose\n");
    //PetscMatrixParallelComplex* L_col_k = Lblock_cit->second;

    PetscMatrixParallelComplex* L_col_k = NULL;
    if(Lblock_cit!=L_blocks.end())
      L_col_k = Lblock_cit->second;
    else
    {  //solve it
      PetscMatrixParallelComplex* sigmaR_col_k = NULL;
      if(scattering_sigmaR_solver != NULL)
      {
        //find sigmaR( row_idx,k )
        std::string row_name = subdomain_names[row_idx];
        std::string col_name = subdomain_names[k];
        PropagationUtilities::extract_sigmaR_submatrix(this_simulation, momentum, row_name, col_name, eom_interface, scattering_sigmaR_solver, sigmaR_col_k);

      }
      PropagationUtilities::solve_L(this_simulation, momentum, eom_interface, number_of_offdiagonal, subdomain_names, col_idx, k, diagonal_indices,
          sigmaR_col_k, L_row_k, L_blocks_container, gR_container, scattering_sigmaR_solver);
    }

    //gR_k_k->assemble();
    //gR_k_k->save_to_matlab_file("gR_"+subdomain_names[k]+".m");
    PetscMatrixParallelComplex *temp_matrix = NULL;
    //multiply L(row_idx,k)*gR(k,k)
    PetscMatrixParallelComplex::mult(*L_row_k,*gR_k_k, &temp_matrix);


    //transpose
    PetscMatrixParallelComplex L_k_col(L_col_k->get_num_cols(), L_col_k->get_num_rows(),
        L_col_k->get_communicator());
    L_col_k->transpose_matrix(L_k_col,MAT_INITIAL_MATRIX);

    PetscMatrixParallelComplex *temp_matrix2 = NULL;
    PetscMatrixParallelComplex::mult(*temp_matrix, L_k_col, &temp_matrix2);
    temp_matrix2->assemble();
//    if(k==0&&row_idx==2)
//    {
//      L_block->save_to_matlab_file("L_block_2_0_withoutsigma.m");
//      temp_matrix2->save_to_matlab_file("sigma_2_0.m");
//    }

    delete temp_matrix;
    temp_matrix = NULL;
    //add result to coupling
    L_block->matrix_convert_dense();
    L_block->assemble();

    if(PropOptionInterface->get_debug_output())
    {
      temp_matrix2->save_to_matlab_file("Ltemp_sum_row_" + subdomain_names[row_idx]+  "_col_" + subdomain_names[col_idx]+"_.m");
      L_block->save_to_matlab_file("L_before_sum_row_" + subdomain_names[row_idx]+  "_col_" + subdomain_names[col_idx]+"_.m");
    }
    L_block->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
    delete temp_matrix2;
    temp_matrix2 = NULL;
  }

  L_block->zero_small_values(1E-12);
  //store coupling in L block
  L_blocks_container->set_block_from_matrix1(*L_block,diagonal_indices[row_idx].first,diagonal_indices[col_idx].second);

  if(PropOptionInterface->get_debug_output())
  {
    L_block->save_to_matlab_file("L_row_" + subdomain_names[row_idx]+  "_col_" + subdomain_names[col_idx]);
  }
}

void PropagationUtilities::extract_sigmaR_submatrix(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum,
    std::string row_name, std::string col_name, EOMMatrixInterface* eom_interface, Simulation* scattering_sigmaR_solver,
    PetscMatrixParallelComplex*& sigmaR)
{
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);

  BlockSchroedingerModule* temp_simulation=dynamic_cast<BlockSchroedingerModule*>(eom_interface);
  Domain* row_domain = Domain::get_domain(row_name);
  const DOFmapInterface& row_DOFmap=temp_simulation->get_dof_map(row_domain);
  Domain* col_domain = Domain::get_domain(col_name);
  const DOFmapInterface& col_DOFmap=temp_simulation->get_const_dof_map(col_domain);
  //PetscMatrixParallelComplex* temp_matrix = NULL;
  std::string temp_name;
  scattering_sigmaR_solver->get_data("writeable_Propagator",temp_name);
  scattering_sigmaR_solver->get_data(temp_name,&momentum,sigmaR,&row_DOFmap, &col_DOFmap);
  //    &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
  if(PropOptionInterface->get_debug_output())
  {
    sigmaR->assemble();
    sigmaR->save_to_matlab_file("sigmaR_"+row_name+"_"+col_name+".m");
  }

}

void PropagationUtilities::solve_offdiagonal_GL_nonlocal_blocks(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum,
    EOMMatrixInterface* eom_interface, std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
             int number_offdiagonal_blocks, int row_index, int col_index, int start_index, std::vector<string> subdomain_names,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gL_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks,
             Simulation* scattering_sigmaL_solver, PetscMatrixParallelComplexContainer*& GL_container,
             PetscMatrixParallelComplex*& result_GL)
{
  std::string tic_toc_prefix = "";//
  if(row_index==col_index)
    tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this_simulation->get_name() + "\")::solve_offdiagonal_GL_nonlocal_blocks diagonal: ";
  else
    tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this_simulation->get_name() + "\")::solve_offdiagonal_GL_nonlocal_blocks: offdiagonal";

  NemoUtils::tic(tic_toc_prefix);

  //Harshad: Temporary fix for unused variable warning
  //please remove when using start_index
  (void)start_index;

  Propagation* this_propagation=dynamic_cast<Propagation*>(this_simulation);
  msg.threshold(4) << "GL off  row " << row_index << " col " << col_index << " \n";
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GL_blocks =
      GL_container->get_const_container_blocks();

  bool print_all = false;
     if(row_index ==38 && col_index == 38)
       print_all = true;

  // copy g<(row,col) to G<(row,col)

  //find g<(row,col)
  std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[col_index].first,diagonal_indices[row_index].second);
  std::map< std::pair < std::vector<int>, std::vector<int> >,
  PetscMatrixParallelComplex* > ::const_iterator gLblock_cit = gL_blocks.find(temp_pair);
  NEMO_ASSERT(gLblock_cit != gL_blocks.end(),
      tic_toc_prefix + " have not found gL off block \n");
  PetscMatrixParallelComplex* gL_row_col = gLblock_cit->second;
  PetscMatrixParallelComplex gL_col_row(gL_row_col->get_num_cols(), gL_row_col->get_num_rows(),
      gL_row_col->get_communicator());
  gL_row_col->hermitian_transpose_matrix(gL_col_row, MAT_INITIAL_MATRIX);

  //if(col_index==row_index)
  gL_col_row *= cplx(-1.0,0.0);

  if(print_all && this_propagation->get_debug_output())
  {
    gL_col_row.save_to_matlab_file("print_gL_col_row_" + subdomain_names[row_index] + ".m");
  }

  PetscMatrixParallelComplex* temp_result_GL = NULL;//new PetscMatrixParallelComplex(gL_col_row);

  // calculate sum(L(row,j)*G<(j,col)
  NemoUtils::tic(tic_toc_prefix + " sumL*G<");

  PetscMatrixParallelComplex *Gl_row_row_temp = NULL;

  unsigned int end_index = row_index + number_offdiagonal_blocks;
  if(end_index>subdomain_names.size()-1)
    end_index = subdomain_names.size()-1;
  unsigned int start_index2 = row_index+1;
  if(start_index2>subdomain_names.size())
    start_index2 = subdomain_names.size()-1;
  for(unsigned int j = start_index2; j <= end_index; j++)
  {
    //find L(j,row) .. then transpose for L(row,j)
    std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[j].first,diagonal_indices[row_index].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >,
    PetscMatrixParallelComplex* > ::const_iterator Lblock_cit = L_blocks.find(temp_pair);
    NEMO_ASSERT(Lblock_cit != L_blocks.end(),
        tic_toc_prefix + " have not found L block \n");
    PetscMatrixParallelComplex* L_j_row = Lblock_cit->second;

    if(j>=(unsigned int)col_index)
    {
      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[j].first,diagonal_indices[col_index].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator GLblock_cit = GL_blocks.find(temp_pair);
      NEMO_ASSERT(GLblock_cit != GL_blocks.end(),
          tic_toc_prefix + " have not found GL off block \n");
      PetscMatrixParallelComplex* GL_j_col = GLblock_cit->second;

      PetscMatrixParallelComplex* temp_matrix = NULL;
      PetscMatrixParallelComplex L_row_j(L_j_row->get_num_cols(), L_j_row->get_num_rows(),
          L_j_row->get_communicator());
      L_j_row->transpose_matrix(L_row_j, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(L_row_j,*GL_j_col,&temp_matrix);

      if(print_all && this_propagation->get_debug_output())
      {
        temp_matrix->save_to_matlab_file("print_L_row_j_Gl_j_col_" + subdomain_names[row_index]
                                                                       + "_" + subdomain_names[j]  + ".m");
      }

      if(Gl_row_row_temp == NULL)
        Gl_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix);
      else
      {
        Gl_row_row_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix;
      temp_matrix = NULL;

    }
    else
    {
      //find GL(col,j)
      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[col_index].first,diagonal_indices[j].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator GLblock_cit = GL_blocks.find(temp_pair);
      NEMO_ASSERT(GLblock_cit != GL_blocks.end(),
          tic_toc_prefix + " have not found gL off block \n");
      PetscMatrixParallelComplex* GL_col_j = GLblock_cit->second;

      PetscMatrixParallelComplex L_row_j(L_j_row->get_num_cols(), L_j_row->get_num_rows(),
          L_j_row->get_communicator());
      L_j_row->transpose_matrix(L_row_j, MAT_INITIAL_MATRIX);

      PetscMatrixParallelComplex* temp_matrix2 = NULL;
      PetscMatrixParallelComplex GL_j_col(GL_col_j->get_num_cols(), GL_col_j->get_num_rows(),
          GL_col_j->get_communicator());
      GL_col_j->hermitian_transpose_matrix(GL_j_col, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(L_row_j,GL_j_col,&temp_matrix2);



      *temp_matrix2 *= cplx(-1.0,0.0);

      if(print_all && this_propagation->get_debug_output())
      {
        temp_matrix2->save_to_matlab_file("print_L_row_j_Gl_j_col_" + subdomain_names[row_index]
                                                                                      + "_j" + subdomain_names[j]  + ".m");
      }

      if(Gl_row_row_temp == NULL)
      {
        temp_matrix2->assemble();
        Gl_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix2);
      }
      else
      {
        Gl_row_row_temp->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix2;
      temp_matrix2 = NULL;
    }
  }

  NemoUtils::toc(tic_toc_prefix + " sumL*G<");


  std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[row_index].first,diagonal_indices[row_index].second);
  std::map< std::pair < std::vector<int>, std::vector<int> >,
  PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
  NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
      tic_toc_prefix + " have not found gr diagonal block \n");
  PetscMatrixParallelComplex* gR_row_row = gRblock_cit->second;

  PetscMatrixParallelComplex* temp_matrix = NULL;
  PetscMatrixParallelComplex::mult(*gR_row_row,*Gl_row_row_temp,&temp_matrix);
  if(print_all && this_propagation->get_debug_output())
  {
    temp_matrix->save_to_matlab_file("print_gR_row_row_Gl_row_row_temp_" + subdomain_names[row_index]  + ".m");
  }
  if(temp_result_GL == NULL)
  {
    temp_result_GL = new PetscMatrixParallelComplex(*temp_matrix);
  }
  else
  {
    temp_result_GL->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
  }

  //temp_result_GL->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
  delete temp_matrix;
  temp_matrix = NULL;

  if(this_propagation->get_debug_output())
  {
    Gl_row_row_temp->save_to_matlab_file("Gl_row_row_temp_"+subdomain_names[col_index]+"_"+subdomain_names[row_index]+".m");
  }

  delete Gl_row_row_temp;
  Gl_row_row_temp = NULL;

  NemoUtils::tic(tic_toc_prefix + " sumL*Ga");

  //sum(gL(row,j)*sum(L(j,k)*Ga(k,col))
  PetscMatrixParallelComplex *Ga_row_row_temp = NULL;
  PetscMatrixParallelComplex *Ga_row_row_sigma_temp = NULL;
  //sum(gR(row,j)*sum(sig<(j,k)*Ga(k,col))
  int start_index_Ga = row_index-number_offdiagonal_blocks;//row_index+1;// - number_offdiagonal_blocks;
  if (start_index_Ga <0)
    start_index_Ga = 0;
  int end_index_Ga = col_index;

  //modify the start and end indices
  if(row_index==col_index)
  {
    //if(start_index_Ga > abs(row_index-number_offdiagonal_blocks))
     start_index_Ga =  row_index - number_offdiagonal_blocks;
     if(start_index_Ga< 0)
       start_index_Ga = 0;
     //end_index_Ga = row_index + number_offdiagonal_blocks-1;
     //if(end_index_Ga > subdomain_names.size()-2)
     //  end_index_Ga = subdomain_names.size()-2;
   //end_index_Ga = start_index_Ga + number_offdiagonal_blocks;
   //if(end_index_Ga > subdomain_names.size() - 1)
   //    end_index_Ga = subdomain_names.size() - 1;
  }
  //sum(gL(row,j)*sum(L(j,k)*Ga(k,col))
  for(int j = start_index_Ga; j <= end_index_Ga; j++)
  {
    PetscMatrixParallelComplex *tmp_sum_L_Ga = NULL;
    PetscMatrixParallelComplex* tmp_sum_sigL_Ga = NULL;
    int start_index_sum_L_Ga = col_index + 1;
    int end_index_sum_L_Ga = j+number_offdiagonal_blocks;//end_index; //col_index + number_offdiagonal_blocks;
    //if(start_index_sum_L_Ga > subdomain_names.size()-1)
    //  start_index_sum_L_Ga = subdomain_names.size()-1;
    if(end_index_sum_L_Ga > subdomain_names.size()-1)
      end_index_sum_L_Ga = subdomain_names.size()-1;

    if(row_index==col_index)
    {
      //end_index_sum_L_Ga = start_index_sum_L_Ga + number_offdiagonal_blocks-1;

      //if(end_index > col_index + number_offdiagonal_blocks)
      //end_index_sum_L_Ga = col_index + number_offdiagonal_blocks;
      //if(end_index_sum_L_Ga > subdomain_names.size()-1)
      //  end_index_sum_L_Ga = subdomain_names.size()-1;
    }
    //if(end_index_sum_L_)
    //if (row_index == col_index)
    //{

    //}
    for(int k = start_index_sum_L_Ga; k <= end_index_sum_L_Ga && ((abs(k-j)<= number_offdiagonal_blocks) || (col_index!=row_index)) ; k++)
    {
      NemoUtils::tic(tic_toc_prefix + " sum(L(j,k)*Ga(k,col))");

      msg.threshold(4) << "GL off  j " << j << " k " << k << " \n";

      NemoUtils::tic(tic_toc_prefix + " L_j_k_conj_Ga_k_col");
      //find L(k,j) .. then Hermitian transpose for L(j,k)'
      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[k].first,diagonal_indices[j].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator Lblock_cit = L_blocks.find(temp_pair);
      NEMO_ASSERT(Lblock_cit != L_blocks.end(),
          tic_toc_prefix + " have not found L block \n");
      PetscMatrixParallelComplex* L_k_j = Lblock_cit->second;

      PetscMatrixParallelComplex L_j_k_conj(L_k_j->get_num_cols(), L_k_j->get_num_rows(),
          L_k_j->get_communicator());
      L_k_j->hermitian_transpose_matrix(L_j_k_conj, MAT_INITIAL_MATRIX);

      //find Gr(col_index,k)
      PetscMatrixParallelComplex * Ga_k_col = NULL;
      PetscMatrixParallelComplex* temp_matrix = NULL;
      if(col_index>=k)
      {
        std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[col_index].first,diagonal_indices[k].second);
        std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator GRblock_cit=GR_blocks.find(temp_pair2);
        NEMO_ASSERT(GRblock_cit != GR_blocks.end(),
            tic_toc_prefix + " have not found GR off block \n");
        PetscMatrixParallelComplex* Gr_col_k = GRblock_cit->second;

        Ga_k_col = new PetscMatrixParallelComplex(Gr_col_k->get_num_cols(), Gr_col_k->get_num_rows(),
            Gr_col_k->get_communicator());
        Gr_col_k->hermitian_transpose_matrix(*Ga_k_col, MAT_INITIAL_MATRIX);
        NemoUtils::tic(tic_toc_prefix + " mult colgreaterk L_j_k_conj_Ga_k_col");
        PetscMatrixParallelComplex::mult(L_j_k_conj,*Ga_k_col,&temp_matrix);
        NemoUtils::toc(tic_toc_prefix + " mult colgreaterk L_j_k_conj_Ga_k_col");

        if(print_all && this_propagation->get_debug_output())
        {
          temp_matrix->save_to_matlab_file("print_L_j_k_conj_Ga_k_col" + subdomain_names[row_index]
                                            + "_j" + subdomain_names[j] + "_k" + subdomain_names[k] + ".m");
        }
      }
      else
      {
        std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[k].first,diagonal_indices[col_index].second);
        std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator GRblock_cit=GR_blocks.find(temp_pair2);
        NEMO_ASSERT(GRblock_cit != GR_blocks.end(),
            tic_toc_prefix + " have not found GR off block \n");

        PetscMatrixParallelComplex* Gr_k_col= GRblock_cit->second;
        PetscMatrixParallelComplex Gr_col_k(Gr_k_col->get_num_cols(), Gr_k_col->get_num_rows(),
            Gr_k_col->get_communicator());
        Gr_k_col->transpose_matrix(Gr_col_k, MAT_INITIAL_MATRIX);
        Ga_k_col = new PetscMatrixParallelComplex(Gr_col_k.get_num_cols(), Gr_col_k.get_num_rows(),
            Gr_col_k.get_communicator());
        Gr_col_k.hermitian_transpose_matrix(*Ga_k_col, MAT_INITIAL_MATRIX);
        NemoUtils::tic(tic_toc_prefix + " mult collesserk L_j_k_conj_Ga_k_col");
        PetscMatrixParallelComplex::mult(L_j_k_conj,*Ga_k_col,&temp_matrix);
        NemoUtils::toc(tic_toc_prefix + " mult collesserk L_j_k_conj_Ga_k_col");

        if(print_all && this_propagation->get_debug_output())
        {
          temp_matrix->save_to_matlab_file("print_L_j_k_conj_Ga_k_col" + subdomain_names[row_index]
                                                                                         + "_j" + subdomain_names[j] + "_k" + subdomain_names[k] + ".m");
        }
      }

      if(tmp_sum_L_Ga == NULL)
      {
        temp_matrix->assemble();
        tmp_sum_L_Ga = new PetscMatrixParallelComplex(*temp_matrix);
      }
      else
      {
        tmp_sum_L_Ga->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix;
      temp_matrix = NULL;

      NemoUtils::toc(tic_toc_prefix + " L_j_k_conj_Ga_k_col");

      //for row!=col the index j should be less than or equal to row. For diagonal case this is not true
      if( (row_index==col_index || (j<=row_index) ) && scattering_sigmaL_solver!=NULL)
      {
        NemoUtils::tic(tic_toc_prefix + " sum_sigL*Ga");

        msg.threshold(4) << "Gl off solving  tmp_sum_sigL_Ga j " << j << " k " << k << " \n";
        PetscMatrixParallelComplex* sigmaL_j_k = NULL;
        int row_sigL = j;
        int col_sigL = k;
        if(j<k)
        {
          row_sigL = k;
          col_sigL = j;
        }
        std::string row_name = subdomain_names[row_sigL];
        std::string col_name = subdomain_names[col_sigL];
        PropagationUtilities::extract_sigmaR_submatrix(this_simulation, momentum, row_name, col_name, eom_interface, scattering_sigmaL_solver, sigmaL_j_k);
        sigmaL_j_k->matrix_convert_dense();
        //sigma<(k,j) = -(sigma(j,k))'
        PetscMatrixParallelComplex sigmaL_k_j_conj(sigmaL_j_k->get_num_cols(), sigmaL_j_k->get_num_rows(),
            sigmaL_j_k->get_communicator());
        sigmaL_j_k->hermitian_transpose_matrix(sigmaL_k_j_conj, MAT_INITIAL_MATRIX);

        PetscMatrixParallelComplex *temp_matrix = NULL;
        PetscMatrixParallelComplex::mult(sigmaL_k_j_conj,*Ga_k_col,&temp_matrix);
        *temp_matrix *= cplx(-1.0,0.0);

        if(tmp_sum_sigL_Ga == NULL)
        {
          temp_matrix->assemble();
          tmp_sum_sigL_Ga = new PetscMatrixParallelComplex(*temp_matrix);
        }
        else
        {
          tmp_sum_sigL_Ga->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
        }
        delete temp_matrix;
        temp_matrix = NULL;

        NemoUtils::toc(tic_toc_prefix + " sum_sigL*Ga");

      }
      delete Ga_k_col;
      Ga_k_col = NULL;
      NemoUtils::toc(tic_toc_prefix + " sum(L(j,k)*Ga(k,col))");
    }
    //Ga_row_row_temp = sum(g<(row,j)*tmp_sum_L_Ga)
    PetscMatrixParallelComplex* temp_matrix = NULL;

    if(tmp_sum_L_Ga!=NULL)
    {
    if(row_index>=j)
    {
      NemoUtils::tic(tic_toc_prefix + " gL*tmp_sum_L_Ga");

      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_index].first,diagonal_indices[j].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator gLblock_cit = gL_blocks.find(temp_pair);
      NEMO_ASSERT(gLblock_cit != gL_blocks.end(),
          tic_toc_prefix + " have not found gL off block \n");
      PetscMatrixParallelComplex* gL_row_j= gLblock_cit->second;

      PetscMatrixParallelComplex::mult(*gL_row_j,*tmp_sum_L_Ga,&temp_matrix);

      if(print_all && this_propagation->get_debug_output())
      {
        temp_matrix->save_to_matlab_file("print_gL_row_j_tmp_sum_L_Ga" + subdomain_names[row_index]
                                                                                       + "_j" + subdomain_names[j]  + ".m");
      }
      NemoUtils::toc(tic_toc_prefix + " gL*tmp_sum_L_Ga");

      //*temp_matrix *= cplx(-1.0,0.0);
    }
    else
    {
      NemoUtils::tic(tic_toc_prefix + " gL*tmp_sum_L_Ga");


      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[j].first,diagonal_indices[row_index].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator gLblock_cit = gL_blocks.find(temp_pair);
      NEMO_ASSERT(gLblock_cit != gL_blocks.end(),
          tic_toc_prefix + " have not found gL off block \n");
      PetscMatrixParallelComplex* gL_j_row = gLblock_cit->second;

      PetscMatrixParallelComplex gL_row_j_conj(gL_j_row->get_num_cols(), gL_j_row->get_num_rows(),
          gL_j_row->get_communicator());
      gL_j_row->hermitian_transpose_matrix(gL_row_j_conj, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(gL_row_j_conj,*tmp_sum_L_Ga,&temp_matrix);
      *temp_matrix *= cplx(-1.0,0.0);

      if(print_all && this_propagation->get_debug_output())
      {
        temp_matrix->save_to_matlab_file("print_gL_row_j_tmp_sum_L_Ga" + subdomain_names[row_index]
                                                                                         + "_j" + subdomain_names[j] + ".m");
      }

      NemoUtils::toc(tic_toc_prefix + " gL*tmp_sum_L_Ga");

    }
    if(this_propagation->get_debug_output())
     {
       if(Ga_row_row_temp!=NULL)
         Ga_row_row_temp->save_to_matlab_file("Ga_row_row_temp_"+subdomain_names[col_index]+"_"+subdomain_names[row_index]+".m");
       if(Ga_row_row_sigma_temp!=NULL)
         Ga_row_row_sigma_temp->save_to_matlab_file("Ga_row_row_sigma_temp_"+subdomain_names[col_index]+"_"+subdomain_names[row_index]+".m");

    if(tmp_sum_L_Ga!=NULL)
    {
      std::ostringstream j_str;
      j_str << j ;
      tmp_sum_L_Ga->save_to_matlab_file("tmp_sum_L_Ga"+subdomain_names[col_index]+"_"+subdomain_names[row_index]+"_" + j_str.str() + ".m");
    }


     }
    delete tmp_sum_L_Ga;
    tmp_sum_L_Ga = NULL;

    if(Ga_row_row_temp == NULL)
    {
      temp_matrix->assemble();
      Ga_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix);
    }
    else
    {
      Ga_row_row_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
    }
    delete temp_matrix;
    temp_matrix = NULL;
    }

    if(scattering_sigmaL_solver!=NULL && tmp_sum_sigL_Ga!=NULL)
    {
      NemoUtils::tic(tic_toc_prefix + " gR*tmp_sum_sigL_Ga");

      if(row_index<=j)
      {
        std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_index].first,diagonal_indices[j].second);
        std::map< std::pair < std::vector<int>, std::vector<int> >,
        PetscMatrixParallelComplex* > ::const_iterator gRblock_cit = gR_blocks.find(temp_pair);
        NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
            tic_toc_prefix + " have not found gR off block \n");
        PetscMatrixParallelComplex* gR_row_j= gRblock_cit->second;

        PetscMatrixParallelComplex::mult(*gR_row_j,*tmp_sum_sigL_Ga,&temp_matrix);

        //*temp_matrix *= cplx(-1.0,0.0);
      }
      else
      {
        std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[j].first,diagonal_indices[row_index].second);
        std::map< std::pair < std::vector<int>, std::vector<int> >,
        PetscMatrixParallelComplex* > ::const_iterator gRblock_cit = gR_blocks.find(temp_pair);
        NEMO_ASSERT(gLblock_cit != gR_blocks.end(),
            tic_toc_prefix + " have not found gR off block \n");
        PetscMatrixParallelComplex* gR_j_row = gRblock_cit->second;

        PetscMatrixParallelComplex gR_row_j_conj(gR_j_row->get_num_cols(), gR_j_row->get_num_rows(),
            gR_j_row->get_communicator());
        gR_j_row->transpose_matrix(gR_row_j_conj, MAT_INITIAL_MATRIX);

        PetscMatrixParallelComplex::mult(gR_row_j_conj,*tmp_sum_sigL_Ga,&temp_matrix);
      }
      if(Ga_row_row_sigma_temp == NULL)
      {
        temp_matrix->assemble();
        Ga_row_row_sigma_temp = new PetscMatrixParallelComplex(*temp_matrix);
      }
      else
      {
        Ga_row_row_sigma_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix;
      temp_matrix = NULL;
      delete tmp_sum_sigL_Ga;
      tmp_sum_sigL_Ga = NULL;

      NemoUtils::toc(tic_toc_prefix + " gR*tmp_sum_sigL_Ga");

    }
  }

  NemoUtils::toc(tic_toc_prefix + " sumL*Ga");

  if(Ga_row_row_temp!=NULL)
    temp_result_GL->add_matrix(*Ga_row_row_temp,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
  if(scattering_sigmaL_solver!=NULL && Ga_row_row_sigma_temp!=NULL)
    temp_result_GL->add_matrix(*Ga_row_row_sigma_temp,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
  if(this_propagation->get_debug_output())
  {
    if(Ga_row_row_temp!=NULL)
      Ga_row_row_temp->save_to_matlab_file("Ga_row_row_temp_"+subdomain_names[col_index]+"_"+subdomain_names[row_index]+".m");
    if(Ga_row_row_sigma_temp!=NULL)
      Ga_row_row_sigma_temp->save_to_matlab_file("Ga_row_row_sigma_temp_"+subdomain_names[col_index]+"_"+subdomain_names[row_index]+".m");

  }
  delete Ga_row_row_temp;
  Ga_row_row_temp = NULL;
  delete Ga_row_row_sigma_temp;
  Ga_row_row_sigma_temp = NULL;

  if(row_index==col_index)// && row_index!=0)//col_index==38 && row_index==38)
  {
    NemoMath::symmetry_type type = NemoMath::antihermitian;
    symmetrize(this_simulation,temp_result_GL, type);
  }

   //temp_result_GL->add_matrix(gL_col_row,SAME_NONZERO_PATTERN,cplx(1.0,0.0));


  //TODO:JC remove the need for transpose
  result_GL = new PetscMatrixParallelComplex(temp_result_GL->get_num_cols(), temp_result_GL->get_num_rows(),
      temp_result_GL->get_communicator());
  temp_result_GL->hermitian_transpose_matrix(*result_GL, MAT_INITIAL_MATRIX);
  delete temp_result_GL;
  temp_result_GL = NULL;

  result_GL->matrix_convert_dense();
  *result_GL *= cplx(-1.0,0.0);

  result_GL->add_matrix(*gL_row_col,SAME_NONZERO_PATTERN,cplx(1.0,0.0));


  if(this_propagation->get_debug_output())
   {
     result_GL->assemble();
     result_GL->save_to_matlab_file("Gl_exact_off_"+subdomain_names[col_index]+"_"+subdomain_names[row_index]+".m");


   }

  unsigned int num_cols = result_GL->get_num_cols();
  unsigned int num_rows = result_GL->get_num_rows();
  for(unsigned int i=0; i<num_rows; i++)
    result_GL->set_num_nonzeros_for_local_row(i,num_cols,0);

  //gL_col_row
  //temp_result_GL

  //  if(row_index==col_index)
  //  {
  //    std::vector<cplx> diagonal;
//    result_GL->get_diagonal(&diagonal);
//
//    double tmp_sum_real = 0.0;
//    double tmp_sum_imag = 0.0;
//    for (unsigned int idx =0; idx < diagonal.size(); idx++)
//    {
//      tmp_sum_real += diagonal[idx].real();
//      tmp_sum_imag += diagonal[idx].imag();
//    }
//    std::string name_of_propagator;
//    Propagator* writeable_Propagator=NULL;
//    PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
//    PropInterface->get_Propagator(writeable_Propagator);
//    double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum,writeable_Propagator);
//    cerr << "Energy " << energy << " row " << row_index << " real " <<  tmp_sum_real << " imag " << tmp_sum_imag << " \n";
//    /diagonal.real();
//  }



  //col > row so store that way
  GL_container->set_block_from_matrix1(*result_GL, diagonal_indices[col_index].first, diagonal_indices[row_index].second);

  NemoUtils::toc(tic_toc_prefix);

}


void PropagationUtilities::solve_offdiagonal_gL_nonlocal_blocks(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum,
    EOMMatrixInterface* eom_interface, std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
             int number_offdiagonal_blocks, int row_index, int col_index, int start_index, std::vector<string> subdomain_names,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
             PetscMatrixParallelComplex* gR_row_row, Simulation* scattering_sigmaL_solver, PetscMatrixParallelComplexContainer*& gL_container,
             PetscMatrixParallelComplex*& result_gL)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this_simulation->get_name() + "\")::solve_offdiagonal_gL_nonlocal_blocks: ";
  NemoUtils::tic(tic_toc_prefix);

  //Harshad: Temporary fix for unused variable warning
  //please remove when using number_offdiagonal_blocks
  (void)number_offdiagonal_blocks;

  Propagation* this_propagation=dynamic_cast<Propagation*>(this_simulation);
  msg.threshold(4) << "gL off  row " << row_index << " col " << col_index << " \n";
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gL_blocks =
      gL_container->get_const_container_blocks();

  unsigned int end_index = row_index-1;

  //stores summation of LDL(row,j)*gL(j,col)
  PetscMatrixParallelComplex *gl_row_row_temp = NULL;
  //stores summation of sigmaL(row,j)*gL(j,col) // where j < col.
  PetscMatrixParallelComplex *gl_row_row_sigma_temp = NULL;

  for(unsigned int j = start_index; j <= end_index; j++)
  {
    msg.threshold(4) << "gr off  j " << j << " col " << col_index << " \n";

    //find L(j,row) .. then transpose for L(row,j)
    std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_index].first,diagonal_indices[j].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >,
    PetscMatrixParallelComplex* > ::const_iterator Lblock_cit = L_blocks.find(temp_pair);
    NEMO_ASSERT(Lblock_cit != L_blocks.end(),
        tic_toc_prefix + " have not found L block \n");
    PetscMatrixParallelComplex* L_j_row = Lblock_cit->second;

    //we are storing lower diagonal blocks
    if(j>=(unsigned int)col_index)
    {
      //find gL(j,col)
      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[j].first,diagonal_indices[col_index].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator gLblock_cit = gL_blocks.find(temp_pair);
      NEMO_ASSERT(gLblock_cit != gL_blocks.end(),
          tic_toc_prefix + " have not found gL off block \n");
      PetscMatrixParallelComplex* gL_j_col = gLblock_cit->second;

      PetscMatrixParallelComplex* temp_matrix = NULL;
     // PetscMatrixParallelComplex L_row_j(L_j_row->get_num_cols(), L_j_row->get_num_rows(),
      //    L_j_row->get_communicator());
      //L_j_row->transpose_matrix(L_row_j, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(*L_j_row/*L_row_j*/,*gL_j_col,&temp_matrix);

      //if(j==col_index)
      //  *temp_matrix *= cplx(-1.0,0.0);

      if(gl_row_row_temp == NULL)
        gl_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix);
      else
      {
        gl_row_row_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix;
      temp_matrix = NULL;

    }
    else // we don't store upper diagonal blocks so need to use symmetry relation g<(i,j) = -(g<(j,i)'
    {
      //find gL(col,j)
      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[col_index].first,diagonal_indices[j].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator gLblock_cit = gL_blocks.find(temp_pair);
      NEMO_ASSERT(gLblock_cit != gL_blocks.end(),
          tic_toc_prefix + " have not found gL off block \n");
      PetscMatrixParallelComplex* gL_col_j = gLblock_cit->second;

      //PetscMatrixParallelComplex L_row_j(L_j_row->get_num_cols(), L_j_row->get_num_rows(),
       //   L_j_row->get_communicator());
      //L_j_row->transpose_matrix(L_row_j, MAT_INITIAL_MATRIX);

      PetscMatrixParallelComplex* temp_matrix2 = NULL;
      PetscMatrixParallelComplex gL_j_col(gL_col_j->get_num_cols(), gL_col_j->get_num_rows(),
          gL_col_j->get_communicator());
      gL_col_j->hermitian_transpose_matrix(gL_j_col, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(*L_j_row,gL_j_col,&temp_matrix2);

      *temp_matrix2 *= cplx(-1.0,0.0);

      if(gl_row_row_temp == NULL)
      {
        temp_matrix2->assemble();
        gl_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix2);
      }
      else
      {
        gl_row_row_temp->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix2;
      temp_matrix2 = NULL;
    }

    if(j<=(unsigned int)col_index && scattering_sigmaL_solver!=NULL)
    {
      //calculate sigL(row,j)*ga(j,col)
      //
      //find gR(col,j)
      std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[j].first,diagonal_indices[col_index].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
      NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
          tic_toc_prefix + " have not found gr off block \n");
      //PetscMatrixParallelComplex* gR_col_j = gRblock_cit->second;
      PetscMatrixParallelComplex gR_col_j(gRblock_cit->second->get_num_cols(), gRblock_cit->second->get_num_rows(),
          gRblock_cit->second->get_communicator());
      gRblock_cit->second->transpose_matrix(gR_col_j, MAT_INITIAL_MATRIX);

      //find sigma(row,j)
      PetscMatrixParallelComplex* sigmaL_row_j = NULL;
      std::string row_name = subdomain_names[row_index];
      std::string col_name = subdomain_names[j];
      PropagationUtilities::extract_sigmaR_submatrix(this_simulation, momentum, row_name, col_name, eom_interface, scattering_sigmaL_solver, sigmaL_row_j);
      sigmaL_row_j->matrix_convert_dense();
      //       //find sigma(j,row) then sigma(row,j) = -sigma(j,row)'
      //PetscMatrixParallelComplex sigmaL_row_j(sigmaL_j_row->get_num_cols(), sigmaL_j_row->get_num_rows(),
      //    sigmaL_j_row->get_communicator());
      //sigmaL_j_row->hermitian_transpose_matrix(sigmaL_row_j, MAT_INITIAL_MATRIX);

      PetscMatrixParallelComplex gA_j_col(gR_col_j.get_num_cols(), gR_col_j.get_num_rows(),
          gR_col_j.get_communicator());
      gR_col_j.hermitian_transpose_matrix(gA_j_col, MAT_INITIAL_MATRIX);

      PetscMatrixParallelComplex* temp_matrix = NULL;
      PetscMatrixParallelComplex::mult(*sigmaL_row_j,gA_j_col,&temp_matrix);

      if(gl_row_row_sigma_temp == NULL)
      {
        //*temp_matrix *= cplx(-1.0,0.0);
        // temp_matrix->assemble();
        gl_row_row_sigma_temp = new PetscMatrixParallelComplex(*temp_matrix);

      }
      else
      {
        gl_row_row_sigma_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix;
      temp_matrix = NULL;

    }
  }
  //std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[row_index].first,diagonal_indices[row_index].second);
  //std::map< std::pair < std::vector<int>, std::vector<int> >,
  //PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
  //NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
 //     tic_toc_prefix + " have not found gr diagonal block \n");
  //PetscMatrixParallelComplex* gR_row_row = gRblock_cit->second;

  PetscMatrixParallelComplex::mult(*gR_row_row,*gl_row_row_temp,&result_gL);

  if(scattering_sigmaL_solver!=NULL)
  {
    //
    PetscMatrixParallelComplex* temp_matrix = NULL;
    PetscMatrixParallelComplex::mult(*gR_row_row,*gl_row_row_sigma_temp,&temp_matrix);
    result_gL->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
    delete temp_matrix;
    temp_matrix = NULL;
  }



  if(this_propagation->get_debug_output())
  {
    result_gL->assemble();
    result_gL->save_to_matlab_file("gl_off_"+subdomain_names[row_index]+"_"+subdomain_names[col_index]+".m");


    gl_row_row_temp->save_to_matlab_file("gl_row_row_temp_" + subdomain_names[row_index]+"_"+subdomain_names[col_index]+".m");
    if(gl_row_row_sigma_temp!=NULL)
      gl_row_row_sigma_temp->save_to_matlab_file("gl_row_row_sigma_temp_" + subdomain_names[row_index]+"_"+subdomain_names[col_index]+".m");

  }

  msg.threshold(4) << " diagonal_indices[row_index].first " <<  diagonal_indices[row_index].first[0] << " \n";
  msg.threshold(4) << " diagonal_indices[col_index].second " <<  diagonal_indices[col_index].second[0] << " \n";
  result_gL->matrix_convert_dense();

  gL_container->set_block_from_matrix1(*result_gL, diagonal_indices[row_index].first, diagonal_indices[col_index].second);



  delete gl_row_row_temp;
  gl_row_row_temp = NULL;
  delete gl_row_row_sigma_temp;
  gl_row_row_sigma_temp = NULL;



  NemoUtils::toc(tic_toc_prefix);

}


void PropagationUtilities::solve_diagonal_gL_nonlocal_blocks(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum,
    EOMMatrixInterface* eom_interface, std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
             int number_offdiagonal_blocks, int row_index, int start_index, std::vector<string> subdomain_names,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
             PetscMatrixParallelComplex* Sigma_lesser_contact, PetscMatrixParallelComplex* gR_row_row,
             Simulation* scattering_sigmaL_solver, PetscMatrixParallelComplexContainer*& gL_container,
             PetscMatrixParallelComplex*& result_gL)

{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this_simulation->get_name() + "\")::solve_diagonal_gL_nonlocal_blocks: ";
  NemoUtils::tic(tic_toc_prefix);

  //Harshad: Temporary fix for unused variable warning
  //please remove when using number_offdiagonal_blocks
  (void)number_offdiagonal_blocks;

  Propagation* this_propagation=dynamic_cast<Propagation*>(this_simulation);
  msg.threshold(4) << "gL diag  row " << row_index << " \n";
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gL_blocks =
      gL_container->get_const_container_blocks();

  bool print_all = false;
  if(row_index ==1)
    print_all = true;
  unsigned int end_index = row_index-1;

  //stores summation of LDL(row,j)*gL(j,col)
  PetscMatrixParallelComplex *gl_row_row_temp = NULL;
  //stores summation of sigmaL(row,j)*gL(j,col) //
  PetscMatrixParallelComplex *gl_row_row_sigma_temp = NULL;

  for(unsigned int j = start_index; j <= end_index; ++j)
  {

    //calculate -sum(LDL(row,j)*glesser(row,j)'
    {
      //find gL(col,j)
      std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_index].first,diagonal_indices[j].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator gLblock_cit = gL_blocks.find(temp_pair);
      NEMO_ASSERT(gLblock_cit != gL_blocks.end(),
          tic_toc_prefix + " have not found gL off block \n");
      PetscMatrixParallelComplex* gL_row_j = gLblock_cit->second;

      std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[row_index].first,diagonal_indices[j].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* > ::const_iterator Lblock_cit = L_blocks.find(temp_pair2);
      NEMO_ASSERT(Lblock_cit != L_blocks.end(),
          tic_toc_prefix + " have not found L block \n");
      PetscMatrixParallelComplex* L_row_j = Lblock_cit->second;


      PetscMatrixParallelComplex* temp_matrix2 = NULL;
      PetscMatrixParallelComplex gL_j_row(gL_row_j->get_num_cols(), gL_row_j->get_num_rows(),
          gL_row_j->get_communicator());
      gL_row_j->hermitian_transpose_matrix(gL_j_row, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(*L_row_j,gL_j_row,&temp_matrix2);

      if(this_propagation->get_debug_output()&&print_all)
      {
        temp_matrix2->save_to_matlab_file("print_L_row_j_gL_j_row"+subdomain_names[row_index]+"_"+subdomain_names[j]+".m");
      }

      if(gl_row_row_temp == NULL)
      {
        *temp_matrix2 *= cplx(-1.0,0.0);
        temp_matrix2->assemble();
        gl_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix2);
      }
      else
      {
        gl_row_row_temp->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN,cplx(-1.0,0.0));
      }
      delete temp_matrix2;
      temp_matrix2 = NULL;
    }

    if(scattering_sigmaL_solver!=NULL)
    {
      //this is sum(gA(j,k)*LDL(k,i)
      PetscMatrixParallelComplex* temp_gA_sum = NULL;

      for(unsigned int k = j; k <= end_index; ++k)
      {
        //find L(row,k)
        std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_index].first,diagonal_indices[k].second);
        std::map< std::pair < std::vector<int>, std::vector<int> >,
        PetscMatrixParallelComplex* > ::const_iterator Lblock_cit = L_blocks.find(temp_pair);
        NEMO_ASSERT(Lblock_cit != L_blocks.end(),
            tic_toc_prefix + " have not found L block \n");
        PetscMatrixParallelComplex* L_row_k = Lblock_cit->second;

        PetscMatrixParallelComplex L_k_row(L_row_k->get_num_cols(), L_row_k->get_num_rows(),
            L_row_k->get_communicator());
        L_row_k->hermitian_transpose_matrix(L_k_row, MAT_INITIAL_MATRIX);

        //find gr(k,j)
        std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[j].first,diagonal_indices[k].second);
        std::map< std::pair < std::vector<int>, std::vector<int> >,
        PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
        NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
            tic_toc_prefix + " have not found gr diagonal block \n");
        PetscMatrixParallelComplex gR_k_j(gRblock_cit->second->get_num_cols(), gRblock_cit->second->get_num_rows(),
            gRblock_cit->second->get_communicator());
        gRblock_cit->second->transpose_matrix(gR_k_j, MAT_INITIAL_MATRIX);

        //PetscMatrixParallelComplex* gR_k_j = gRblock_cit->second;

        PetscMatrixParallelComplex gA_j_k(gR_k_j.get_num_cols(), gR_k_j.get_num_rows(),
            gR_k_j.get_communicator());
        gR_k_j.hermitian_transpose_matrix(gA_j_k, MAT_INITIAL_MATRIX);
        PetscMatrixParallelComplex* temp_matrix = NULL;
        PetscMatrixParallelComplex::mult(gA_j_k,L_k_row,&temp_matrix);
        if(temp_gA_sum == NULL)
        {
          temp_gA_sum = new PetscMatrixParallelComplex(*temp_matrix);
        }
        else
        {
          temp_gA_sum->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
        }
        delete temp_matrix;
        temp_matrix = NULL;

      }


      PetscMatrixParallelComplex* sigmaL_row_j = NULL;
      std::string row_name = subdomain_names[row_index];
      std::string col_name = subdomain_names[j];
      PropagationUtilities::extract_sigmaR_submatrix(this_simulation, momentum, row_name, col_name, eom_interface, scattering_sigmaL_solver, sigmaL_row_j);
      sigmaL_row_j->matrix_convert_dense();
      PetscMatrixParallelComplex* temp_matrix = NULL;
      PetscMatrixParallelComplex::mult(*sigmaL_row_j,*temp_gA_sum,&temp_matrix);

      if(gl_row_row_sigma_temp == NULL)
      {
        gl_row_row_sigma_temp = new PetscMatrixParallelComplex(*temp_matrix);

      }
      else
      {
        gl_row_row_sigma_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }
      delete temp_matrix;
      temp_matrix = NULL;
      delete temp_gA_sum;
      temp_gA_sum = NULL;
    }
  }

  PetscMatrixParallelComplex* temp_result_gL = NULL;
  if(gl_row_row_temp!=NULL)
  {
    PetscMatrixParallelComplex* temp_matrix2 = NULL;
    PetscMatrixParallelComplex::mult(*gR_row_row,*gl_row_row_temp,&temp_matrix2);
    delete gl_row_row_temp;
    gl_row_row_temp = NULL;
    gl_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix2);

    delete temp_matrix2;
    temp_matrix2 = NULL;

    //      {
      //        std::vector<cplx> diagonal;
    //        gl_row_row_temp->get_diagonal(&diagonal);
    //        double tmp_sum_real = 0.0;
    //        for (unsigned int idx =0; idx < diagonal.size(); idx++)
    //          tmp_sum_real += diagonal[idx].real();
    //        std::string name_of_propagator;
    //        Propagator* writeable_Propagator=NULL;
    //        PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
    //        PropInterface->get_Propagator(writeable_Propagator);
    //        double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum,writeable_Propagator);
    //        cerr << "gR_row_row*gl_row_row_temp Energy " << energy << " row " << row_index << " " <<  tmp_sum_real << " \n";
    //      }

  }


  PetscMatrixParallelComplex* Sigma_lesser_scattering = NULL;
  if(Sigma_lesser_contact!=NULL || scattering_sigmaL_solver!=NULL)
  {
    //get gA(row,row)
    PetscMatrixParallelComplex gA_row_row(gR_row_row->get_num_cols(), gR_row_row->get_num_rows(),
        gR_row_row->get_communicator());
    gR_row_row->hermitian_transpose_matrix(gA_row_row, MAT_INITIAL_MATRIX);


    if(Sigma_lesser_contact!=NULL)
    {
      PetscMatrixParallelComplex* temp_matrix3 = NULL;
      PetscMatrixParallelComplex::mult(*Sigma_lesser_contact,gA_row_row,&temp_matrix3);
      std::vector<cplx> diagonal;
      PetscMatrixParallelComplex* temp_matrix2 = NULL;
      PetscMatrixParallelComplex::mult(*gR_row_row,*temp_matrix3,&temp_matrix2);

      if(temp_result_gL == NULL)
        temp_result_gL = new PetscMatrixParallelComplex(*temp_matrix2);
      else
        temp_result_gL->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      NemoMath::symmetry_type type = NemoMath::antihermitian;
      symmetrize(this_simulation, temp_result_gL, type);
      temp_result_gL->get_diagonal(&diagonal);
      delete temp_matrix3;
      temp_matrix3 = NULL;
      delete temp_matrix2;
      temp_matrix2 = NULL;
      double tmp_sum_real = 0.0;
      for (unsigned int idx =0; idx < diagonal.size(); idx++)
        tmp_sum_real += diagonal[idx].real();
      std::string name_of_propagator;
      Propagator* writeable_Propagator=NULL;
      PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
      PropInterface->get_Propagator(writeable_Propagator);
      double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum,writeable_Propagator);
      //Harshad: Temporary fix for unused variable warning
      //please remove when using energy
      (void)energy;
      //cerr << "gR_row_row*Sigma_lesser_real_contact*gA_row_row Energy " << energy << " row " << row_index << " " <<  tmp_sum_real << " \n";
      //diagonal.real();
    }

    if(scattering_sigmaL_solver!=NULL)
    {
      if(Sigma_lesser_scattering==NULL)
        Sigma_lesser_scattering = new PetscMatrixParallelComplex(*gl_row_row_sigma_temp);
      //else
      //  Sigma_lesser_contact->add_matrix(*gl_row_row_sigma_temp,SAME_NONZERO_PATTERN,cplx(1.0,0.0));

      //also add the row row sigma

      PetscMatrixParallelComplex* sigmaL_row_row = NULL;
      std::string row_name = subdomain_names[row_index];
      std::string col_name = subdomain_names[row_index];
      PropagationUtilities::extract_sigmaR_submatrix(this_simulation, momentum, row_name, col_name, eom_interface, scattering_sigmaL_solver, sigmaL_row_row);
      sigmaL_row_row->matrix_convert_dense();
      Sigma_lesser_scattering->add_matrix(*sigmaL_row_row, SAME_NONZERO_PATTERN, cplx(1.0,0.0));



      PetscMatrixParallelComplex* temp_matrix = NULL;
      PetscMatrixParallelComplex::mult(*Sigma_lesser_scattering,gA_row_row,&temp_matrix);

      std::vector<cplx> diagonal;
      PetscMatrixParallelComplex* temp_matrix2 = NULL;
      PetscMatrixParallelComplex::mult(*gR_row_row,*temp_matrix,&temp_matrix2);
      //        {
      //          temp_matrix2->get_diagonal(&diagonal);
      //
      //          double tmp_sum_real = 0.0;
      //          for (unsigned int idx =0; idx < diagonal.size(); idx++)
      //            tmp_sum_real += diagonal[idx].real();
      //          std::string name_of_propagator;
      //          Propagator* writeable_Propagator=NULL;
      //          PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
      //          PropInterface->get_Propagator(writeable_Propagator);
      //          double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum,writeable_Propagator);
      //          cerr << "gR_row_row*Sigma_lesser_contact*gA_row_row Energy " << energy << " row " << row_index << " " <<  tmp_sum_real << " \n";
      //        }


      delete Sigma_lesser_scattering;
      Sigma_lesser_scattering = NULL;
      if(temp_result_gL == NULL)
        temp_result_gL = new PetscMatrixParallelComplex(*temp_matrix2);
      else
        temp_result_gL->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      delete temp_matrix;
      temp_matrix = NULL;
      delete temp_matrix2;
      temp_matrix2 = NULL;
    }
    delete Sigma_lesser_contact;
    Sigma_lesser_contact = NULL;
  }

  //gR_row_row*temp_result_gL
  if(temp_result_gL!=NULL)
  {
    //PetscMatrixParallelComplex::mult(*gR_row_row,*temp_result_gL,&result_gL);
    //NemoMath::symmetry_type type = NemoMath::antihermitian;
    //symmetrize(this_simulation, result_gL, type);
    result_gL = new PetscMatrixParallelComplex(*temp_result_gL);

  }



  delete temp_result_gL;
  temp_result_gL = NULL;
  //delete temp_gA_sum;
  //temp_gA_sum = NULL;
  delete gl_row_row_sigma_temp;
  gl_row_row_sigma_temp = NULL;

  //try this
  if(gl_row_row_temp!=NULL)
  {

    if(result_gL == NULL)
    {
      result_gL = new PetscMatrixParallelComplex(*gl_row_row_temp);

    }
    else
    {
      result_gL->add_matrix(*gl_row_row_temp,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
    }
  }
  delete gl_row_row_temp;
  gl_row_row_temp = NULL;

  //result_gL->matrix_convert_dense();
  NemoMath::symmetry_type type = NemoMath::antihermitian;
  symmetrize(this_simulation, result_gL, type);
  gL_container->set_block_from_matrix1(*result_gL, diagonal_indices[row_index].first, diagonal_indices[row_index].second);

  if(this_propagation->get_debug_output())
  {
    result_gL->assemble();
    result_gL->save_to_matlab_file("gl_diag_"+subdomain_names[row_index]+"_"+subdomain_names[row_index]+".m");

  }


  //     {
  //       std::vector<cplx> diagonal;
  //       result_gL->get_diagonal(&diagonal);
  //
  //       double tmp_sum_real = 0.0;
  //       for (unsigned int idx =0; idx < diagonal.size(); idx++)
  //         tmp_sum_real += diagonal[idx].real();
  //       std::string name_of_propagator;
  //       Propagator* writeable_Propagator=NULL;
  //       PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
  //       PropInterface->get_Propagator(writeable_Propagator);
  //       double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum,writeable_Propagator);
  //       cerr << "gL Energy " << energy << " row " << row_index << " " <<  tmp_sum_real << " \n";
  //     }

  NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::solve_offdiagonal_gR_nonlocal_blocks(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
             int number_offdiagonal_blocks, int row_index, int col_index, int start_index, std::vector<string> subdomain_names,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
             PetscMatrixParallelComplexContainer*& gR_container, PetscMatrixParallelComplex*& result_gr)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this_simulation->get_name() + "\")::solve_offdiagonal_gR_nonlocal_blocks: ";
  NemoUtils::tic(tic_toc_prefix);

  //Harshad: Temporary fix for unused variable warning
  //please remove when using momentum
  (void)momentum;

  Propagation* this_propagation=dynamic_cast<Propagation*>(this_simulation);
  msg.threshold(4) << "gr off  row " << row_index << " col " << col_index << " \n";
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks2 =
      gR_container->get_const_container_blocks();
  //Harshad: Temporary fix for unused variable warning
  //please remove when using gR_blocks2
  (void)gR_blocks2;

  unsigned int end_index = start_index+number_offdiagonal_blocks;//number_offdiagonal_blocks;
  if(row_index<col_index)
  {
    end_index = col_index;
  }

  PetscMatrixParallelComplex *gr_row_row_temp = NULL;
  msg.threshold(4) << "gr off start " << start_index << " end " << end_index << " \n";

  for(unsigned int j = start_index; j <= end_index; j++)
  {
    msg.threshold(4) << "gr off  j " << j << " col " << col_index << " \n";

    //find L(j,row) then transpose for L(row,j)
    std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[j].first,diagonal_indices[row_index].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >,
    PetscMatrixParallelComplex* > ::const_iterator Lblock_cit = L_blocks.find(temp_pair);
    NEMO_ASSERT(Lblock_cit != L_blocks.end(),
        tic_toc_prefix + " have not found L block \n");

    //find gR(j,col)
    std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[j].first,diagonal_indices[col_index].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >,
    PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
    NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
        tic_toc_prefix + " have not found GR off block after tranpose \n");

    PetscMatrixParallelComplex* L_j_row = Lblock_cit->second;
    PetscMatrixParallelComplex* gR_j_col = gRblock_cit->second;
    //temp_matrix = LDL(row,j)*gR(j,col)
    PetscMatrixParallelComplex* temp_matrix = NULL;
    PetscMatrixParallelComplex L_row_j(L_j_row->get_num_cols(), L_j_row->get_num_rows(),
        L_j_row->get_communicator());
    L_j_row->transpose_matrix(L_row_j, MAT_INITIAL_MATRIX);
    PetscMatrixParallelComplex::mult(L_row_j, *gR_j_col, &temp_matrix);

    if(gr_row_row_temp == NULL)
      gr_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix);
    else
    {
      gr_row_row_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
    }
    delete temp_matrix;
    temp_matrix = NULL;

  }
  std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_index].first,diagonal_indices[row_index].second);
  std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair);
  NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
      tic_toc_prefix + " have not found gR block \n");
  PetscMatrixParallelComplex* gR_row_row = gRblock_cit->second;

  PetscMatrixParallelComplex::mult(*gR_row_row,*gr_row_row_temp,&result_gr);
  result_gr->matrix_convert_dense();
  gR_container->set_block_from_matrix1(*result_gr, diagonal_indices[row_index].first, diagonal_indices[col_index].second);


  if(this_propagation->get_debug_output())
   {
     result_gr->assemble();
     result_gr->save_to_matlab_file("gr_off_"+subdomain_names[row_index]+"_"+subdomain_names[col_index]+".m");
   }
  delete gr_row_row_temp;
  gr_row_row_temp = NULL;

  msg.threshold(4) << " diagonal_indices[row_index].first " <<  diagonal_indices[row_index].first[0] << " \n";
  msg.threshold(4) << " diagonal_indices[col_index].second " <<  diagonal_indices[col_index].second[0] << " \n";
  result_gr->matrix_convert_dense();
  gR_container->set_block_from_matrix1(*result_gr, diagonal_indices[row_index].first, diagonal_indices[col_index].second);

  NemoUtils::toc(tic_toc_prefix);

}

void PropagationUtilities::solve_offdiagonal_GR_nonlocal_blocks(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
             int number_offdiagonal_blocks, int row_index, int col_index, int start_index, std::vector<string> subdomain_names,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
             PetscMatrixParallelComplexContainer*& GR_container, PetscMatrixParallelComplex*& result_Gr)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this_simulation->get_name() + "\")::solve_offdiagonal_GR_nonlocal_blocks: ";
  NemoUtils::tic(tic_toc_prefix);

  //Harshad: Temporary fix for unused variable warning
  //please remove when using momentum
  (void)momentum;

  Propagation* this_propagation=dynamic_cast<Propagation*>(this_simulation);

  msg.threshold(4) << " row " << row_index << " col " << col_index << " \n";
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
      GR_container->get_const_container_blocks();

  //this is used if we are using this function for diagonal Gr instead.
  bool use_tranpose = false;
  //int start_idx = row_index+1;
  //if(start_idx>subdomain_names.size()-1)
  //  start_idx = subdomain_names.size()-1;
  unsigned int end_index = start_index+number_offdiagonal_blocks;//number_offdiagonal_blocks;
  if(end_index>subdomain_names.size()-1)
    end_index = subdomain_names.size()-1;
  PetscMatrixParallelComplex *Gr_row_row_temp = NULL;

  for(unsigned int j = start_index; j <= end_index; j++)
  {

    //calculate Gr(row_index,j)*L(j,col_index)

    //find L(j,col_index)
    std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[j].first,diagonal_indices[col_index].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator Lblock_cit=L_blocks.find(temp_pair);
    NEMO_ASSERT(Lblock_cit != L_blocks.end(),
        tic_toc_prefix + " have not found L block \n");

    //find Gr(row_index,j)
    std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[row_index].first,diagonal_indices[j].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator GRblock_cit=GR_blocks.find(temp_pair2);
    //NEMO_ASSERT(GRblock_cit != GR_blocks.end(),
    //    tic_toc_prefix + " have not found GR off block \n");
    PetscMatrixParallelComplex* Gr_row_j = NULL;

    //if can't find Gr(row_index,j) then check
    //Gr(j,row_index) if this one exists then use it instead
    if(GRblock_cit==GR_blocks.end())
    {
      std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[j].first,diagonal_indices[row_index].second);
      GRblock_cit=GR_blocks.find(temp_pair2);
      NEMO_ASSERT(GRblock_cit != GR_blocks.end(),
          tic_toc_prefix + " have not found GR off block after tranpose \n");
      use_tranpose = true;
    }
    Gr_row_j = GRblock_cit->second;


    PetscMatrixParallelComplex* L_j_col = Lblock_cit->second;


    PetscMatrixParallelComplex *temp_matrix = NULL;
    if(!use_tranpose)
    {
      PetscMatrixParallelComplex::mult(*Gr_row_j, *L_j_col, &temp_matrix);
      if(Gr_row_row_temp == NULL)
        Gr_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix);
      else
      {
        Gr_row_row_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }

    }
    else
    {
      PetscMatrixParallelComplex L_col_j(L_j_col->get_num_cols(), L_j_col->get_num_rows(),
          L_j_col->get_communicator());
      L_j_col->transpose_matrix(L_col_j, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(L_col_j, *Gr_row_j, &temp_matrix); // Gr_row_j is really Gr_j_row
      PetscMatrixParallelComplex temp_matrix2(temp_matrix->get_num_cols(), temp_matrix->get_num_rows(),
          temp_matrix->get_communicator());
      temp_matrix->transpose_matrix(temp_matrix2, MAT_INITIAL_MATRIX);
      if(Gr_row_row_temp == NULL)
        Gr_row_row_temp = new PetscMatrixParallelComplex(temp_matrix2);
      else
      {
        Gr_row_row_temp->add_matrix(temp_matrix2,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
      }

    }
    delete temp_matrix;
    temp_matrix = NULL;


  }

  //PetscMatrixParallelComplex* Gr_row_row = NULL;
  std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[col_index].first,diagonal_indices[col_index].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair);
      NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
          tic_toc_prefix + " have not found gR block \n");
  PetscMatrixParallelComplex* gR_col_col = gRblock_cit->second;
  //PetscMatrixParallelComplex::mult(*gR_row_row,*Gr_row_row_temp,&Gr_row_row);
  delete result_Gr;
  result_Gr = NULL;
  if(!use_tranpose)
    PetscMatrixParallelComplex::mult(*Gr_row_row_temp,*gR_col_col,&result_Gr);
  else
  {

    //PetscMatrixParallelComplex::mult(*gR_col_col,*Gr_row_row_temp,&result_Gr);
    PetscMatrixParallelComplex::mult(*Gr_row_row_temp,*gR_col_col,&result_Gr);

  }

  if(row_index!=col_index)
  {
    msg.threshold(4) << " row2 " << row_index << " col2 " << col_index << " \n";
    msg.threshold(4) << " diagonal_indices[row_index].first " <<  diagonal_indices[row_index].first[0] << "size  " <<diagonal_indices[row_index].first.size()<< "\n";
    msg.threshold(4) << " diagonal_indices[col_index].second " <<  diagonal_indices[col_index].second[0] << "size  " <<  diagonal_indices[col_index].second.size()<< "\n";
    msg.threshold(4) << " resultGr row size " << result_Gr->get_num_rows() << "\n";
    msg.threshold(4) << " resultGr col size " << result_Gr->get_num_cols() << "\n";
    //result_Gr->consider_as_full();
    //possible bug in consider as full
    unsigned int num_cols = result_Gr->get_num_cols();
    unsigned int num_rows = result_Gr->get_num_rows();
    for(unsigned int i=0; i<num_rows; i++)
      result_Gr->set_num_nonzeros_for_local_row(i,num_cols,0);

    result_Gr->matrix_convert_dense();

    GR_container->set_block_from_matrix1(*result_Gr, diagonal_indices[row_index].first, diagonal_indices[col_index].second);
    //hopefully temporary
    //PetscMatrixParallelComplex temp_GR_transpose(result_Gr->get_num_cols(), result_Gr->get_num_rows(),
    //    result_Gr->get_communicator());
    //      result_Gr->transpose_matrix(temp_GR_transpose, MAT_INITIAL_MATRIX);
    //GR_container->set_block_from_matrix1(temp_GR_transpose, diagonal_indices[col_index].second, diagonal_indices[row_index].first);

    /*
    //TODO:JC fill G< with zeros
    Propagator::PropagatorMap& result_prop_map2 = lesser_green_propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it2 = result_prop_map2.find(momentum);
    PetscMatrixParallelComplex* temp_container2 = prop_it2->second;
    PetscMatrixParallelComplexContainer* GL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container2);
    PetscMatrixParallelComplex* temp_GL = new PetscMatrixParallelComplex(result_Gr->get_num_rows(),result_Gr->get_num_cols(),
        result_Gr->get_communicator());
    temp_GL->set_num_owned_rows(diagonal_indices[row_index].first.size());
    temp_GL->consider_as_full();
    for(unsigned int idx=0; idx<diagonal_indices[row_index].first.size(); idx++)
      temp_GL->set_num_nonzeros_for_local_row(idx,diagonal_indices[col_index].second.size(),0);
    temp_GL->allocate_memory();
    temp_GL->set_to_zero();
    temp_GL->assemble();
    GL_container->set_block_from_matrix1(*temp_GL, diagonal_indices[row_index].first, diagonal_indices[col_index].second);
    //PetscMatrixParallelComplex temp_GL_transpose(temp_GL->get_num_cols(), temp_GL->get_num_rows(),
    //    temp_GL->get_communicator());
    //temp_GL->transpose_matrix(temp_GL_transpose, MAT_INITIAL_MATRIX);
    //GL_container->set_block_from_matrix1(temp_GL_transpose, diagonal_indices[col_index].second, diagonal_indices[row_index].first);
    delete temp_GL;
    temp_GL = NULL;
    */
  }

  if(row_index==col_index)
  {
    NemoMath::symmetry_type type = NemoMath::symmetric;
    PropagationUtilities::symmetrize(this_simulation, result_Gr,type);
  }
  if(this_propagation->get_debug_output())
  {
    result_Gr->assemble();
    //gR_col_col->save_to_matlab_file("gr_"+subdomain_names[row_index]+"_"+subdomain_names[col_index]+".m");

    result_Gr->save_to_matlab_file("Gr_off_"+subdomain_names[row_index]+"_"+subdomain_names[col_index]+".m");
    //Gr_row_row_temp->save_to_matlab_file("Grtemp_"+subdomain_names[row_index]+"_"+subdomain_names[col_index]+".m");
  }
  //delete Gr_row_row;
  //Gr_row_row = NULL;
  delete Gr_row_row_temp;
  Gr_row_row_temp = NULL;
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::destroy_parallelization(Simulation* this_simulation,std::map<std::string, NemoPhys::Momentum_type>& momentum_mesh_types, std::map<std::string, Simulation*>& Mesh_Constructors)
{
  std::string prefix = "PropagationUtilities(\""+this_simulation->get_name()+"\")::destroy_parallelization() ";
  std::string tic_toc_name = this_simulation->get_options().get_option("tic_toc_name",this_simulation->get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+tic_toc_name+"\")::destroy_parallelization ");
  NemoUtils::tic(tic_toc_prefix);
  //----------------------------------
  //destroy MPI Hierarchy
  //----------------------------------
  std::map<string,NemoPhys::Momentum_type>::const_iterator c_it=momentum_mesh_types.begin();
  bool parallelizing_but_not_constructing_mesh=false;
  for(; c_it!=momentum_mesh_types.end(); ++c_it)
  {
    if (Mesh_Constructors.find(c_it->first)->second->parallelized_by_this())
    {
      MPIEnvironment* temp_pointer = dynamic_cast<MPIEnvironment*> (Mesh_Constructors.find(c_it->first)->second->get_mpi_variable());
      NEMO_ASSERT(temp_pointer!=NULL,prefix+"Error in casting pointer of \""
                  +Mesh_Constructors.find(c_it->first)->second->get_name()
                  +"\" to MPIEnvironment\n");
      temp_pointer->clean();
      Mesh_Constructors.find(c_it->first)->second->delete_mpi_variable();
    }
    else if(this_simulation->parallelized_by_this())
      parallelizing_but_not_constructing_mesh=true;
  }
  if(parallelizing_but_not_constructing_mesh)
  {
    MPIEnvironment* temp_pointer = dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable());
    NEMO_ASSERT(temp_pointer!=NULL,prefix+"Error in casting pointer of \""
                +this_simulation->get_mpi_variable()->get_name()+"\" to MPIEnvironment\n");
    temp_pointer->clean();
  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::load_velocity(Simulation* this_simulation, std::vector <double>*& velocity)
{
  //open the file that hold the momentum list
  fstream infile;
  velocity->clear();
  std::string momentumFileName = std::string("velocity.dat");
  infile.open(momentumFileName.c_str(),std::fstream::in);

  //check if the  file is correct
  NEMO_ASSERT(infile.is_open(),this_simulation->get_name() + " Error open file "+momentumFileName.c_str()+"\n");
  //Delimiter to separate the data in the file
  char Delimiter =',';

  //initialize the string
  std::string dataline(&Delimiter,30);
  //read number of procs from the file.
  infile.getline(&(dataline[0]),30,Delimiter);
  int count=atoi(dataline.c_str());

  //read the periodicity from the file
  //the periodicity is equal to number of equivelant nodes.
  dataline.resize(30,Delimiter);
  infile.getline(&(dataline[0]),30,Delimiter);
  int period=atoi(dataline.c_str());

  double temp;
  velocity->resize(count*period,0);
  //read the velocity of each procs from the file
  for(int i=0; i<count; i++)
  {
    dataline.resize(30,Delimiter);
    //read velocity from the file one by one.
    infile.getline(&(dataline[0]),30,Delimiter);
    temp=atof(dataline.c_str());
    (*velocity)[i]=temp;

  }
  //if the file for single node copy the first node for the rest of nodes
  if(period>1)
  {
    for(int i=0; i<(period-1); i++)
    {
      for(int j=0; j<count; j++)
      {
        (*velocity)[count+i*(count)+j]=(*velocity)[j];
      }
    }
  }
}


void PropagationUtilities::reset_MPI_parallelization(Simulation* this_simulation,std::map<string,NemoPhys::Momentum_type>& momentum_mesh_types,std::map<std::string, NemoMesh*>& Momentum_meshes,
                                                     std::map<std::string, Simulation*>& Mesh_Constructors,std::map<NemoMesh*,std::vector<NemoMesh*> >& Mesh_tree_topdown,
                                                     std::vector<std::string>& Mesh_tree_names, std::map<NemoMesh*, NemoMesh* >& Mesh_tree_downtop)
{

  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization ");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(NemoUtils::MsgLevel(4));

  std::string tic_toc_prefix1 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 1 ");
  NemoUtils::tic(tic_toc_prefix1);

  std::string prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization: ";
  unsigned int number_of_momenta=momentum_mesh_types.size();

  std::vector<bool> reuse_parallel_treeV;
  if (this_simulation->get_options().check_option("reuse_parallel_hierarchy"))
    this_simulation->get_options().get_option("reuse_parallel_hierarchy",reuse_parallel_treeV);
  else
    reuse_parallel_treeV.resize(number_of_momenta,false);
  NEMO_ASSERT(reuse_parallel_treeV.size()==number_of_momenta,prefix+"dimension of reuse_parallel_treeV does not agree with number of momenta\n");
  unsigned int reuse_counter=0;
  for (unsigned int i=0; i<number_of_momenta; i++)
    if (reuse_parallel_treeV[i])
      reuse_counter+=1;
  NEMO_ASSERT(reuse_counter<number_of_momenta,prefix+"have at least one element of reuse_parallel_hierarchy set to \"false\"\n");
  NemoUtils::toc(tic_toc_prefix1);

  ////----------------------------------
  ////destroy MPI Hierarchy
  ////----------------------------------
  std::string tic_toc_prefix2 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 2 ");
  NemoUtils::tic(tic_toc_prefix2);
  PropagationUtilities::destroy_parallelization(this_simulation,momentum_mesh_types, Mesh_Constructors);
  std::map<string,NemoPhys::Momentum_type>::const_iterator c_it=momentum_mesh_types.begin();
  NemoUtils::toc(tic_toc_prefix2);

  //----------------------------------
  //set up a new MPI-Hierarchy, where required
  //----------------------------------

  std::string tic_toc_prefix3 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 3");
  NemoUtils::tic(tic_toc_prefix3);
  unsigned int top_most_counter=0; // how many top-most meshes do exist?
  std::string parallel_parent;
  c_it=momentum_mesh_types.begin();
  for(; c_it!=momentum_mesh_types.end(); ++c_it)
  {
    {
      std::string variable_name=c_it->first+std::string("_parent");
      parallel_parent=this_simulation->get_options().get_option(variable_name,std::string("none")); //no dependence per default
      if (parallel_parent!="none" && parallel_parent!=c_it->first)
      {
        msg << prefix<< "momentum \""<<c_it->first<<"\" is a parallel child of momentum \"" << parallel_parent <<"\"\n";
        NemoMesh* temp_parent=Momentum_meshes.find(parallel_parent)->second;
        NEMO_ASSERT(temp_parent!=NULL,prefix+"have received NULL for mesh \""+parallel_parent+"\"\n");
        for (unsigned int j=0; j<temp_parent->get_num_points(); j++)
        {
          NemoMesh* parent = Momentum_meshes.find(parallel_parent)->second;
          NemoMesh* child  = Momentum_meshes.find(c_it->first)->second;
          /*if(child->get_parent()!=NULL)
            if(parent->get_name()!=child->get_parent()->get_name())
            {*/
          //NEMO_ASSERT(child->get_parent()==NULL||child->get_parent()!=parent,prefix+"problems with MPIvariable parenthood\n");
          parent->set_child(j,child);
          child->set_parent(parent);
          //}
          std::map<NemoMesh*,std::vector<NemoMesh*> >::iterator temp_it=Mesh_tree_topdown.find(parent);
          if(temp_it!=Mesh_tree_topdown.end())
          {
            //temp_it->second.insert(child);
            NEMO_ASSERT(temp_it->second.size()==parent->get_num_points(),prefix+"mismatch in number of children\n");
            temp_it->second[j]=child;
          }
          else
          {
            Mesh_tree_topdown[parent]=std::vector<NemoMesh*>(parent->get_num_points(),child);
            //Propagation::Mesh_tree_topdown[parent].insert(child);
          }
          if(Mesh_tree_downtop.find(child)!=Mesh_tree_downtop.end())
            NEMO_ASSERT(Mesh_tree_downtop.find(child)->second==parent,prefix+"parent mesh of \""+child->get_name()+"\" is ambiguously set\n");
          Mesh_tree_downtop[child]=parent;
        }
      }
      else
      {
        NemoMesh* child  = Momentum_meshes.find(c_it->first)->second;
        top_most_counter+=1;
        msg << prefix << "momentum \""<<c_it->first<<"\" is a top-most parallel object"<<std::endl;
        NEMO_ASSERT(child!=NULL,prefix+"momentum \""+c_it->first+"\" is not ready for parallelization.\n");
        dynamic_cast<MPIEnvironment*>(this_simulation->get_mpi_variable())->set_variable(child);
        if(Mesh_tree_downtop.find(child)!=Mesh_tree_downtop.end())
          NEMO_ASSERT(Mesh_tree_downtop.find(child)->second==NULL,prefix+"parent mesh of \""+child->get_name()+"\" is ambiguously set\n");
        Mesh_tree_downtop[child]=NULL;
        if(momentum_mesh_types.size()==1)
          Mesh_tree_topdown[child];//just produce the key, if there is only on mesh given
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix3);



  std::string tic_toc_prefix4 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 4");
  NemoUtils::tic(tic_toc_prefix4);

  c_it=momentum_mesh_types.begin();
  for(; c_it!=momentum_mesh_types.end(); ++c_it)
  {
    NemoMesh* this_mesh_pointer = Momentum_meshes.find(c_it->first)->second;
    NEMO_ASSERT(this_mesh_pointer!=NULL, prefix+"have not found momentum \""+c_it->first+"\"\n");
    if (top_most_counter>1)
    {
      //put all top most MPI-Variables underneath a single faked one
      MPIVariable* fake_top_most_MPIVariable = new MPIVariable(std::string("fake_top_most_MPIVariable"));
      fake_top_most_MPIVariable->set_parent(this_simulation->get_mpi_variable());
      dynamic_cast<MPIEnvironment*>(this_simulation->get_mpi_variable())->set_variable(fake_top_most_MPIVariable);
      fake_top_most_MPIVariable->set_num_points(top_most_counter);
      for (unsigned int i=0; i<number_of_momenta; i++)
      {
        unsigned int temp_counter=0;
        if(this_mesh_pointer->get_parent()==this_simulation->get_mpi_variable())
        {
          fake_top_most_MPIVariable->set_child(temp_counter,Momentum_meshes.find(c_it->first)->second);
          temp_counter+=1;
        }
      }
      std::string strategy=this_simulation->get_options().get_option("top_most_MPI_strategy",std::string("cluster"));
      if(strategy=="Scatter" ||strategy=="scatter" ||strategy=="SCATTER")
        fake_top_most_MPIVariable->set_strategy(MPIVariable::scatter);
      else if(strategy=="Cluster" ||strategy=="cluster" ||strategy=="CLUSTER")
        fake_top_most_MPIVariable->set_strategy(MPIVariable::cluster);
      else
        throw std::invalid_argument(std::string(prefix+"unknown parallelization strategy found for \"top_most_MPI_strategy\"\n"));

      fake_top_most_MPIVariable->set_granularity(1);
    }
    else if(top_most_counter<1)
      throw std::runtime_error(prefix+"no parallel top-most mesh found\n");
    else {}
    //----------------------------------
    // set the granularity for momentum[i]
    //----------------------------------
    std::string variable_name=c_it->first + std::string("_granularity");
    unsigned int ui_temp=1;
    if(this_simulation->get_options().check_option(variable_name))
      ui_temp = this_simulation->get_options().get_option(variable_name,1);
    else
    {
      //check for real space parallelization and set ui_temp TODO: this is a temporary solution only to make regtest run fine!
      if (dynamic_cast<Schroedinger*>(this_simulation)!=NULL)
        ui_temp=this_simulation->get_const_simulation_domain()->get_geometry_comm_size();
      msg <<"[PropagationUtilities] using granularity " <<ui_temp<<" for momentum \"" << c_it->first << "\"\n";
    }
    this_mesh_pointer->set_granularity(ui_temp);
    //----------------------------------
    // set the strategy for momentum[i]
    //----------------------------------
    variable_name=c_it->first + std::string("_strategy");
    std::string strategy=this_simulation->get_options().get_option(variable_name,std::string("Scatter"));
    msg<<"[PropagationUtilities] using strategy "<<strategy<< " for momentum \"" << c_it->first << "\"\n";
    if(strategy=="Scatter" ||strategy=="scatter" ||strategy=="SCATTER")
    {
      this_mesh_pointer->set_strategy(MPIVariable::scatter);
    }
    else if(strategy=="simple_scatter")
    {
      this_mesh_pointer->set_strategy(MPIVariable::simple_scatter);
    }
    else if(strategy=="Cluster" ||strategy=="cluster" ||strategy=="CLUSTER")
    {
      this_mesh_pointer->set_strategy(MPIVariable::cluster);
    }
    else
    {
      throw std::invalid_argument(std::string(prefix+"unknown parallelization strategy found for "+c_it->first+"\n"));
    }
  }
  NemoUtils::toc(tic_toc_prefix4);

  //----------------------------------------------------------------------------
  //-----------------------------------
  // create multiple of NemoMeshes according to the hierarchy defined above
  //-----------------------------------
  //1. loop over all Meshes in the hierarchal order and fill a slim_Mesh_tree (one instance per tree level
  //find the top-most NemoMesh
  std::string tic_toc_prefix5 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 5");
  NemoUtils::tic(tic_toc_prefix5);

  NemoMesh* topmost_Mesh=NULL;
  std::map<NemoMesh*,NemoMesh* >::iterator temp_it=Mesh_tree_downtop.begin();
  for(; temp_it!=Mesh_tree_downtop.end(); ++temp_it)
  {
    if(temp_it->second==NULL)
    {
      NEMO_ASSERT(topmost_Mesh==NULL,prefix+"found more than one topmost mesh\n");
      topmost_Mesh=temp_it->first;
    }
  }
  NemoUtils::toc(tic_toc_prefix5);


  std::string tic_toc_prefix6 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 6");
  NemoUtils::tic(tic_toc_prefix6);
  //store the Mesh hierarchy with a single branch per level in slim_Mesh_tree
  std::vector<NemoMesh*> slim_Mesh_tree(Mesh_tree_downtop.size());
  Mesh_tree_names.resize(Mesh_tree_downtop.size());
  slim_Mesh_tree[0]=topmost_Mesh;
  Mesh_tree_names[0]=topmost_Mesh->get_name();
  for(unsigned int i=1; i<slim_Mesh_tree.size(); i++)
  {
    std::map<NemoMesh*,std::vector<NemoMesh*> >::iterator it2=Mesh_tree_topdown.find(slim_Mesh_tree[i-1]);
    NEMO_ASSERT(it2!=Mesh_tree_topdown.end(),prefix+"lost track of NemoMesh \""+slim_Mesh_tree[i-1]->get_name()+"\"\n");
    slim_Mesh_tree[i]=it2->second[0];
    Mesh_tree_names[i]=it2->second[0]->get_name();
  }
  NemoUtils::toc(tic_toc_prefix6);


  //(Bozidar) Hack for transverse k parallelization. Instead of doing 1D band structure multiple times in serial, do 2D or 3D once by using
  //the existing Schroedinger parallelization.
  if(this_simulation->get_options().get_option("parallelize_adaptive_grid_construction",false))
  {
    bool non_rectangular_mesh = false;
    Simulation* mesh_constructor = NULL;
    NemoMesh* first_child = NULL;
    std::set<NemoMesh*>::iterator this_level_it;
    std::set<NemoMesh*> momentum_meshes;

    for(unsigned int i = 0; i < slim_Mesh_tree.size() - 1; i++)
    {
      NemoMesh* parent_mesh_pointer=slim_Mesh_tree[i];
      std::string parent_mesh_name=parent_mesh_pointer->get_name();
      std::set<NemoMesh*> Meshes_of_same_hierarchy;
      Meshes_of_same_hierarchy.insert(parent_mesh_pointer);
      std::map<NemoMesh*,NemoMesh* >::const_iterator c_it=
        Mesh_tree_downtop.begin(); //Mesh_tree_downtop contains an up-to-date list of all Meshes
      for(; c_it!=Mesh_tree_downtop.end(); ++c_it)
        if(parent_mesh_name==c_it->first->get_name())
          Meshes_of_same_hierarchy.insert(c_it->first);
      this_level_it=Meshes_of_same_hierarchy.begin();

      if(slim_Mesh_tree[i+1]->get_name().find("energy") != std::string::npos)
      {
        first_child = slim_Mesh_tree[i+1];
        std::map<std::string, Simulation*>::const_iterator c_it = Mesh_Constructors.find(first_child->get_name());
        mesh_constructor = c_it->second;
        NEMO_ASSERT(mesh_constructor != NULL, prefix + "received NULL as constructor of \"" + (*this_level_it)->get_name() + "\"\n");
        InputOptions& mesh_constructor_options = mesh_constructor->get_reference_to_options();
        non_rectangular_mesh = mesh_constructor_options.get_option("non_rectangular",false);
        momentum_meshes = Meshes_of_same_hierarchy;
        break;
      }
    }
    NEMO_ASSERT(mesh_constructor != NULL, prefix + "received NULL as constructor of \"" + (*this_level_it)->get_name() + "\"\n");
    if(non_rectangular_mesh)
    {
      std::vector<NemoMeshPoint> momentum_points;

      NEMO_ASSERT(momentum_meshes.size() == 1, prefix + "For parallel construction of adaptive energy grid currently only one momentum NemoMesh on the" +
                  "same hierarchy level supported\n");
      std::set<NemoMesh*>::iterator it;
      for(it = momentum_meshes.begin(); it != momentum_meshes.end(); it++)
      {
        std::vector<NemoMeshPoint*>::iterator it2;
        for (it2 = (*it)->get_mesh_points().begin(); it2 != (*it)->get_mesh_points().end(); it2++)
          momentum_points.push_back(**it2);
      }

      std::map<NemoMeshPoint, NemoMesh*> new_children;
      mesh_constructor->get_data("e_space_"+first_child->get_name(), momentum_points, new_children);
    }
  }

  std::string tic_toc_prefix7 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 7");
  NemoUtils::tic(tic_toc_prefix7);
  std::vector<NemoMeshPoint> current_mesh_tupel;
  //2.loop over the full slim Mesh tree hierarchy (the lowest level does not have children)
  for(unsigned int j=0; j<slim_Mesh_tree.size()-1; j++)
  {
    NemoMesh* parent_mesh_pointer=slim_Mesh_tree[j];
    std::string parent_mesh_name=parent_mesh_pointer->get_name();
    NemoMesh* first_child=slim_Mesh_tree[j+1];
    //gather all Meshes that are siblings of parent_mesh_pointer (they have the same name) including parent_mesh_pointer itself
    std::set<NemoMesh*> Meshes_of_same_hierarchy;
    Meshes_of_same_hierarchy.insert(parent_mesh_pointer);
    std::map<NemoMesh*,NemoMesh*>::const_iterator c_it=
      Mesh_tree_downtop.begin();//Mesh_tree_downtop contains an up-to-date list of all Meshes
    for(; c_it!=Mesh_tree_downtop.end(); ++c_it)
      if(parent_mesh_name==c_it->first->get_name())
        Meshes_of_same_hierarchy.insert(c_it->first);

    //3.loop over all meshes with the same name/hierarchy level
    std::set<NemoMesh*>::iterator this_level_it=Meshes_of_same_hierarchy.begin();
    for(; this_level_it!=Meshes_of_same_hierarchy.end(); ++this_level_it)
    {
      std::map<NemoMesh*,std::vector<NemoMesh*> >::iterator topdown_it=Mesh_tree_topdown.find(*this_level_it);
      if(topdown_it==Mesh_tree_topdown.end())
      {
        //add to Mesh_tree_topdown the required entry
        unsigned int number_of_points=(*this_level_it)->get_num_points();
        Mesh_tree_topdown[*this_level_it]=std::vector<NemoMesh*>(number_of_points,NULL);
        topdown_it=Mesh_tree_topdown.find(*this_level_it);
      }

      //4. create as many subMeshes as points in the parent mesh and set the new meshes as children of the respective parent
      std::vector<NemoMeshPoint*> temp_points=(*this_level_it)->get_mesh_points();
      NEMO_ASSERT(temp_points.size()==(*this_level_it)->get_num_points(),
                  prefix+"inconsistent number of points in mesh \""+(*this_level_it)->get_name()+"\"found\n");
      for(unsigned int i=0; i<(*this_level_it)->get_num_points(); i++)
      {
        //if(i!=0 || *this_level_it!=parent_mesh_pointer) //i==0&&*this_level_it==parent_mesh_pointer gives the already existing first_child
        {
          //4.1 get the NemoMeshPoint of this i
          NEMO_ASSERT(i<temp_points.size(),prefix+"inconsistent number of mesh points for \""+(*this_level_it)->get_name()+"\"\n");
          NemoMeshPoint temp_point = *(temp_points[i]); //(*this_level_it)->get_point(i);

          ////4.1.1 add that coordinate to the current mesh point tupel
          //if(current_mesh_tupel.size()<(*this_level_it)->get_num_points())
          //  current_mesh_tupel.resize((*this_level_it)->get_num_points(),temp_point);
          //else
          //  current_mesh_tupel[i]=temp_point;

          //NOTE: we use type NemoMesh for the moment. It might become necessary to distinguish between Espace and Kspace...
          //4.1.1 find the mesh constructor
          std::map<std::string, Simulation*>::const_iterator c_it=Mesh_Constructors.find(first_child->get_name());
          NEMO_ASSERT(c_it!=Mesh_Constructors.end(),prefix+"have not found mesh constructor of \""+(*this_level_it)->get_name()+"\"\n");
          Simulation* mesh_constructor=c_it->second;

          NEMO_ASSERT(mesh_constructor!=NULL,prefix+"received NULL as constructor of \""+(*this_level_it)->get_name()+"\"\n");
          //4.1 create or get the new mesh
          NemoMesh* new_child=NULL;
          InputOptions& mesh_constructor_options = mesh_constructor->get_reference_to_options();

          bool non_rectangular_mesh=mesh_constructor_options.get_option("non_rectangular",false);
          if(non_rectangular_mesh)
          {

            //4.1.2 find the momentum tupel for this mesh branch
            std::vector<NemoMeshPoint> temp_momentum_tupel(j+1,temp_point);
            std::vector<bool> delete_from_temp_momentum_tupel(j+1,false);
            //4.1.2.1 figure out which meshes to include in the get_data call
            std::vector<std::string> temp_list_of_extra_momenta;
            this_simulation->get_options().get_option("Hamilton_momenta",temp_list_of_extra_momenta);
            std::set<std::string> list_of_extra_momenta;
            for(unsigned int ii=0; ii<temp_list_of_extra_momenta.size(); ii++)
              list_of_extra_momenta.insert(temp_list_of_extra_momenta[ii]);

            NemoMesh* temp_mesh=(*this_level_it);
            NemoMesh* parent=NULL;
            temp_momentum_tupel[j]=temp_point;

            for(int tupel_i=j-1; tupel_i>=0; tupel_i--)
            {
              //4.1.2.1 find the grand parent
              parent=dynamic_cast<NemoMesh*>(temp_mesh->get_parent());
              NEMO_ASSERT(parent!=NULL,prefix+"have not found parent of \""+temp_mesh->get_name()+"\"\n");
              //4.1.2.2 find the index of the parent in the grand parent mesh
              unsigned int index=parent->get_child_index(temp_mesh);
              //4.1.2.3 find the NemoMeshPoint of the index
              NemoMeshPoint higher_temp_point=parent->get_point(index);
              //4.1.2.4 store the point in the appropriate temp_momentum_tupel slot
              temp_momentum_tupel[tupel_i]=higher_temp_point;
              //4.1.2.5 check whether the parent is listed in list_of_extra_momenta
              std::set<std::string>::const_iterator list_cit=list_of_extra_momenta.find(parent->get_name());
              if(list_cit==list_of_extra_momenta.end())
                delete_from_temp_momentum_tupel[tupel_i]=true;
              //4.1.2.6 store the parent in temp_mesh (for next iteration)
              temp_mesh=parent;
            }
            //delete those entries of temp_momentum_tupel that correspond to meshes not in list_of_extra_momenta
            std::vector<NemoMeshPoint> temp_momentum_tupel2;
            for(unsigned int tupel_i=0; tupel_i<temp_momentum_tupel.size(); tupel_i++)
            {
              if(!delete_from_temp_momentum_tupel[tupel_i])
                temp_momentum_tupel2.push_back(temp_momentum_tupel[tupel_i]);
            }
            temp_momentum_tupel=temp_momentum_tupel2;

            //4.1.3 get the mesh from the mesh constructor
            if(first_child->get_name().find("energy")!=std::string::npos)
            {
              mesh_constructor->get_data("e_space_"+first_child->get_name(),temp_momentum_tupel,new_child);
            }
            else
            {
              //Fabio: non rectangular mesh is allowed for energy or k only
              //currently k mesh is identical for different valley
              NemoPhys::Momentum_type temp_child_mesh_name = momentum_mesh_types.find(first_child->get_name())->second;
              NEMO_ASSERT(temp_child_mesh_name==NemoPhys::Momentum_1D||temp_child_mesh_name==NemoPhys::Momentum_2D||temp_child_mesh_name==NemoPhys::Momentum_3D, prefix+"non rectangular mesh is allowed for energy or k only \n");
              this_simulation->get_data("k_space_"+first_child->get_name(),new_child);

              //new_child = Momentum_meshes.find(first_child->get_name())->second;
              //NemoMesh* temp_mesh2=NULL;
              //mesh_constructor->get_data(first_child->get_name(),temp_momentum_tupel,new_child);
              //mesh_constructor->get_data(first_child->get_name(),temp_momentum_tupel,temp_mesh2);
              //new_child = new NemoMesh(*temp_mesh2);
              //delete temp_mesh2;
              //produced_a_mesh=true;
            }
            new_child->set_name(first_child->get_name()); //should have been done by the mesh constructor - but who knows...
            std::string scheme_color=this_simulation->get_options().get_option("parallelization_scheme",std::string("top_down"));
            if(scheme_color=="color_sort" ||scheme_color=="Color_Sort" ||scheme_color=="COLOR_SORT"||scheme_color=="color_sort_velocity" ||scheme_color=="Color_Sort_Velocity" ||scheme_color=="COLOR_SORT_VELOCITY")
            {
              new_child->set_color(this_simulation->get_options().get_option("color_cord_index",0)  
                ,this_simulation->get_options().get_option("color_rank_scale",100000));
            }
          }
          else
          {
            new_child = new NemoMesh(); //NOTE_E(k): this has to change if we want to have meshpoint dependent child meshes
            //copy the attributes of first_child to its siblings
            new_child->set_name(first_child->get_name());
            new_child->set_num_points(first_child->get_num_points());
            new_child->set_granularity(first_child->get_granularity());
            new_child->set_strategy(first_child->get_strategy());

            //4.2 set the coordinates of the new_child taken from first_child
            std::vector<NemoMeshPoint*> temp_points=first_child->get_mesh_points();
            std::vector<std::vector<double> > coordinates(temp_points.size());
            for(unsigned int ii=0; ii<temp_points.size(); ii++)
            {
              coordinates[ii]=temp_points[ii]->get_coords();
            }
            NEMO_ASSERT(new_child->get_num_points()==coordinates.size(),
                        prefix+"mismatch in get_num_points for mesh \""+new_child->get_name()+"\"\n");
            new_child->set_mesh(coordinates);
            Propagation* temp_propagation=dynamic_cast<Propagation*> (this_simulation);
            if(temp_propagation!=NULL)
              temp_propagation->set_produced_a_mesh(true);

            //correct the slim_Mesh_tree
            if(j<slim_Mesh_tree.size()-1&&i==0)
            {
              std::map<NemoMesh*, NemoMesh* >::iterator tempit=Mesh_tree_downtop.find(slim_Mesh_tree[j+1]);
              if(tempit!=Mesh_tree_downtop.end())
                Mesh_tree_downtop.erase(tempit);
              slim_Mesh_tree[j+1]=new_child;
            }
          }

          Mesh_tree_downtop[new_child]=*this_level_it;

          //4.3 enter the new child into the mesh tree
          topdown_it->second[i]=new_child;
          (*this_level_it)->set_child(i,new_child);
          //(*this_level_it)->set_load(i,new_child->get_num_points());
          new_child->set_parent(*this_level_it);
        }
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix7);
  //----------------------------------------------------------------------------
  std::string tic_toc_prefix8 = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::reset_MPI_parallelization 8");
  NemoUtils::tic(tic_toc_prefix8);
  msg << prefix << "parallelizing\n";
  std::string scheme=this_simulation->get_options().get_option("parallelization_scheme",std::string("top_down"));
  if(scheme=="Top_down" ||scheme=="top_down" ||scheme=="TOP_DOWN")
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_scheme(MPIVariable::TOP_DOWN);
  else if(scheme=="Bottom_up" ||scheme=="bottom_up" ||scheme=="BOTTOM_UP")
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_scheme(MPIVariable::BOTTOM_UP);
  else if(scheme=="sort" ||scheme=="Sort" ||scheme=="SORT")
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_scheme(MPIVariable::SORT);
  else if(scheme=="color_sort" ||scheme=="Color_Sort" ||scheme=="COLOR_SORT")
     dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_scheme(MPIVariable::COLOR_SORT);
  else if(scheme=="velocity" ||scheme=="Velocity" ||scheme=="VELOCITY")
  {
    // the vector to hold processors velocity
    std::vector<double>* velocity;
    std::vector<double>  velocity_obj;
    velocity=& velocity_obj;
    //load the velocities from file
    PropagationUtilities::load_velocity(this_simulation,velocity);
    //set the velocity in the MPIVariable class.
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_scheme(MPIVariable::VELOCITY);
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_velocity(velocity);
  }
  else if(scheme=="velocity_sort" ||scheme=="Velocity_Sort" ||scheme=="VELOCITY_SORT")
  {
    // the vector to hold processors velocity
    std::vector<double>* velocity;
    std::vector<double>  velocity_obj;
    velocity=& velocity_obj;
    //load the velocities from file
    PropagationUtilities::load_velocity(this_simulation,velocity);
    //set the velocity in the MPIVariable class.
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_scheme(MPIVariable::VELOCITY_SORT);
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_velocity(velocity);
  }
  else if(scheme=="color_sort_velocity" ||scheme=="Color_Sort_Velocity" ||scheme=="COLOR_SORT_VELOCITY")
   {
     // the vector to hold processors velocity
     std::vector<double>* velocity;
     std::vector<double>  velocity_obj;
     velocity=& velocity_obj;
     //load the velocities from file
     PropagationUtilities::load_velocity(this_simulation,velocity);
     //set the velocity in the MPIVariable class.
     dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_scheme(MPIVariable::COLOR_SORT_VELOCITY);
     dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_velocity(velocity);
   }
  else
    throw std::invalid_argument(std::string(prefix+" Unknown parallelization scheme \n"));
  if (this_simulation->get_options().check_option("parallelization_mpi_load"))
  {
    vector<vector<unsigned int> > mpi_loads;
    map<unsigned int, unsigned int> mpi_map;
    this_simulation->get_options().get_option("parallelization_mpi_load", mpi_loads);
    vector<vector<unsigned int> >::iterator it_vec = mpi_loads.begin();
    for(; it_vec < mpi_loads.end(); it_vec++)
    {
      vector<unsigned int> temp_vec = *it_vec;
      NEMO_ASSERT(temp_vec.size() == 2,prefix+" parallelization_mpi_load option needs to be a vector of tuples\n");
      mpi_map[temp_vec[0]] = temp_vec[1];
    }
    dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->set_mpi_load(mpi_map);
  }

  dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->parallelize();
  this_simulation->set_parallelized_by_this(true);
  dynamic_cast<MPIEnvironment*> (this_simulation->get_mpi_variable())->display_info();
  NemoUtils::toc(tic_toc_prefix8);
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::set_parallel_environment(Simulation* this_simulation, std::vector<std::string>& Mesh_tree_names, 
      std::map<NemoMesh*, NemoMesh* >& Mesh_tree_downtop,std::map<NemoMesh*,std::vector<NemoMesh*> >& Mesh_tree_topdown,
      std::map<std::string, NemoMesh*>& Momentum_meshes,std::map<std::string, NemoPhys::Momentum_type>& momentum_mesh_types,
      std::map<std::string, Simulation*>& Mesh_Constructors,Simulation* Parallelizer,const MPI_Comm& input_communicator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::set_parallel_environment ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="PropagationUtilities(\""+this_simulation->get_name()+"\")::set_parallel_environment: ";

  //----------------------------------
  //replace all Momentum_meshes NULL pointers with NemoMesh*
  //----------------------------------
  std::map<std::string, NemoMesh*>::iterator it=Momentum_meshes.begin();
  for (; it!=Momentum_meshes.end(); ++it)
  {
    std::string temp_name=it->first;
    if(it->second==NULL)
    {
      if (momentum_mesh_types.find(temp_name)->second==NemoPhys::Momentum_1D||momentum_mesh_types.find(temp_name)->second==NemoPhys::Momentum_2D||
          momentum_mesh_types.find(temp_name)->second==NemoPhys::Momentum_3D)
      {
        Mesh_Constructors.find(temp_name)->second->get_data("k_space_"+temp_name,it->second);
      }
      else if (momentum_mesh_types.find(temp_name)->second==NemoPhys::Valley)
      {
        Mesh_Constructors.find(temp_name)->second->get_data("v_space_"+temp_name,it->second);
      }
      else if(momentum_mesh_types.find(temp_name)->second==NemoPhys::Energy||momentum_mesh_types.find(temp_name)->second==NemoPhys::Complex_energy)
      {
        InputOptions& mesh_options=Mesh_Constructors.find(temp_name)->second->get_reference_to_options();
        if(!mesh_options.get_option(std::string("non_rectangular"),false))
          //if(!options.get_option("non_rectangular_"+temp_name,false))
          Mesh_Constructors.find(temp_name)->second->get_data("e_space_"+temp_name,it->second);
        else
        {
          //call this first mesh with the Gamma point...
          std::vector<std::string> list_of_extra_momenta;
          this_simulation->get_options().get_option("Hamilton_momenta",list_of_extra_momenta);
          std::vector<double> temp_vector(3,0.0);
          NemoMeshPoint temp_momentum(0,temp_vector);
          std::vector<NemoMeshPoint> temp_vector2(list_of_extra_momenta.size(),temp_momentum);
          for(unsigned int i=0; i<list_of_extra_momenta.size(); i++)
          {
            //2. create fake mesh points (such as the k=0 point to do a first get_data call
            std::map<std::string, NemoPhys::Momentum_type>::const_iterator cit=momentum_mesh_types.find(list_of_extra_momenta[i]);
            NEMO_ASSERT(cit!=momentum_mesh_types.end(),prefix+"have not found momentum \""+list_of_extra_momenta[i]+"\" in momentum_mesh_types\n");
            if(cit->second==NemoPhys::Momentum_1D||cit->second==NemoPhys::Momentum_2D||cit->second==NemoPhys::Momentum_3D)
            {
              //call this first mesh with the Gamma point...
              //setup done already in the definition of the temp_vector2
            }
            else if(cit->second==NemoPhys::Valley)
            {
              std::vector<double> temp_vectorV(1,0.0);
              NemoMeshPoint temp_momentumV(0,temp_vectorV);
              temp_vector2[i]=temp_momentumV;
            }
            else
              throw std::invalid_argument(prefix+"called with non-supported momentum \""+cit->first+"\"\n");
          }
          Mesh_Constructors.find(temp_name)->second->get_data("e_space_"+temp_name,temp_vector2,it->second);
        }
      }
      else
        throw std::invalid_argument(prefix+"there are only Energy, Momentum and Valley implemented so far\n");
    }
    NEMO_ASSERT(it->second!=NULL,prefix+"cannot get mesh \""+temp_name+"\" from "+Mesh_Constructors.find(temp_name)->second->get_name()+"\n");
    /*std::cerr<<prefix+"set mesh: "<<temp_name<<"\n";*/
    it->second->set_name(temp_name);
  }

  if (Parallelizer!=this_simulation)
  {
    //DM force propagation to start/ I think this need to be re-evaluated since 'network concept' also needs to include parallelizer propagation
    if(!Parallelizer->parallelized_by_this() && Parallelizer->get_options().get_option("force_start_if_required",false))
    {
      Propagation* PParallelizer = dynamic_cast<Propagation*>(Parallelizer);
      if(PParallelizer)
        PParallelizer->initialize_Propagation();//set_parallel_environment();
    }
    NEMO_ASSERT(Parallelizer->parallelized_by_this(),prefix+"Parallelizer "+ Parallelizer->get_name() + " has not parallelized\n");
    NEMO_ASSERT(Parallelizer->get_mpi_variable()->mpi_is_ready(),
                prefix+"Parallelizer's top most MPI-Variable is not ready (parallelize() not executed)\n");
    //get the mesh-tree related data (former static variables from the Parallelizer)
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  else
  {
    //M.P.
    this_simulation->set_simulation_communicator(input_communicator);

    //delete _top_mpi_variable;
    this_simulation->delete_mpi_variable();
    this_simulation->set_mpi_variable(new MPIEnvironment(input_communicator));

    /*NemoMesh* temp_parent2=Momentum_meshes.find("momentum_1D")->second;
    NEMO_ASSERT(temp_parent2!=NULL,"have received NULL for mesh momentum_1D\n");*/
    //reset_MPI_parallelization();
    PropagationUtilities::reset_MPI_parallelization(this_simulation,momentum_mesh_types,Momentum_meshes,Mesh_Constructors,Mesh_tree_topdown,
                                                    Mesh_tree_names,Mesh_tree_downtop);

  }
  NemoUtils::toc(tic_toc_prefix);
}

void PropagationUtilities::prepare_communicator_for_parallelization(Simulation* this_simulation, const MPI_Comm& input_communicator, MPI_Comm& result_communicator, 
      const bool calculate_bandstructure, const unsigned int granularity, const unsigned int bandstructure_points)
{
  const InputOptions& options=this_simulation->get_options();
  int local_rank, local_size;
  MPI_Comm_rank(input_communicator, &local_rank);
  MPI_Comm_size(input_communicator, &local_size);
  if (local_size%granularity != 0)
    throw invalid_argument("[PropagationUtilities(\""+this_simulation->get_name()+"\")] Number of cores needs to be a multiple of granularity\n");
  //check the number of geometry partitioning
  MPI_Comm temp_communicator=this_simulation->get_const_simulation_domain()->get_communicator();
  int temp_size;
  MPI_Comm_size(temp_communicator,&temp_size);
  if(granularity==1 && calculate_bandstructure && temp_size==1 && options.get_option("enable_hack",true))
  {
    //check whether the number of CPUs is larger than the number_of_nodes (for bandstructure calulations)
    if(local_size>(const int) bandstructure_points)
    {
      MPI_Comm_split(input_communicator, local_rank/bandstructure_points, local_rank%bandstructure_points, &result_communicator);
    }
    else
      MPI_Comm_split(input_communicator, local_rank%granularity, local_rank/granularity, &result_communicator);
  }
  else
    MPI_Comm_split(input_communicator, local_rank%granularity, local_rank/granularity, &result_communicator);
  
}

void PropagationUtilities::integrate_diagonal(Simulation* this_simulation, Simulation* source_of_data, Simulation* Parallelizer, std::map<std::string, Simulation*>& Mesh_Constructors,
                                              NemoPhys::Propagator_type input_type, const std::string& data_name, bool get_energy_resolved_data, 
                                              bool get_k_resolved_data, bool& get_energy_resolved_nonrectangular_data, std::map<std::vector<NemoMeshPoint>, std::complex<double> >& density_momentum_map_interp,
                                              bool solve_on_single_replica, bool energy_resolved_density_ready,
                                              bool density_by_hole_factor)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("PropagationUtilities(\""+this_simulation->get_name()+"\")::integrate_diagonal ");
  std::string tic_toc_name=tic_toc_prefix;
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="PropagationUtilities::(\""+this_simulation->get_name()+"\")::integrate_diagonal: ";
  const InputOptions& options=this_simulation->get_options();
  PropagationOptionsInterface* PropOptionInterface=get_PropagationOptionsInterface(this_simulation);
  PropagatorInterface* PropInterface=get_PropagatorInterface(source_of_data);
  NEMO_ASSERT(PropOptionInterface!=NULL,prefix+this_simulation->get_name()+"is not a Propagation. This method cannot be called from other solver types\n");
  Propagation* this_propagation=dynamic_cast<Propagation*>(this_simulation);
  NEMO_ASSERT(this_propagation!=NULL,prefix+"should be called by a Propagation only\n");
  Simulation * Hamilton_Constructor = PropOptionInterface->get_Hamilton_Constructor();
 
  std::string name_of_writeable_Propagator=this_propagation->get_writeable_Propagator_name();

  std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
  std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
  std::vector<std::string> Mesh_tree_names;
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }

  bool complex_energy = PropInterface->complex_energy_used();
  std::string name_of_propagator;
  Propagator* writeable_Propagator=NULL;
  PropInterface->get_Propagator(writeable_Propagator);
  if(writeable_Propagator!=NULL)
    name_of_propagator=this_propagation->get_writeable_Propagator_name();
  else if(options.check_option("density_of"))
    name_of_propagator=options.get_option("density_of",std::string(""));
  //NemoPhys::Propagator_type input_type = get_Propagator_type(name_of_propagator);
  //1. we assume that data_name is the name of a propagator; use get_data to get a pointer to the Propagator
  NemoUtils::tic(tic_toc_prefix+"1.");
  Propagator* Propagator_pointer;
  std::string prefix_get_data1="PropagationUtilities::(\""+this_simulation->get_name()+"\")::integrate_diagonal: get_data 1 ";
  NemoUtils::tic(prefix_get_data1);
  source_of_data->get_data(data_name,Propagator_pointer);
  NemoUtils::toc(prefix_get_data1);
  Propagator::PropagatorMap::const_iterator momentum_c_it=Propagator_pointer->propagator_map.begin();

  PetscMatrixParallelComplex* temp_matrix=NULL;
  std::string prefix_get_data2="PropagationUtilities::(\""+this_simulation->get_name()+"\")::integrate_diagonal: get_data 2 ";
  NemoUtils::tic(prefix_get_data2);
  source_of_data->get_data(data_name,&(momentum_c_it->first),temp_matrix,&(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));
  //GreensfunctionInterface* green_source=dynamic_cast<GreensfunctionInterface*>(source_of_data);
  //NEMO_ASSERT(green_source!=NULL, prefix + source_of_data->get_name() + "is not a GreensfunctionInterface\n");
  //green_source->get_Greensfunction(momentum_c_it->first, temp_matrix, &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())),
  //    &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())), input_type);


  NemoUtils::toc(prefix_get_data2);
  NEMO_ASSERT(temp_matrix!=NULL,prefix+"have received NULL for the matrix pointer\n");
  //if(temp_matrix->if_container())
  //  temp_matrix->assemble();

  std::vector<std::complex<double> > temp_result(temp_matrix->get_num_cols(),std::complex<double>(0.0,0.0));
  std::string momentum_loop_tic = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_diagonal momentum loop ";
  NemoUtils::toc(tic_toc_prefix+"1.");
  NemoUtils::tic(momentum_loop_tic);

  //bool get_energy_resolved_data=options.get_option(data_name+"_energy_resolved_output",false) ||
  //                              options.get_option(data_name + "_energy_resolved",bool(false)) || options.get_option("energy_resolved_density",false);
  //bool get_energy_resolved_nonrectangular_data = false;
  if(get_energy_resolved_data)
  {
    //loop over all mesh_constructors and check whether one of them has the option ("non_rectangular = true")

    std::map<std::string, Simulation*>::const_iterator mesh_cit=Mesh_Constructors.begin();
    for(; mesh_cit!=Mesh_Constructors.end() && !get_energy_resolved_nonrectangular_data; ++mesh_cit)
    {
      InputOptions& mesh_options = mesh_cit->second->get_reference_to_options();
      if(mesh_options.get_option(std::string("non_rectangular"),false))
      {
        get_energy_resolved_nonrectangular_data = true;
        //don't want to store both energy resolved all k and energy resolved per k. This won't fit into memory.
        get_energy_resolved_data = false;
      }
    }
  }
  //1.1 if energy resolved data is wanted, prepare the required container and the all_energies
  std::vector<double> all_energies;
  std::vector<std::complex<double> > all_complex_energies;
  std::map<double, int> translation_map_energy_index;
  std::map<std::complex<double>, int, Compare_complex_numbers> translation_map_complex_energy_index;
  std::vector<vector<double> > all_kvectors;
  std::map<vector<double>, set<double> > all_energies_per_kvector;
  std::map<vector<double>, int > translation_map_kvector_index;

  std::set<std::vector<NemoMeshPoint> > all_momenta;
  std::set<std::vector<NemoMeshPoint> >* pointer_to_all_momenta =&all_momenta;
  if(get_energy_resolved_data)
  {
    //1.1 find all energies that exist - on all MPI processes
    if(pointer_to_all_momenta->size()==0)
      Parallelizer->get_data("all_momenta",pointer_to_all_momenta);
    NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"received empty all_momenta\n");
    std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
    std::set<double> temp_all_energies;
    std::set<std::complex<double>,Compare_complex_numbers > temp_all_complex_energies;
    std::string tic_toc_prefix1b = NEMOUTILS_PREFIX(tic_toc_prefix+" 1b.");
    NemoUtils::tic(tic_toc_prefix1b);
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      if(!complex_energy)
      {
        double temp_energy=0.0;
        Propagator * temp_propagator=NULL;
        source_of_data->get_data(data_name,temp_propagator);
        temp_energy = PropagationUtilities::read_energy_from_momentum(this_simulation,*c_it,temp_propagator);
        temp_all_energies.insert(temp_energy);
      }
      else
      {
        std::complex<double> temp_energy(0,0);
        Propagator * temp_propagator=NULL;
        source_of_data->get_data(data_name,temp_propagator);
        temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this_simulation,*c_it, temp_propagator);
        temp_all_complex_energies.insert(temp_energy);
      }
    }
    NemoUtils::toc(tic_toc_prefix1b);
    //1.2 save the ordered energies into a vector and save the index-mapping
    if(!complex_energy)
      all_energies.resize(temp_all_energies.size(),0.0);
    else
      all_complex_energies.resize(temp_all_complex_energies.size(),std::complex<double>(0.0,0.0));
    std::string tic_toc_prefix1c = NEMOUTILS_PREFIX(tic_toc_prefix+" 1c.");
    NemoUtils::tic(tic_toc_prefix1c);
    int counter = 0;
    if(!complex_energy)
    {
      std::set<double>::iterator e_it;
      for(e_it=temp_all_energies.begin(); e_it!=temp_all_energies.end(); e_it++)
      {
        double temp_energy = *e_it;
        translation_map_energy_index[temp_energy]=counter;
        all_energies[counter]=temp_energy;
        counter++;
      }
    }
    else
    {
      std::set<std::complex<double>,Compare_complex_numbers >::iterator e_it;
      for(e_it=temp_all_complex_energies.begin(); e_it!=temp_all_complex_energies.end(); e_it++)
      {
        std::complex<double> temp_energy = *e_it;
        translation_map_complex_energy_index[temp_energy]=counter;
        all_complex_energies[counter]=temp_energy;
        counter++;
      }
    }
    NemoUtils::toc(tic_toc_prefix1c);
  }
  if(get_energy_resolved_nonrectangular_data)
  {//stores abs of average value of the diagonal, needed for resonance mesh
    //find all energies that exist - on all MPI processes
    if(pointer_to_all_momenta->size()==0)
      Parallelizer->get_data("all_momenta",pointer_to_all_momenta);
    NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"received empty all_momenta\n");
    std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();

    std::set<std::complex<double>,Compare_complex_numbers > temp_all_complex_energies;
    //loop through all momenta
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      double temp_energy=0.0;
      Propagator * temp_propagator=NULL;
      source_of_data->get_data(data_name, temp_propagator);
      temp_energy = PropagationUtilities::read_energy_from_momentum(this_simulation, *c_it, temp_propagator);
      std::vector<double> momentum_point = PropagationUtilities::read_kvector_from_momentum(this_simulation, *c_it, temp_propagator);
      std::map<vector<double>, set<double> >::iterator k_it = all_energies_per_kvector.find(momentum_point);
      if (k_it != all_energies_per_kvector.end())
        k_it->second.insert(temp_energy);
      else
      {
        std::set<double> temp_set;
        temp_set.insert(temp_energy);
        all_energies_per_kvector[momentum_point] = temp_set;
      }
    }
  }
  //k resolved data requires storing all k vectors and their indexing
  //bool get_k_resolved_data = options.get_option(data_name+"_k_resolved_output", false) || k_resolved_density || options.get_option("k_resolved_density",false);
  if(get_k_resolved_data)
   {
     //1.1 find all energies that exist - on all MPI processes
     if(pointer_to_all_momenta->size()==0)
       Parallelizer->get_data("all_momenta",pointer_to_all_momenta);
     NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"received empty all_momenta\n");
     std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
     //store unique k-vectors in set
     std::set<vector<double> > temp_kvectors;
     //loop through all momenta
     for(; c_it!=pointer_to_all_momenta->end(); c_it++)
     {
       Propagator * temp_propagator=NULL;
       source_of_data->get_data(data_name,temp_propagator);
       std::vector<double> momentum_point=PropagationUtilities::read_kvector_from_momentum(this_simulation,*c_it, temp_propagator);
       temp_kvectors.insert(momentum_point);
     }
     //translate set into vector (preserve ordering)
     all_kvectors.resize(temp_kvectors.size());
     std::set<vector<double > >::iterator set_it = temp_kvectors.begin();
     for(int idx = 0; set_it!=temp_kvectors.end(); set_it++, idx++)
     {
       std::vector<double> temp_vector = *set_it;
       translation_map_kvector_index[*set_it]=idx;
       all_kvectors[idx]=*set_it;
     }
   }

  density_momentum_map_interp.clear();

  //2. loop over all momenta
  for(; momentum_c_it!=Propagator_pointer->propagator_map.end(); ++momentum_c_it)
  {
    //3. determining the integration weight of this momentum vector, i.e. this (E, k1,k2,...)
    NemoUtils::tic(tic_toc_prefix+"3.");
    double weight=1.0;
    std::complex<double> complex_weight(1.0,0.0);
    std::string momentum_type=std::string(""); //container for analytical integration
    double energy_resolved_integration_weight=1.0;
    double k_resolved_integration_weight = 1.0;
    std::complex<double> complex_k_resolved_integration_weight(1.0,0.0);
    //std::cerr<<" number of momentum: "<<momentum_c_it->first.size()<<endl;

    for(unsigned int i=0; i<momentum_c_it->first.size(); i++)
    {
      //get the constructor of this meshpoint
      std::string momentum_mesh_name=Propagator_pointer->momentum_mesh_names[i];
      std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
      NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
      Simulation* mesh_constructor=temp_cit->second;
      //ask the constructor for the weighting of this point
      double temp_double=0;
      std::complex<double> temp_complex(1.0,0.0);
      InputOptions& mesh_options=mesh_constructor->get_reference_to_options();
      std::string prefix_get_data_loop="Propagation::(\""+this_simulation->get_name()+"\")::integrate_diagonal: get_data in loop ";
      NemoUtils::tic(prefix_get_data_loop);

      if(!mesh_options.get_option(std::string("non_rectangular"),false))
      {
        if(complex_energy && momentum_mesh_name.find("energy") != std::string::npos)
        {
          if(input_type==NemoPhys::Fermion_retarded_Green)
          {
            bool isPole=0;
            std::complex<double> temp_complex_energy((momentum_c_it->first)[i].get_x(),(momentum_c_it->first)[i].get_y());
            mesh_constructor->get_data(temp_complex_energy, isPole);
            if(isPole)
            {
              temp_complex = 0.0;
            }
            else
            {
              mesh_constructor->get_data("integration_weight",(momentum_c_it->first)[i],temp_complex);
            }
          }
          else
          {
            mesh_constructor->get_data("integration_weight",(momentum_c_it->first)[i],temp_complex);
          }
        }
        else
        {
          mesh_constructor->get_data("integration_weight",(momentum_c_it->first)[i],temp_double);
          if (complex_energy)
            temp_complex.real() = temp_double;
        }
      }
      else
      {
        //3.1 if integration weight for energy
        if(Propagator_pointer->momentum_mesh_names[i].find("energy")!=std::string::npos)
        {
          //get the k-point from the momentum
          std::vector<double> temp_vector=PropagationUtilities::read_kvector_from_momentum(this_simulation, momentum_c_it->first, Propagator_pointer);
          NemoMeshPoint temp_momentum(0,temp_vector);
          std::vector<NemoMeshPoint> temp_vector_momentum(1,temp_momentum);
          //mesh_constructor->get_data("integration_weight",only-k (1dvector<NemoMeshPoint>,only energy NemoMeshPoint,temp_double);
          if(complex_energy)
          {
            if (input_type == NemoPhys::Fermion_retarded_Green)
            {
              bool isPole = 0;
              std::complex<double> temp_complex_energy((momentum_c_it->first)[i].get_x(), (momentum_c_it->first)[i].get_y());
              mesh_constructor->get_data(temp_complex_energy, isPole);
              if (isPole)
              {
                temp_complex = 0.0;
              }
              else
              {
                mesh_constructor->get_data("integration_weight", temp_vector_momentum, (momentum_c_it->first)[i], temp_complex);
              }
            }
            else
              mesh_constructor->get_data("integration_weight", temp_vector_momentum, (momentum_c_it->first)[i], temp_complex);
          }
          else
          {
            mesh_constructor->get_data("integration_weight",temp_vector_momentum,(momentum_c_it->first)[i],temp_double);
          }
        }
        //3.1' if integration weight for momentum or anything else
        else
        {
          mesh_constructor->get_data("integration_weight",momentum_c_it->first,(momentum_c_it->first)[i],temp_double);
          if(complex_energy)
            temp_complex.real() = temp_double;
        }

      }
      NemoUtils::toc(prefix_get_data_loop);
      if(complex_energy)
        complex_weight*=temp_complex;
      else
        weight*=temp_double;
      //if the current mesh weighting factor does not belong to energy mesh use it for the energy_resolved_integration_weight
      if(Propagator_pointer->momentum_mesh_names[i].find("energy")==std::string::npos)
        energy_resolved_integration_weight*=temp_double;
      else if(Propagator_pointer->momentum_mesh_names[i].find("Momentum")==std::string::npos)
      {
        if(complex_energy)
          complex_k_resolved_integration_weight*=temp_complex;
        else
          k_resolved_integration_weight*=temp_double;
      }
    }

    NemoUtils::toc(tic_toc_prefix+"3.");
    
    NemoUtils::tic(tic_toc_prefix+"4.");
    std::vector<std::complex<double> > diagonal;
    //4. use source_of_data->get_data to get reading access to the matrix of this respective momentum
    source_of_data->get_data(data_name,&(momentum_c_it->first),temp_matrix,&(Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain())));
    NemoUtils::toc(tic_toc_prefix+"4.");
    NemoUtils::tic(tic_toc_prefix+"5.");
    //5. get the diagonal of the matrix of this respective momentum
    //if(temp_matrix->if_container())
    //  temp_matrix->assemble();
    temp_matrix->get_diagonal(&diagonal);
    NEMO_ASSERT(temp_result.size()==diagonal.size(),prefix+"mismatch of diagonal size\n");
    NemoUtils::toc(tic_toc_prefix+"5.");
    
    NemoUtils::tic(tic_toc_prefix+"6.");
    density_momentum_map_interp[momentum_c_it->first]=0;
    for(unsigned int i =0; i<diagonal.size(); i++)
    {
      density_momentum_map_interp[momentum_c_it->first]+=diagonal[i];
    }

    {
      //6. do the integral - i.e. weighted sum
      if(density_by_hole_factor)
      {
        std::map<int,int> index_to_atom_id_map;

        std::map<unsigned int, double> threshold_map;
        std::map<unsigned int, double> threshold_map_right;
        double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum_c_it->first,Propagator_pointer);
        Hamilton_Constructor->get_data(std::string("hole_factor_dof"), energy, threshold_map, threshold_map_right);
        for(unsigned int j=0; j<temp_result.size(); j++)
        {

          std::map<unsigned int, double>::iterator thresh_it = threshold_map.find(j);
          NEMO_ASSERT(thresh_it!=threshold_map.end(),prefix+" could not find atom id in threshold map \n");
          double hole_factor = thresh_it->second;
          if(input_type == NemoPhys::Fermion_lesser_Green)
            temp_result[j]+=diagonal[j]*weight*(1-hole_factor);
          else if (input_type == NemoPhys::Fermion_greater_Green)
            temp_result[j]+=-diagonal[j]*weight*hole_factor;
          else
            throw std::invalid_argument(prefix+"called with unknown input type \n");

        }
      }
      else
      {
        if(!complex_energy)
        {
          for(unsigned int i=0; i<temp_result.size(); i++)
          {
            temp_result[i]+=diagonal[i]*weight;
          }
        }
        else
          for(unsigned int i=0; i<temp_result.size(); i++)
          {
            temp_result[i]+=diagonal[i]*complex_weight;
          }
      }

      if (options.get_option("output_energy_resolved_charge_derivative_integerand", false))
      {
        bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);
        double temp_energy = (momentum_c_it->first)[0].get_x();
        double temp_energy_imag = 0.0;
        if (complex_energy)
          temp_energy_imag = (momentum_c_it->first)[0].get_y();

        if (differentiate_over_energy)
        {
          std::ofstream f;
          if (!exists_file("energy_resolved_charge_derivative_integrand.dat"))
          {
            f.open("energy_resolved_charge_derivative_integrand.dat");
          }
          else
          {
            f.open("energy_resolved_charge_derivative_integrand.dat", ios::app);
          }
          f << "%" << temp_energy << " " << temp_energy_imag << "     ";
          for (vector<std::complex<double> >::const_iterator ii = diagonal.begin(); ii != diagonal.end(); ++ii) {
            std::complex<double> integrand;
            if (complex_energy)
              integrand = (*(ii)*complex_weight);
            else
              integrand = *(ii)*weight;
            f << integrand.real() << "     " << integrand.imag() << "      ";
          }
          f << "\n";
          f.close();
        }
      }

      //6.1 if required, solve the energy resolved integrated diagonal
      std::vector<cplx> temp_result_transformed = temp_result;
      //transform if not already transformed
      unsigned int number_of_DOFs = Hamilton_Constructor->get_const_dof_map(this_simulation->get_const_simulation_domain()).get_number_of_dofs();
      if (number_of_DOFs == temp_result.size() && options.get_option("transform_density",false))
      {
        temp_result_transformed.clear();
        double multiplication_factor = 1.0;
        TransformationUtilities::transform_vector_orbital_to_atom_resolved(Hamilton_Constructor, multiplication_factor, temp_result, temp_result_transformed);
      }

      if(get_energy_resolved_data)
      {
        if(complex_energy)
        {
          std::complex<double> energy = PropagationUtilities::read_complex_energy_from_momentum(this_simulation,momentum_c_it->first,Propagator_pointer);
          std::map<std::complex<double>,std::vector<std::complex<double> >,Compare_double_or_complex_number >* energy_resolved_data=Propagator_pointer->get_complex_energy_resolved_integrated_diagonal();
          //allocate the energy_resolved_data map
          if(energy_resolved_data->size()!=all_complex_energies.size())
          {
            energy_resolved_data->clear();
            for(unsigned int i=0; i<all_complex_energies.size(); i++)
              (*energy_resolved_data)[all_complex_energies[i]]=std::vector<std::complex<double> >(temp_result_transformed.size(),0.0);
          }
          std::map<std::complex<double>,std::vector<std::complex<double> >,Compare_double_or_complex_number >::iterator edata_it=energy_resolved_data->find(energy);
          if(edata_it==energy_resolved_data->end())
          {
            (*energy_resolved_data)[energy]=std::vector<std::complex<double> >(temp_result_transformed.size(),0.0);
            edata_it=energy_resolved_data->find(energy);
          }
          NEMO_ASSERT(edata_it!=energy_resolved_data->end(),prefix+"have not found energy entry in the energy resolved data map\n");
          for(unsigned int i=0; i<temp_result_transformed.size(); i++)
            (edata_it->second)[i]+=diagonal[i]*energy_resolved_integration_weight;
        }
        else
        {
          double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum_c_it->first,Propagator_pointer);
          std::map<double,std::vector<std::complex<double> >,Compare_double_or_complex_number >* energy_resolved_data=Propagator_pointer->get_energy_resolved_integrated_diagonal();
          //allocate the energy_resolved_data map
          if(energy_resolved_data->size()!=all_energies.size())
          {
            energy_resolved_data->clear();
            for(unsigned int i=0; i<all_energies.size(); i++)
              (*energy_resolved_data)[all_energies[i]]=std::vector<std::complex<double> >(temp_result_transformed.size(),0.0);
          }
          std::map<double,std::vector<std::complex<double> >,Compare_double_or_complex_number >::iterator edata_it=energy_resolved_data->find(energy);
          if(edata_it==energy_resolved_data->end())
          {
            (*energy_resolved_data)[energy]=std::vector<std::complex<double> >(temp_result_transformed.size(),0.0);
            edata_it=energy_resolved_data->find(energy);
          }
          NEMO_ASSERT(edata_it!=energy_resolved_data->end(),prefix+"have not found energy entry in the energy resolved data map\n");
          for(unsigned int i=0; i<temp_result_transformed.size(); i++)
            (edata_it->second)[i]+=diagonal[i]*energy_resolved_integration_weight;
        }
      }
      if(get_energy_resolved_nonrectangular_data)
      {//stores abs of average value of the diagonal, needed for resonance mesh
        double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum_c_it->first,Propagator_pointer);
        vector<double> k_vector = PropagationUtilities::read_kvector_from_momentum(this_simulation,momentum_c_it->first,Propagator_pointer);
        std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >* energy_per_k_data =
            Propagator_pointer->get_energy_resolved_per_k_integrated_diagonal();

        if(energy_per_k_data->size()!=all_energies_per_kvector.size())
        {
          energy_per_k_data->clear();
          std::map<std::vector<double>, std::set<double> >::iterator k_it = all_energies_per_kvector.begin();
          //for(unsigned int i=0; i<all_energies_per_kvector.size(); i++)
          for(; k_it != all_energies_per_kvector.end(); ++k_it)
          {
            std::map<double, std::vector<std::complex<double> > > temp_map;
            (*energy_per_k_data)[k_it->first]=temp_map;//std::vector<std::complex<double> >(temp_result.size(),0.0);
          }
        }

        std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >::iterator k_data_it = energy_per_k_data->begin();
        for(; k_data_it!=energy_per_k_data->end(); k_data_it++)
        {
          std::map<vector<double>,std::set<double> >::iterator k_it = all_energies_per_kvector.find(k_data_it->first);
          if(k_data_it->second.size()!=k_it->second.size())
          {
            std::set<double>::iterator set_it = k_it->second.begin();
            for(; set_it != k_it->second.end(); set_it++)
              (k_data_it->second)[*set_it] = std::vector<std::complex<double> >(1,0.0);
          }
        }

        //find specific k
        k_data_it = energy_per_k_data->find(k_vector);
        std::map<vector<double>,std::set<double> >::iterator k_it = all_energies_per_kvector.find(k_vector);
        {
          //std::map<double,std::vector<std::complex<double> > > temp_e_data = k_data_it->second;
          std::map<double,std::vector<std::complex<double> > >::iterator edata_it = k_data_it->second.begin();//temp_e_data.begin();

          edata_it = k_data_it->second.find(energy);
          NEMO_ASSERT(edata_it!=k_data_it->second.end(),prefix+"have not found energy entry in the energy resolved data map\n");
          std::vector<std::complex<double> > temp_vector;
          cplx average_input_scalar(0.0,0.0);
          for(unsigned int i=0; i<temp_result_transformed.size(); i++)
            average_input_scalar+=diagonal[i]*energy_resolved_integration_weight;

          average_input_scalar/=temp_result_transformed.size();
          average_input_scalar = std::abs(average_input_scalar);
          (edata_it->second)[0]=average_input_scalar;
        }

      }
      //if required also solve the k resolved diagonal
      if(get_k_resolved_data)
      {
        std::map<std::vector<double>,std::vector<std::complex<double> > >* k_resolved_data=Propagator_pointer->get_k_resolved_integrated_diagonal();

        if(k_resolved_data->size()!=all_kvectors.size())
        {
          k_resolved_data->clear();
          for(unsigned int i=0; i<all_kvectors.size(); i++)
            (*k_resolved_data)[all_kvectors[i]]=std::vector<std::complex<double> >(temp_result_transformed.size(),0.0);
        }
        {
          vector<double> k_vector = PropagationUtilities::read_kvector_from_momentum(this_simulation,momentum_c_it->first,Propagator_pointer);
          std::map<vector<double>,std::vector<std::complex<double> > >::iterator kdata_it=k_resolved_data->find(k_vector);
          if(kdata_it==k_resolved_data->end())
          {
            (*k_resolved_data)[k_vector]=std::vector<std::complex<double> >(temp_result_transformed.size(),0.0);
            kdata_it=k_resolved_data->find(k_vector);
          }
          NEMO_ASSERT(kdata_it!=k_resolved_data->end(),prefix+"have not found kvector entry in the k_resolved data map\n");
          for (unsigned int i = 0; i < temp_result_transformed.size(); i++)
          {
            if(!complex_energy)
              (kdata_it->second)[i] += diagonal[i] * k_resolved_integration_weight;
            else
              (kdata_it->second)[i] += diagonal[i] * complex_k_resolved_integration_weight;
          }
        }
      }

    }
    NemoUtils::toc(tic_toc_prefix+"6.");
  }
  NemoUtils::toc(momentum_loop_tic);
  
  std::string mpi_tic_tocs = "PropagationUtilities(\""+tic_toc_name+"\")::integrate_diagonal2 MPI Barrier/Reduce ";
  NemoUtils::tic(mpi_tic_tocs);
  //store the result in the Propagator data_name
  std::vector<std::complex<double> >* pointer_to_result=Propagator_pointer->get_writeable_integrated_diagonal();
  if(pointer_to_result->size()>0)
    pointer_to_result->clear();
  pointer_to_result->resize(temp_result.size());
  //sum up the results of all other MPI processes
  if(Mesh_tree_topdown.size()==0)
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,prefix+"Mesh_tree_topdown is not ready for usage\n");
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  std::string barrier_tic_tocs = "Propagation(\""+tic_toc_name+"\")::integrate_diagonal2 MPI Barrier ";
  NemoUtils::tic(barrier_tic_tocs);

  MPI_Barrier(topcomm);
  NemoUtils::toc(barrier_tic_tocs);
  if(!solve_on_single_replica)
    MPI_Allreduce(&(temp_result[0]),&((*pointer_to_result)[0]),temp_result.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
  else
    MPI_Reduce(&(temp_result[0]),&((*pointer_to_result)[0]),temp_result.size(), MPI_DOUBLE_COMPLEX, MPI_SUM , 0, topcomm);

  //sum up the results of all other MPI processes, energy resolved
  if(get_energy_resolved_data)
  {
    if(complex_energy)
    {
      std::map<std::complex<double>,std::vector<std::complex<double> >,Compare_double_or_complex_number >* energy_resolved_data=Propagator_pointer->get_complex_energy_resolved_integrated_diagonal();
      std::map<std::complex<double>,std::vector<std::complex<double> >,Compare_double_or_complex_number >::iterator edata_it=energy_resolved_data->begin();
      //perform the sum over MPI data for a single energy
      for(; edata_it!=energy_resolved_data->end(); ++edata_it)
      {
        std::vector<std::complex<double> > send_vector=edata_it->second;
        if(!solve_on_single_replica)
          MPI_Allreduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
        else
          MPI_Reduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM , 0, topcomm);
      }
      energy_resolved_density_ready = true;
      //clean up the rest
      if(options.get_option("clean_up_other_ranks",false))
      {
        int rank;
        MPI_Comm_rank(topcomm,&rank);
        if(rank!=0)
          energy_resolved_data->clear();
        energy_resolved_density_ready = false;
      }
    }
    else
    {
      std::map<double,std::vector<std::complex<double> >,Compare_double_or_complex_number >* energy_resolved_data=Propagator_pointer->get_energy_resolved_integrated_diagonal();
      std::map<double,std::vector<std::complex<double> >,Compare_double_or_complex_number >::iterator edata_it=energy_resolved_data->begin();
      //perform the sum over MPI data for a single energy
      for(; edata_it!=energy_resolved_data->end(); ++edata_it)
      {
        std::vector<std::complex<double> > send_vector=edata_it->second;
        if(!solve_on_single_replica)
          MPI_Allreduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
        else
          MPI_Reduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM , 0, topcomm);
      }
      energy_resolved_density_ready = true;
      //clean up the rest
      if(options.get_option("clean_up_other_ranks",false))
      {
        int rank;
        MPI_Comm_rank(topcomm,&rank);
        if(rank!=0)
          energy_resolved_data->clear();
        energy_resolved_density_ready = false;
      }
    }
  }
  if(get_energy_resolved_nonrectangular_data)
   {
    std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >* energy_resolved_per_k_data=
        Propagator_pointer->get_energy_resolved_per_k_integrated_diagonal();
    std::map<vector<double>, std::map<double,std::vector<std::complex<double> > > >::iterator k_it=energy_resolved_per_k_data->begin();
     //perform the sum over MPI data for a single energy
    for(; k_it != energy_resolved_per_k_data->end(); ++k_it)
    {
      std::map<double,std::vector<std::complex<double> > >::iterator edata_it= k_it->second.begin();
      for(; edata_it!=k_it->second.end(); ++edata_it)
      {
        std::vector<std::complex<double> > send_vector=edata_it->second;
        MPI_Allreduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
      }
    }
   }
  if(get_k_resolved_data)
  {
    std::map<vector<double>,std::vector<std::complex<double> > >* k_resolved_data=Propagator_pointer->get_k_resolved_integrated_diagonal();
    std::map<vector<double>,std::vector<std::complex<double> > >::iterator kdata_it=k_resolved_data->begin();
    //perform the sum over MPI data for a single energy
    for(; kdata_it!=k_resolved_data->end(); ++kdata_it)
    {
      std::vector<std::complex<double> > send_vector=kdata_it->second;
      if(!solve_on_single_replica)
        MPI_Allreduce(&(send_vector[0]),&(kdata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
      else
        MPI_Reduce(&(send_vector[0]),&(kdata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM , 0, topcomm);
    }

    //clean up the rest
    if(options.get_option("clean_up_other_ranks",false))
    {
      int rank;
      MPI_Comm_rank(topcomm,&rank);
      //if(rank!=0)
      //  k_resolved_data->clear();
    }
  }
  NemoUtils::toc(mpi_tic_tocs);
  NemoUtils::toc(tic_toc_prefix);
}
