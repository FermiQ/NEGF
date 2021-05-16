// NEMO5, The Nanoelectronics simulation package.
// This package is a free software.
// It is distributed under the NEMO5 Non-Commercial License (NNCL).
// Purdue Research Foundation, 1281 Win Hentschel Blvd., West Lafayette, IN 47906, USA
//$Id: Greensolver.cpp 23239 $


#include "Simulation.h"
#include "Propagator.h"
#include "Propagation.h"
#include "Greensolver.h"
#include "QTBM.h"
#include "Matrix.h"
#include "PetscMatrixParallelComplex.h"
#include "PetscMatrixParallelComplexContainer.h"
#include <map>
#include <complex>
#include <stdexcept>
#include "Nemo.h"
#include "DOFmap.h"
#include "NemoUtils.h"
#include "EigensolverSlepc.h"
#include "Domain.h"
#include "NemoPhys.h"
#include "NemoMICUtils.h"
#include "QuantumNumberUtils.h"
#include "PropagationUtilities.h"

#include <iostream>
#include <fstream>


/*  Purpose:    Greensolver class = base class for every Greensolver-simulation
                irrespective of system's dimensionality or its representation
    Note:       child class of the propagation class
 */

using  NemoUtils::MsgLevel;

Greensolver::~Greensolver()
{
//#ifdef MAGMA_ENABLE
//  if(gpu_memory_initialized)
//  {
//    if(temp1_gpu != NULL)
//      delete temp1_gpu;
//    if(temp2_gpu != NULL)
//      delete temp2_gpu;
//    if(temp3_gpu != NULL)
//      delete temp3_gpu;
//    if(temp4_gpu != NULL)
//      delete temp4_gpu;
//    if(temp5_gpu != NULL)
//      delete temp5_gpu;
//    if(temp6_gpu != NULL)
//      delete temp6_gpu;
//
//    destroyGPUContext(ctx);
//    destroyGPUContext(transfer_ctx);
//  }
//#endif
}

Greensolver::Greensolver()
{
//#ifdef MAGMA_ENABLE
//  temp1_gpu = NULL;
//  temp2_gpu = NULL;
//  temp3_gpu = NULL;
//  temp4_gpu = NULL;
//  temp5_gpu = NULL;
//  temp6_gpu = NULL;
//  ctx = NULL;
//  transfer_ctx = NULL;
//  gpu_memory_initialized = false;
//#endif
}

void Greensolver::do_init()
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_init ";
  NemoUtils::tic(tic_toc_prefix);

  //initialize the base class
  base_init();
  known_Propagators.clear();
  std::map<std::string, const Propagator*>::iterator prop_it=Propagators.begin();
  for(;prop_it!=Propagators.end();++prop_it)
    if(prop_it->second!=NULL)
      known_Propagators.insert(prop_it->second);
  if(writeable_Propagator!=NULL)
    known_Propagators.insert(writeable_Propagator);
  //NEMO_ASSERT(name_of_writeable_Propagator.size()>0||Parallelizer==this,tic_toc_prefix+"empty name_of_writeable_Propagator\n");

  //  test_debug();
  //-----------------------------
  //first, check that all propagators to solve are indeed Green's functions
  //-----------------------------
  //std::map<std::string,Propagator*>::const_iterator it = writeable_Propagators.begin();
  //for (; it!=writeable_Propagators.end(); ++it)

  if(writeable_Propagator!=NULL)
  {
    //NEMO_ASSERT(Propagator_type_map.find(Propagator_types.find(name_of_writeable_Propagator)->second)->second.find(std::string("Green"))!=std::string::npos,
    //            std::string("Green_Solver(\""+this->get_name()+"\")::do_init() NEGF-object \""+name_of_writeable_Propagator+"\" is not a Green's function\n"));

    if(Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Inverse_Green)
      do_init_inverse();
    else if(Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Fermion_retarded_Green||
            Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Boson_retarded_Green)
      do_init_retarded();
    else if(Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Fermion_lesser_Green||
            Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Boson_lesser_Green)
      do_init_lesser();
  }
  //  test_debug();
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::get_Greensfunction(const std::vector<NemoMeshPoint>& momentum,PetscMatrixParallelComplex*& result,
                                     const DOFmapInterface* row_dofmap, const DOFmapInterface* column_dofmap,const NemoPhys::Propagator_type& /*type*/)
{
  get_data(name_of_writeable_Propagator,&momentum,result,row_dofmap,column_dofmap);
}

void Greensolver::do_solve_retarded(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_solve_retarded";
  NemoUtils::tic(tic_toc_prefix);
  activate_regions();
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=output_Propagator->propagator_map.find(momentum_point);
  if(matrix_it!=output_Propagator->propagator_map.end())
    temp_matrix=matrix_it->second;
  do_solve_retarded(output_Propagator,momentum_point,temp_matrix);
  NEMO_ASSERT(is_ready(output_Propagator->get_name(),momentum_point),tic_toc_prefix+"still not ready\n");
  write_propagator(output_Propagator->get_name(),momentum_point, temp_matrix);
  conclude_after_do_solve(output_Propagator,momentum_point);
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::do_solve_retarded(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_solve_retarded ";
  NemoUtils::tic(tic_toc_prefix);

  //std::string error_prefix = "Greensolver(\""+this->get_name()+"\")::do_solve_retarded: ";
  //msg<<error_prefix<<"of \""<<output_Propagator->get_name()<<"\""<<std::endl;
  //specify how to solve(invert)
  if(inversion_method == NemoMath::exact_inversion)
  {
    GreensfunctionInterface* green_source=NULL;
    if(read_NEGF_object_list)
    {
      //find the name of the inverse retarded Greens function
      std::string inverse_name; //name of the inverse Green's function
      std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
      for(; c_it!=Propagators.end(); ++c_it)
      {
        if(name_of_writeable_Propagator!=c_it->first)
          if(c_it->first.find("inverse")!=std::string::npos)
          {
            inverse_name=c_it->first;
          }
      }
      //Simulation* data_source=find_source_of_data(inverse_name);
      green_source=find_source_of_Greensfunction(inverse_name);
    }
    else
    {
      NEMO_ASSERT(inverse_GR_solver!=NULL,tic_toc_prefix+"inverse_GR_solver is NULL\n");
      green_source=dynamic_cast<GreensfunctionInterface*>(inverse_GR_solver);
      NEMO_ASSERT(green_source!=NULL,tic_toc_prefix+"inverse_GR_solver is not a GreensfunctionInterface\n");
    }

    PetscMatrixParallelComplex* inverse_Green=NULL;
    green_source->get_Greensfunction(momentum_point,inverse_Green,&(Hamilton_Constructor->get_dof_map(get_const_simulation_domain())),NULL,NemoPhys::Inverse_Green);
    //data_source->get_data(inverse_name,&momentum_point,inverse_Green,
    //  &(Hamilton_Constructor->get_dof_map(get_const_simulation_domain())));

    delete result;

    std::string invert_solver_option;
    std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;
    if (offload_solver_initialized && offload_solver->offload)
    {
      c_it_momentum_offload_info = offload_solver->offloading_momentum_map.find(momentum_point);
      NEMO_ASSERT(c_it_momentum_offload_info != offload_solver->offloading_momentum_map.end(),
          tic_toc_prefix + "have not found momentum in the offloading_momentum_map\n");

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
      invert_solver_option = options.get_option("invert_solver",std::string("petsc"));
    }

    if(invert_solver_option==std::string("lapack"))
    {
      NemoUtils::tic(tic_toc_prefix+"invertLapack");
      //use invertLapack
      //PetscMatrixParallelComplex* tran_inverse=NULL;
      //inverse_Green->matrix_convert_dense(tran_inverse);
      //PetscMatrixParallelComplex::invertLapack(*tran_inverse, &result);
      //delete tran_inverse;
      inverse_Green->matrix_convert_dense();
      PetscMatrixParallelComplex::invertLapack(*inverse_Green, &result);
      NemoUtils::toc(tic_toc_prefix+"invertLapack");
    }
    else if (invert_solver_option==std::string("MIC_offload"))
    {
      NemoUtils::tic(tic_toc_prefix+"invertLapack");
      //use invertLapack
      //PetscMatrixParallelComplex* tran_inverse=NULL;
      //inverse_Green->matrix_convert_dense(tran_inverse);
      //PetscMatrixParallelComplex::invertLapack(*tran_inverse, &result);
      //delete tran_inverse;
      inverse_Green->matrix_convert_dense();

      if (offload_solver_initialized && offload_solver->offload)
      {
        NEMO_ASSERT(c_it_momentum_offload_info != offload_solver->offloading_momentum_map.end(),
            tic_toc_prefix + "have not found momentum in the offloading_momentum_map\n");

        //ResourceUtils::OffloadInfo offload_info = c_it_momentum_offload_info->second;

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
      /*PetscMatrixParallelComplex* isitidentity = NULL;
      PetscMatrixParallelComplex::mult(*inverse_Green,*result,&isitidentity);
      inverse_Green->assemble();
      inverse_Green->save_to_matlab_file("inverse_Green.m");
      result->assemble();
      result->save_to_matlab_file("result.m");
      isitidentity->assemble();
      isitidentity->save_to_matlab_file("isitidentity.m");
      NEMO_ASSERT(false,"Multiplied the answer");*/
    }
    else  if(invert_solver_option==std::string("magma"))
    {
#ifdef MAGMA_ENABLE
      //        std::cerr<<"\n Invert using MAGMA\n";
      inverse_Green->matrix_convert_dense();
      PetscMatrixParallelComplex::invertMAGMA(*inverse_Green, &result);
#else
      NEMO_EXCEPTION("Cannot use MAGMA library without compiling and linking with MAGMA!\n");
#endif

    }
    else  if(invert_solver_option==std::string("magmamic"))
    {
#ifdef MAGMAMIC_ENABLE
      //        std::cerr<<"\n Invert using MAGMAMIC\n";
      inverse_Green->matrix_convert_dense();
      PetscMatrixParallelComplex::invertMAGMAMIC(*inverse_Green, &result);
#else
      NEMO_EXCEPTION("Cannot use MAGMAMIC library without compiling and linking with MAGMA!\n");
#endif

    }
    else
    {
      //std::cerr<<"\n Invert using PetscMatrix\n";
      NEMO_ASSERT(inverse_Green!=NULL,tic_toc_prefix+"received NULL for the inverseGR matrix\n");
      PetscMatrixParallelComplex::invert(*inverse_Green, &result);
      /*PetscMatrixParallelComplex* isitidentity = NULL;
      PetscMatrixParallelComplex::mult(*inverse_Green,*result,&isitidentity);
      inverse_Green->assemble();
      inverse_Green->save_to_matlab_file("inverse_Green.m");
      result->assemble();
      result->save_to_matlab_file("result.m");
      isitidentity->assemble();
      isitidentity->save_to_matlab_file("isitidentity.m");
      NEMO_ASSERT(false,"Multiplied the answer");*/

    }

    if(debug_output)
    {
      std::string temp_string;
      const std::vector<NemoMeshPoint>* temp_pointer=&momentum_point;
      translate_momentum_vector(temp_pointer, temp_string);
      temp_string+="_dsr_"+get_output_suffix();//options.get_option("output_suffix",std::string(""));
      inverse_Green->save_to_matlab_file("inverse_"+output_Propagator->get_name()+temp_string+".m");
      result->save_to_matlab_file(output_Propagator->get_name()+temp_string+".m");
    }

    NEMO_ASSERT(result!=NULL, "Greensolver(\""+this->get_name()+"\")::do_solve_retarded: inversion failed\n");

  }
  else if(inversion_method == NemoMath::recursive_backward_inversion)
    solve_retarded_Green_back_RGF(output_Propagator, momentum_point, result);
  else if(inversion_method == NemoMath::recursive_off_diagonal_inversion)
    solve_interdomain_Green_RGF(output_Propagator, momentum_point, result);
  else if(inversion_method == NemoMath::sancho_rubio_inversion)
    solve_retarded_Green_SanchoRubio(output_Propagator, momentum_point, result);
  else
    throw std::invalid_argument(tic_toc_prefix+"unknown inversion type\n");
  set_job_done_momentum_map(&(output_Propagator->get_name()), &momentum_point, true);

  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::solve_retarded_Green_back_RGF(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::solve_retarded_Green_back_RGF ";
  NemoUtils::tic(tic_toc_prefix);

  //std::string error_prefix = "Greensolver(\""+this->get_name()+"\")::solve_retarded_Green_back_RGF: ";
  //std::string tic_toc_g = tic_toc_prefix+" obtain half way gR ";
  //NemoUtils::tic(tic_toc_g);

  //1. get the g-function of this domain, i.e. the Greensfunction that is only connected to one neighbor domain (cf. J. Appl. Phys. 81, 7845; Eq.59)
  std::string half_way_name;
  std::string variable_name="half_way_retarded_green";
  if (options.check_option(variable_name))
    half_way_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix+"define \""+variable_name+"\"\n");
  //Simulation* source_of_half_way = find_source_of_data(half_way_name);
  GreensfunctionInterface* source_of_half_way = find_source_of_Greensfunction(half_way_name);
  //NEMO_ASSERT(source_of_half_way!=NULL,error_prefix+half_way_name+" is not Greensfunction compatible\n");
  PetscMatrixParallelComplex* half_way_g_matrix=NULL;
  const DOFmapInterface& half_way_DOFmap = Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain());
  //source_of_half_way->get_data(half_way_name,&momentum_point,half_way_g_matrix,&half_way_DOFmap);
  if(particle_type_is_Fermion)
    source_of_half_way->get_Greensfunction(momentum_point,half_way_g_matrix,&half_way_DOFmap,NULL,NemoPhys::Fermion_retarded_Green);
  else
    source_of_half_way->get_Greensfunction(momentum_point,half_way_g_matrix,&half_way_DOFmap,NULL,NemoPhys::Boson_retarded_Green);

  //NemoUtils::toc(tic_toc_g);

  //std::string tic_toc_couplingH = tic_toc_prefix+" obtain coupling Hamiltonian to neighbor ";
  //NemoUtils::tic(tic_toc_couplingH);
  //2. get the coupling Hamiltonian that couples this domain to the neighboring one
  //what is the neighboring domain:
  variable_name=output_Propagator->get_name()+std::string("_lead_domain");
  std::string neighbor_domain_name;
  if (options.check_option(variable_name))
    neighbor_domain_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix+"define \""+variable_name+"\"\n");
  const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);
  Simulation* temp_simulation=find_simulation_on_domain(neighbor_domain,Hamilton_Constructor->get_type());
  if (temp_simulation == NULL && (Hamilton_Constructor->get_type().find("module") != std::string::npos || Hamilton_Constructor->get_type().find("Module") != std::string::npos))
    temp_simulation = Hamilton_Constructor;
  NEMO_ASSERT(temp_simulation!=NULL,tic_toc_prefix+"have not found Hamilton_Constructor for neighbor_domain\n");
  PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;
  DOFmapInterface* coupling_DOFmap=NULL;
  DOFmapInterface& temp_DOFmap=temp_simulation->get_dof_map(neighbor_domain);
  coupling_DOFmap=&temp_DOFmap;
  DOFmapInterface* temp_pointer=coupling_DOFmap;
  //find the momentum that the Hamilton constructor can read out of momentum_point
  //std::set<unsigned int> Hamilton_momentum_indices;
  //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
  //find_Hamiltonian_momenta(output_Propagator,pointer_to_Hamilton_momentum_indices);
  //set_valley(output_Propagator, momentum_point, Hamilton_Constructor); //Hamilton_Constructor added by Bozidar
  //std::vector<double> temp_vector(3,0.0);
  //NemoMeshPoint temp_momentum(0,temp_vector);
  //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_momentum=momentum_point[*(Hamilton_momentum_indices.begin())];

  std::vector<NemoMeshPoint> sorted_momentum;
  QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum,options,momentum_mesh_types,Hamilton_Constructor);
  Hamilton_Constructor->get_data(std::string("Hamiltonian"), sorted_momentum, neighbor_domain, coupling_Hamiltonian,coupling_DOFmap,
                                 get_const_simulation_domain());
  if(coupling_DOFmap!=temp_pointer)
    delete coupling_DOFmap;


//  PetscMatrixParallelComplex transpose_coupling_Hamiltonian(coupling_Hamiltonian->get_num_cols(),
//      coupling_Hamiltonian->get_num_rows(),
//      coupling_Hamiltonian->get_communicator());
//
//  coupling_Hamiltonian->hermitian_transpose_matrix(transpose_coupling_Hamiltonian,MAT_INITIAL_MATRIX);

  //NemoUtils::toc(tic_toc_couplingH);

  //std::string tic_toc_G_neighbor = tic_toc_prefix+" get exact GR of neighbor ";
  //NemoUtils::tic(tic_toc_G_neighbor);
  //3. get the exact Greensfunction of the neighboring domain (cf. J. Appl. Phys. 81, 7845; Eq.60)
  std::string exact_neighbor_name;
  variable_name="exact_lead_retarded_green";
  if (options.check_option(variable_name))
    exact_neighbor_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(tic_toc_prefix+"define \""+variable_name+"\"\n");
  //Simulation* source_of_exact_neighbor = find_source_of_data(exact_neighbor_name);
  GreensfunctionInterface* source_of_exact_neighbor =find_source_of_Greensfunction(exact_neighbor_name);

  //if lead_Hamilton_constructor is defined, get the lead DOFmap from it, otherwise use source_of_exact_neighbor
  const DOFmapInterface* exact_lead_DOFmap=NULL;
  if(options.check_option("exact_lead_Hamilton_constructor"))
  {
    std::string temp_name = options.get_option("exact_lead_Hamilton_constructor",std::string(""));
    Simulation* lead_Hamilton_constructor = find_simulation(temp_name);
    exact_lead_DOFmap=&(lead_Hamilton_constructor->get_dof_map(neighbor_domain));
  }
  else
    throw std::invalid_argument(tic_toc_prefix+"please define \"exact_lead_Hamilton_constructor\"\n");

  PetscMatrixParallelComplex* exact_neighbor_matrix=NULL;
  //source_of_exact_neighbor->get_data(exact_neighbor_name,&momentum_point,exact_neighbor_matrix,
  //                                   exact_lead_DOFmap); //NOTE: here we have to add the DOFmap as an input, to define the submatrix...
  //NEMO_ASSERT(exact_neighbor_matrix!=NULL,error_prefix+"received NULL for exact_neighbor_matrix1\n");
  if(particle_type_is_Fermion)
    source_of_exact_neighbor->get_Greensfunction(momentum_point,exact_neighbor_matrix,exact_lead_DOFmap, NULL, NemoPhys::Fermion_retarded_Green);
  else
    source_of_exact_neighbor->get_Greensfunction(momentum_point,exact_neighbor_matrix,exact_lead_DOFmap, NULL, NemoPhys::Boson_retarded_Green);

  //NemoUtils::toc(tic_toc_G_neighbor);
  //std::string tic_toc_GLL = tic_toc_prefix+" calculate G_LL ";
  //NemoUtils::tic(tic_toc_GLL);
  //4. G_LL = g_LL + g_LL (t_LL-1) G_L-1L-1 (t_L-1L) gLL
  //for this equation, we are using the super-matrix concept of the contact self-energy methods again, i.e creating matrices of the size of both domains...
  //we need four super matrices (for g_LL, G_L-1L-1, t_LL-1 and the result)
  //Note: g_LL is of this domain and therefore in the upper left corner of the super-matrix
  // G_L-1L-1 is of the neighbor domain and therefore in the lower right corner of the super-matrix
  // t_LL-1 (coupling_Hamiltonian) is in the upper right corner of the super-matrix
  //construct the domain3=domain1+domain2 matrices (i.e. "super-matrices")
  unsigned int number_of_super_rows;
  unsigned int number_of_super_cols;
  int start_own_super_rows;
  int end_own_super_rows_p1;
  unsigned int number_of_rows1;
  unsigned int number_of_cols1;

  number_of_super_rows = coupling_Hamiltonian->get_num_rows();
  number_of_super_cols = coupling_Hamiltonian->get_num_cols();
  coupling_Hamiltonian->get_ownership_range(start_own_super_rows, end_own_super_rows_p1);
  //NEMO_ASSERT(number_of_super_rows == number_of_super_cols, tic_toc_prefix + "rectangular matrix received for the super-domain\n");

  bool LRA_RGF = options.get_option("LRA_RGF", bool(false));
  if (!LRA_RGF)
  {
    //these are the dimensions of this domain's matrices:
    number_of_rows1 = half_way_g_matrix->get_num_rows();
    number_of_cols1 = half_way_g_matrix->get_num_cols();
  }
  else //called by LRA_RGF
  {
    std::string Hamilton_Constructor2_name = options.get_option("Hamilton_constructor_for_coupling", std::string(""));
    Simulation* Hamilton_Constructor2 = find_simulation(Hamilton_Constructor2_name);
    const DOFmapInterface& device_DOFmap2 = Hamilton_Constructor2->get_const_dof_map(get_const_simulation_domain());
    //these are the dimensions of this domain's matrices:
    number_of_rows1 = device_DOFmap2.get_global_dof_number();
    //if(coupling_Hamiltonian->get_num_rows() < number_of_rows1)
    //{
    //  number_of_rows1 = 0;
    //}

    number_of_cols1 = number_of_rows1;
  }
  //these are the dimensions of the neighbor domain matrices:


  int number_of_rows2 = number_of_super_rows - number_of_rows1;
  int number_of_cols2 = number_of_super_cols - number_of_cols1;
  //NEMO_ASSERT(number_of_rows2 == number_of_cols2, tic_toc_prefix + "rectangular matrix received for neighbor domain\n");
  if(coupling_Hamiltonian->get_num_rows() <= number_of_rows1)
  {
    number_of_rows1 = number_of_super_rows;
    number_of_cols1 = 0;//number_of_super_cols;
    number_of_rows2 = number_of_super_rows;
    number_of_cols2  = number_of_super_cols;
  }
  //=====================================
  //Yu: new version -- avoid super matrix
  //-------------------------------------
  //1. get the submatrix (upper right block) from coupling Hamiltonian
  //NemoUtils::tic(tic_toc_prefix + "get submatrix from coupling Hamiltonian");
  std::vector<int> temp_rows(number_of_rows1);
  std::vector<int> temp_cols(number_of_cols2);
  for(unsigned int i=0; i<number_of_rows1; i++)
    temp_rows[i]=i;
  for(int i=0; i<number_of_cols2; i++)
    temp_cols[i]=i+number_of_cols1;
  PetscMatrixParallelComplex* coupling = NULL;
  coupling= new PetscMatrixParallelComplex(number_of_rows1,number_of_cols2,get_simulation_domain()->get_communicator());
  coupling->set_num_owned_rows(number_of_rows1);
  vector<int> rows_diagonal(number_of_rows1,0);
  vector<int> rows_offdiagonal(number_of_rows1,0);
  for(unsigned int i=0; i<number_of_rows1; i++)
  {
    rows_diagonal[i]=coupling_Hamiltonian->get_nz_diagonal(i);
    rows_offdiagonal[i]=coupling_Hamiltonian->get_nz_offdiagonal(i);
  }
  for(unsigned int i=0; i<number_of_rows1; i++)
    coupling->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
  coupling_Hamiltonian->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,coupling);
  coupling->assemble();
  delete coupling_Hamiltonian;
  coupling_Hamiltonian=NULL;
  //NemoUtils::toc(tic_toc_prefix + "get submatrix from coupling Hamiltonian");


  //2. perform the multiplication
  Mat_mult_method mat_mult = Greensolver::petsc;

  // Obtain propagator offload map to determine if this momentum should be offloaded
  std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;

  if (offload_solver_initialized && offload_solver->offload)
  {
    c_it_momentum_offload_info = offload_solver->offloading_momentum_map.find(momentum_point);

    if (c_it_momentum_offload_info->second.offload_to == ResourceUtils::MIC)
    {
      if (options.get_option("use_pcl_lapack", bool(true)))
        mat_mult = Greensolver::pcl_MIC;
      else
        mat_mult = Greensolver::blas_MIC;
    }
    else if(c_it_momentum_offload_info->second.offload_to == ResourceUtils::GPU)
    {
      if(options.get_option("use_cuda_interface", bool(false)))
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

    //NemoUtils::tic(tic_toc_prefix + "couplingH x GR");
    coupling->multBLAS(exact_neighbor_matrix, temp_matrix1, no_transpose, no_transpose, one, zero); //t_LL-1*G_L-1L-1
    //NemoUtils::toc(tic_toc_prefix + "couplingH x GR");

    //NemoUtils::tic(tic_toc_prefix + "temp1 x couplingH'");
    temp_matrix1->multBLAS(coupling, temp_matrix2, no_transpose, conj_transpose, one, zero);  //t_LL-1*G_L-1L-1*t_L-1L
    //NemoUtils::toc(tic_toc_prefix + "temp1 x couplingH'");

    delete temp_matrix1;
    temp_matrix1=NULL;
    delete coupling;
    coupling=NULL;

    //NemoUtils::tic(tic_toc_prefix + "temp2 x gR");
    temp_matrix2->multBLAS(half_way_g_matrix, temp_matrix1, no_transpose, no_transpose, one, zero);  //t_LL-1*G_L-1L-1*t_L-1L*g_LL
    //NemoUtils::toc(tic_toc_prefix + "temp2 x gR");

    delete temp_matrix2;
    temp_matrix2=NULL;

    //NemoUtils::tic(tic_toc_prefix + "gR x temp1");
    half_way_g_matrix->multBLAS(temp_matrix1, temp_matrix2, no_transpose, no_transpose, one, zero);  // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
    //NemoUtils::toc(tic_toc_prefix + "gR x temp1");
  }
  else if(mat_mult == Greensolver::blas_MIC)
  {
    std::complex<double> one = std::complex<double>(1.0,0.0);
    std::complex<double> zero = std::complex<double>(0.0,0.0);
    std::string no_transpose = std::string("N");
    std::string conj_transpose = std::string("C");

    //NemoUtils::tic(tic_toc_prefix + "couplingH x GR");
    PetscMatrixParallelComplex::multBLAS_MIC(coupling, exact_neighbor_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        c_it_momentum_offload_info->second); //t_LL-1*G_L-1L-1
    //NemoUtils::toc(tic_toc_prefix + "couplingH x GR");

    //NemoUtils::tic(tic_toc_prefix + "temp1 x couplingH'");
    PetscMatrixParallelComplex::multBLAS_MIC(temp_matrix1, coupling, temp_matrix2, no_transpose, conj_transpose, one, zero,
        c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L
    //NemoUtils::toc(tic_toc_prefix + "temp1 x couplingH'");

    delete temp_matrix1;
    temp_matrix1=NULL;
    delete coupling;
    coupling=NULL;

    //NemoUtils::tic(tic_toc_prefix + "temp2 x gR");
    PetscMatrixParallelComplex::multBLAS_MIC(temp_matrix2, half_way_g_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L*g_LL
    //NemoUtils::toc(tic_toc_prefix + "temp2 x gR");

    delete temp_matrix2;
    temp_matrix2=NULL;

    //NemoUtils::tic(tic_toc_prefix + "gR x temp1");
    PetscMatrixParallelComplex::multBLAS_MIC(half_way_g_matrix, temp_matrix1, temp_matrix2, no_transpose, no_transpose, one, zero,
        c_it_momentum_offload_info->second);  // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
    //NemoUtils::toc(tic_toc_prefix + "gR x temp1");
  }
  else if(mat_mult == Greensolver::pcl_MIC)
  {
    NEMO_ASSERT(offload_solver != NULL, get_name() + ": Offload solver has not been initialized!\n");

    std::complex<double> one = std::complex<double>(1.0,0.0);
    std::complex<double> zero = std::complex<double>(0.0,0.0);
    std::string no_transpose = std::string("N");
    std::string conj_transpose = std::string("C");

    //NemoUtils::tic(tic_toc_prefix + "couplingH x GR");
    NemoMICUtils::pcl_lapack_zgemm_interface(coupling, exact_neighbor_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        coupling->get_communicator(), *offload_solver, c_it_momentum_offload_info->second); //t_LL-1*G_L-1L-1
    //NemoUtils::toc(tic_toc_prefix + "couplingH x GR");

    //NemoUtils::tic(tic_toc_prefix + "temp1 x couplingH'");
    NemoMICUtils::pcl_lapack_zgemm_interface(temp_matrix1, coupling, temp_matrix2, no_transpose, conj_transpose, one, zero,
        temp_matrix1->get_communicator(), *offload_solver, c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L
    //NemoUtils::toc(tic_toc_prefix + "temp1 x couplingH'");

    delete temp_matrix1;
    temp_matrix1=NULL;
    delete coupling;
    coupling=NULL;

    //NemoUtils::tic(tic_toc_prefix + "temp2 x gR");
    NemoMICUtils::pcl_lapack_zgemm_interface(temp_matrix2, half_way_g_matrix, temp_matrix1, no_transpose, no_transpose, one, zero,
        temp_matrix2->get_communicator(), *offload_solver, c_it_momentum_offload_info->second);  //t_LL-1*G_L-1L-1*t_L-1L*g_LL
    //NemoUtils::toc(tic_toc_prefix + "temp2 x gR");

    delete temp_matrix2;
    temp_matrix2=NULL;

    //NemoUtils::tic(tic_toc_prefix + "gR x temp1");
    NemoMICUtils::pcl_lapack_zgemm_interface(half_way_g_matrix, temp_matrix1, temp_matrix2, no_transpose, no_transpose, one, zero,
        half_way_g_matrix->get_communicator(), *offload_solver, c_it_momentum_offload_info->second);  // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
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
    delete coupling;
    coupling = NULL;
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
    //NemoUtils::tic(tic_toc_prefix + "couplingH x GR");
    PropagationUtilities::supress_noise(this,coupling);
    PropagationUtilities::supress_noise(this,exact_neighbor_matrix);
    PetscMatrixParallelComplex::mult(*coupling,*exact_neighbor_matrix,&temp_matrix1); //t_LL-1*G_L-1L-1
    PropagationUtilities::supress_noise(this,temp_matrix1);
    //NemoUtils::toc(tic_toc_prefix + "couplingH x GR");

    PetscMatrixParallelComplex transpose_coupling(coupling->get_num_cols(),coupling->get_num_rows(),
        coupling->get_communicator());
    coupling->hermitian_transpose_matrix(transpose_coupling,MAT_INITIAL_MATRIX);
    delete coupling;
    coupling=NULL;

    //NemoUtils::tic(tic_toc_prefix + "temp1 x couplingH'");
    PropagationUtilities::supress_noise(this,&transpose_coupling);
    PetscMatrixParallelComplex::mult(*temp_matrix1,transpose_coupling,&temp_matrix2); //t_LL-1*G_L-1L-1*t_L-1L
    PropagationUtilities::supress_noise(this,temp_matrix2);
    //NemoUtils::toc(tic_toc_prefix + "temp1 x couplingH'");
    delete temp_matrix1;
    temp_matrix1=NULL;

    //NemoUtils::tic(tic_toc_prefix + "temp2 x gR");
    PropagationUtilities::supress_noise(this,half_way_g_matrix);
    PetscMatrixParallelComplex::mult(*temp_matrix2,*half_way_g_matrix,&temp_matrix1); //t_LL-1*G_L-1L-1*t_L-1L*g_LL
    PropagationUtilities::supress_noise(this,temp_matrix1);
    //NemoUtils::toc(tic_toc_prefix + "temp2 x gR");
    delete temp_matrix2;
    temp_matrix2=NULL;

    //NemoUtils::tic(tic_toc_prefix + "gR x temp1");
    PetscMatrixParallelComplex::mult(*half_way_g_matrix,*temp_matrix1,&temp_matrix2); // g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
    PropagationUtilities::supress_noise(this,temp_matrix2);
    //NemoUtils::toc(tic_toc_prefix + "gR x temp1");
  }


  delete temp_matrix1;
  temp_matrix1=NULL;

  result = new PetscMatrixParallelComplex(*half_way_g_matrix);
  result->add_matrix(*temp_matrix2,SAME_NONZERO_PATTERN); //g_LL + g_LL*t_LL-1*G_L-1L-1*t_L-1L*g_LL
  delete temp_matrix2;
  temp_matrix2=NULL;
  PropagationUtilities::supress_noise(this,result);
  //NemoUtils::toc(tic_toc_GLL);
  //-------------------------------------
  //Yu: end of new version
  //=====================================

  //update the list of allocated propagation matrices
  output_Propagator->allocated_momentum_Propagator_map[momentum_point]=true;

  set_job_done_momentum_map(&(output_Propagator->get_name()), &momentum_point, true);

  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::solve_interdomain_Green_RGF(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::solve_interdomain_Green_RGF ";
  NemoUtils::tic(tic_toc_prefix);

  //std::string prefix = "Greensolver(\""+this->get_name()+"\")::solve_interdomain_Green_RGF ";
  {
    //NOTE about the domains: this domain is "1", target_domain is "L", coupling_domain="L-1"

    //find the momentum that the Hamilton constructor can read out of momentum_point
    //std::set<unsigned int> Hamilton_momentum_indices;
    //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
    //find_Hamiltonian_momenta(output_Propagator,pointer_to_Hamilton_momentum_indices);
    ////NEMO_ASSERT(pointer_to_Hamilton_momentum_indices!=NULL ||momentum_point.size()==1,prefix+"Please define \"Hamilton_momenta\"\n");
    //set_valley(output_Propagator, momentum_point, Hamilton_Constructor); //Hamilton_Constructor added by Bozidar
    //NemoMeshPoint temp_momentum(0,std::vector<double>(3,0.0));
    //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_momentum=momentum_point[*(Hamilton_momentum_indices.begin())];

    //std::string tic_toc_H= tic_toc_prefix+" get device Hamiltonian and DOFmap ";
    //NemoUtils::tic(tic_toc_H);
    //1a. get the Hamiltonian and the DOFmap of the full device
    std::string variable_name = "full_device_Hamiltonian_constructor";
    NEMO_ASSERT(options.check_option(variable_name),tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string full_H_constructor_name = options.get_option(variable_name,std::string(""));
    Simulation* full_H_constructor = this->find_simulation(full_H_constructor_name);
    NEMO_ASSERT(full_H_constructor!=NULL,tic_toc_prefix+"simulation \""+full_H_constructor_name+"\" not found\n");
    const DOFmapInterface& device_DOFmap = full_H_constructor->get_const_dof_map(get_const_simulation_domain());

    //1b. get some device matrix characteristics
    unsigned int number_of_super_rows = device_DOFmap.get_global_dof_number();
    unsigned int number_of_super_cols = number_of_super_rows;

    NEMO_ASSERT(number_of_super_rows==number_of_super_cols,tic_toc_prefix+"rectangular matrix received for the super-domain\n");

    //NemoUtils::toc(tic_toc_H);

    //2. get the DOFmap of the target domain "L"
    variable_name = "target_domain_Hamiltonian_constructor";
    NEMO_ASSERT(options.check_option(variable_name),tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string target_H_constructor_name = options.get_option(variable_name,std::string(""));
    Simulation* target_H_constructor = this->find_simulation(target_H_constructor_name);
    NEMO_ASSERT(target_H_constructor!=NULL,tic_toc_prefix+"simulation \""+target_H_constructor_name+"\" not found\n");
    const Domain* target_domain=target_H_constructor->get_const_simulation_domain();
    if(options.check_option("target_domain"))
    {
      std::string temp_domain_name=options.get_option("target_domain",std::string(""));
      target_domain=Domain::get_domain(temp_domain_name);
      NEMO_ASSERT(target_domain!=NULL,tic_toc_prefix+"have not found target_domain \""+temp_domain_name+"\"\n");
    }
    const DOFmapInterface& target_DOFmap = target_H_constructor->get_const_dof_map(target_domain);
    //3. if the domains "1" and "L" are equal - throw exception
    if(target_domain==get_const_simulation_domain())
      throw std::invalid_argument(tic_toc_prefix+"domain of \""+target_H_constructor_name+"\" and this domain are equal\n");

    //std::string tic_toc_couplingH= tic_toc_prefix+" get Hamiltonian and DOFmap for L-1";
    //NemoUtils::tic(tic_toc_couplingH);
    //4. get the Hamiltonian and the DOFmap of the coupling domain "L-1"
    variable_name ="coupling_domain_Hamiltonian_constructor"; //"L-1" in Eq.64 of J. Appl. Phys. 81, 7845 (1997)
    if(!options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string coupling_H_constructor_name=options.get_option(variable_name,std::string(""));
    Simulation* coupling_H_constructor = this->find_simulation(coupling_H_constructor_name);
    const Domain* coupling_domain = coupling_H_constructor->get_const_simulation_domain();
    if(options.check_option("coupling_domain"))
    {
      std::string temp_domain_name=options.get_option("coupling_domain",std::string(""));
      coupling_domain=Domain::get_domain(temp_domain_name);
      NEMO_ASSERT(coupling_domain!=NULL,tic_toc_prefix+"have not found target_domain \""+temp_domain_name+"\"\n");
    }
    DOFmapInterface& coupling_DOFmap = coupling_H_constructor->get_dof_map(coupling_domain);

    DOFmapInterface* large_coupling_DOFmap=NULL; //will store the DOFmap for the matrix of the target (L) domain and the coupling (L-1) domain

    //NemoUtils::toc(tic_toc_couplingH);


    //========================================
    //Yu He: new version -- avoid super matrix
    //----------------------------------------
    //std::string tic_toc_HLLm1= tic_toc_prefix+" get interdomain Hamiltonian and DOFmap for L,L-1";
    //NemoUtils::tic(tic_toc_HLLm1);


    //5a. get the interdomain Hamiltonian and the large_interdomain DOFmap ( H_(L,L-1) )
    PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;  //should become a matrix in the upper right corner
    PetscMatrixParallelComplex* temp_overlap = NULL;
    PetscMatrixParallelComplex* overlap=NULL;
    {
      std::vector<NemoMeshPoint> sorted_momentum;
      QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum,options,momentum_mesh_types,target_H_constructor);
      large_coupling_DOFmap=&coupling_DOFmap;
      DOFmapInterface* temp_pointer=large_coupling_DOFmap;
      target_H_constructor->get_data(std::string("Hamiltonian"), sorted_momentum, coupling_domain, coupling_Hamiltonian,large_coupling_DOFmap,target_domain);
      if(temp_pointer!=large_coupling_DOFmap)
        delete large_coupling_DOFmap;
	    large_coupling_DOFmap=NULL;
	    target_H_constructor->get_data(std::string("overlap_matrix_coupling"),sorted_momentum,coupling_domain, temp_overlap, large_coupling_DOFmap,target_domain);
      if(temp_overlap!=NULL)
        overlap=new PetscMatrixParallelComplex(*temp_overlap);
      delete large_coupling_DOFmap;
      //coupling_Hamiltonian->save_to_matlab_file("Ham_int.m");
      //temp_overlap->save_to_matlab_file("S_int.m");

      if(overlap!=NULL)
      {
        std::complex<double> energy = std::complex<double>(PropagationUtilities::read_energy_from_momentum(this,momentum_point,output_Propagator),0.0);
        //cerr<<energy;
        *overlap *= -energy; //-ES
        coupling_Hamiltonian->add_matrix(*overlap, DIFFERENT_NONZERO_PATTERN, std::complex<double>(1.0, 0.0));
        delete overlap;
        overlap=NULL;
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

    //5b. extract sub block of coupling Hamiltonian
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

    //bool LRA_RGF = options.get_option("use_LRA", bool(false));
    //if(!LRA_RGF)
    {
      coupling= new PetscMatrixParallelComplex(number_of_sub_couple_rows,number_of_sub_couple_cols,
          get_simulation_domain()->get_communicator());
      coupling->set_num_owned_rows(number_of_own_target_rows);
    }
    vector<int> rows_diagonal(number_of_sub_couple_rows,0);
    vector<int> rows_offdiagonal(number_of_sub_couple_rows,0);

    //if(!LRA_RGF)
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
    //else //use LRA_RGF
    //{

    //  //if(use_LRA) //LRA needs the couplling Hamiltonian of the untransformed domain but target Hamiltonian of reduced space is needed else where

    //  variable_name = "target_domain_Hamiltonian_constructor_for_coupling";
    //  NEMO_ASSERT(options.check_option(variable_name),prefix+"please define \""+variable_name+"\"\n");
    //  std::string target_H_constructor_name = options.get_option(variable_name,std::string(""));
    //  Simulation* target_H_constructor_for_coupling = this->find_simulation(target_H_constructor_name);
    //  NEMO_ASSERT(target_H_constructor_for_coupling!=NULL,prefix+"simulation \""+target_H_constructor_name+"\" not found\n");
    //  unsigned int number_of_target_rows = target_H_constructor_for_coupling->get_dof_map().get_global_dof_number();
    //  unsigned int number_of_target_cols = number_of_target_rows;
    //  unsigned int number_of_own_target_rows = target_H_constructor_for_coupling->get_dof_map().get_number_of_dofs();

    //  delete coupling;
    //  InputOptions trans_opt = Hamilton_Constructor->get_options();
    //  //double ratio = trans_opt.get_option("ratio_of_eigenvalues_to_be_solved", double(1.0));

    //  unsigned int number_of_sub_couple_rows = number_of_target_rows;
    //  unsigned int number_of_sub_couple_cols = number_of_couple_cols-number_of_target_cols;
    //  coupling= new PetscMatrixParallelComplex(number_of_sub_couple_rows,number_of_sub_couple_cols,
    //      get_simulation_domain()->get_communicator());
    //  coupling->set_num_owned_rows(number_of_own_target_rows);

    //  std::vector<int> temp_rows(number_of_sub_couple_rows);
    //  std::vector<int> temp_cols(number_of_sub_couple_cols);
    //  int num_of_target_cols = number_of_target_cols;
    //  for(unsigned int i=0; i<number_of_sub_couple_rows; i++)
    //    temp_rows[i]=i;
    //  for(unsigned int i=0; i<number_of_sub_couple_cols; i++)
    //    temp_cols[i]=i+num_of_target_cols;

    //  //delete coupling;
    //  coupling_Hamiltonian->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, coupling);
    //  coupling->assemble();

    //  //Hamilton_Constructor->get_data("one_sided_subcoupling_transformation", &momentum_point, coupling);
    //  Hamilton_Constructor->get_data("subcoupling_Hamiltonian", &momentum_point, coupling);

    //}


    delete coupling_Hamiltonian;
    coupling_Hamiltonian=NULL;
    //NemoUtils::toc(tic_toc_prefix + "get submatrix from coupling Hamiltonian");

    //std::string tic_toc_GLm11= tic_toc_prefix+" get G_L-1,1";
    //NemoUtils::tic(tic_toc_GLm11);
    //6. get the exact_greens_function (G_L-1,1 in Eq. 64 of J. Appl. Phys. 81, 7845 (1997))
    variable_name="exact_greens_function";
    if(!options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string exact_green_name = options.get_option(variable_name,std::string(""));
    //Simulation* exact_green_source = find_source_of_data(exact_green_name);
    GreensfunctionInterface* exact_green_source=find_source_of_Greensfunction(exact_green_name);
    PetscMatrixParallelComplex* exact_matrix=NULL;
    if(particle_type_is_Fermion)
      exact_green_source->get_Greensfunction(momentum_point,exact_matrix,&coupling_DOFmap,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())),
                                             NemoPhys::Fermion_retarded_Green);
    else
      exact_green_source->get_Greensfunction(momentum_point,exact_matrix,&coupling_DOFmap,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())),
                                             NemoPhys::Boson_retarded_Green);
    //exact_green_source->get_data(exact_green_name,&momentum_point,exact_matrix,&coupling_DOFmap,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    int start_own_exact_rows;
    int end_own_exact_rows;
    exact_matrix->get_ownership_range(start_own_exact_rows,end_own_exact_rows);
    //NemoUtils::toc(tic_toc_GLm11);

    //std::string tic_toc_halfg= tic_toc_prefix+" get half way greens function";
    //NemoUtils::tic(tic_toc_halfg);
    //7. get the half_way_greens_function and the DOFmap
    variable_name="half_way_greens_function";
    if(!options.check_option(variable_name))
      throw std::invalid_argument(tic_toc_prefix+"please define \""+variable_name+"\"\n");
    std::string half_way_green_name = options.get_option(variable_name,std::string(""));
    //Simulation* half_way_green_source = find_source_of_data(half_way_green_name);
    GreensfunctionInterface* half_way_green_source=find_source_of_Greensfunction(half_way_green_name);
    PetscMatrixParallelComplex* half_way_g_matrix=NULL;
    if(particle_type_is_Fermion)
      half_way_green_source->get_Greensfunction(momentum_point,half_way_g_matrix,&target_DOFmap,NULL,NemoPhys::Fermion_retarded_Green);
    else
      half_way_green_source->get_Greensfunction(momentum_point,half_way_g_matrix,&target_DOFmap,NULL,NemoPhys::Boson_retarded_Green);
    //half_way_green_source->get_data(half_way_green_name,&momentum_point,half_way_g_matrix,&target_DOFmap);
    //NemoUtils::toc(tic_toc_halfg);

    //8. do the actual multiplication (Eq.(64) of J. Appl. Phys. 81, 7845 (1997))
    PetscMatrixParallelComplex* temp_matrix=NULL;
    //std::string tic_toc_mult1= tic_toc_prefix+" half_way_g x couplingH";
    //NemoUtils::tic(tic_toc_mult1);
    Mat_mult_method mat_mult = Greensolver::petsc;

    std::string invert_solver_option;
    std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;
    if (offload_solver_initialized && offload_solver->offload)
    {
      c_it_momentum_offload_info = offload_solver->offloading_momentum_map.find(momentum_point);

      if(c_it_momentum_offload_info->second.offload_to == ResourceUtils::GPU)
        mat_mult = Greensolver::magma;
      else
        mat_mult = Greensolver::petsc;
    }

#ifdef MAGMA_ENABLE
    magmaDoubleComplex* temp_gpu = NULL;
#endif

    if(mat_mult == Greensolver::petsc)
    {
      PropagationUtilities::supress_noise(this,half_way_g_matrix);
      PropagationUtilities::supress_noise(this,coupling);
      PetscMatrixParallelComplex::mult(*half_way_g_matrix,*coupling,&temp_matrix);
      PropagationUtilities::supress_noise(this,temp_matrix);
    }
    else
    {
#ifdef MAGMA_ENABLE
      PetscMatrixParallelComplex::multMAGMA(*half_way_g_matrix,MagmaNoTrans,NULL,
                                            *coupling,MagmaNoTrans,NULL,
                                            &temp_matrix,false,&temp_gpu);
#else
      NEMO_EXCEPTION("solve_interdomain_Green_RGF: Can't use MAGMA without compiling with MAGMA!");
#endif
    }
    //NemoUtils::toc(tic_toc_mult1);
    delete coupling;
    coupling=NULL;
    //std::string tic_toc_mult2= tic_toc_prefix+" temp_matrix x full_exact";
    //NemoUtils::tic(tic_toc_mult2);
    if(mat_mult == Greensolver::petsc)
    {
      PropagationUtilities::supress_noise(this,exact_matrix);
      PetscMatrixParallelComplex::mult(*temp_matrix,*exact_matrix,&result);
      PropagationUtilities::supress_noise(this,result);
    }
    else
    {
#ifdef MAGMA_ENABLE
      PetscMatrixParallelComplex::multMAGMA(*temp_matrix,MagmaNoTrans,&temp_gpu,
                                            *exact_matrix,MagmaNoTrans,NULL,
                                            &result,true,NULL);
      magma_free(temp_gpu);
#else
      NEMO_EXCEPTION("solve_interdomain_Green_RGF: Can't use MAGMA without compiling with MAGMA!");
#endif
    }

    //NemoUtils::toc(tic_toc_mult2);
    delete temp_matrix;
    temp_matrix=NULL;
    //----------------------------------------
    //Yu He: end of new version
    //========================================
  }

  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::do_solve_spectral(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_solve_spectral ";
  NemoUtils::tic(tic_toc_prefix);
  //std::string prefix="Self_enGreensolverergy(\""+this->get_name()+"\")::do_solve_spectral(): ";
  activate_regions();
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=output_Propagator->propagator_map.find(momentum_point);
  if(matrix_it!=output_Propagator->propagator_map.end())
    temp_matrix=matrix_it->second;
  //do_solve_spectral(output_Propagator,momentum_point,temp_matrix);
  PropagationUtilities::do_solve_spectral(this, output_Propagator, momentum_point,temp_matrix);
  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum_point,true);

  NEMO_ASSERT(is_ready(output_Propagator->get_name(),momentum_point),tic_toc_prefix+"still not ready\n");
  write_propagator(output_Propagator->get_name(),momentum_point, temp_matrix);
  conclude_after_do_solve(output_Propagator,momentum_point);
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::solve_retarded_Green_LRA(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum,
    PetscMatrixParallelComplex*& inverse_Green, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::solve_retarded_Green_LRA ";
  NemoUtils::tic(tic_toc_prefix);

  msg.set_level(MsgLevel(4));
  msg<<"Greensolver(\""+this->get_name()+"\")::solve_retarded_Green_LRA\n"<<std::endl;
  double time1=NemoUtils::get_time();
  //figure out which index of momentum is the energy
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
  std::string energy_name=std::string("");
  bool complex_energy=false;
  for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
    if(momentum_name_it->second==NemoPhys::Energy)
      energy_name=momentum_name_it->first;
    else if(momentum_name_it->second==NemoPhys::Complex_energy)
    {
      energy_name=momentum_name_it->first;
      complex_energy=true;
    }
  unsigned int energy_index=0;
  for (unsigned int i=0; i<output_Propagator->momentum_mesh_names.size(); i++)
    if(output_Propagator->momentum_mesh_names[i].find(std::string("energy"))!=std::string::npos)
      energy_index=i;
  //-------------------------------------------------------
  NemoMeshPoint energy_point=momentum[energy_index];
  std::complex<double> energy;
  if(!complex_energy)
    energy=std::complex<double> (energy_point.get_x(),0.0);
  else
    energy=std::complex<double> (energy_point.get_x(),energy_point.get_y());

  double number_of_set_num_ratio=options.get_option("ratio_of_eigenvalues_to_be_solved",double(1.0));
  string transform_type=options.get_option("transform_type",string("none"));
  string solver_type=options.get_option("solver_type",string("lapack"));
  string precond_type=options.get_option("precond_type",string(""));
  string linsolver=options.get_option("linsolver",string(""));
  int number_of_set_num=inverse_Green->get_num_rows();
  number_of_set_num*=number_of_set_num_ratio;
  int ncv_number=options.get_option("ncv",int(2*number_of_set_num));
  msg<<"Greensolver(\""+this->get_name()+"\")::solving eigenproblem\n"<<std::endl;
  EigensolverSlepc solver;
  solver.set_matrix(inverse_Green);
  solver.set_num_evals(number_of_set_num);
  solver.set_spectral_region(EigensolverSlepc::smallest_magnitude);
  solver.set_shift_sigma(energy);
  solver.set_problem_type("eps_non_hermitian");
  solver.set_transformation_type(transform_type);
  solver.set_precond_type(precond_type);
  solver.set_ncv(ncv_number);
  solver.set_solver_type(solver_type);
  solver.set_linsolver_type(linsolver);
  double time2=NemoUtils::get_time();
  msg<<"\n\n prepare for eigensolver: "<<time2-time1<<"\n\n";
  solver.solve();
  double time3=NemoUtils::get_time();
  msg<<"\n\n solving eigenproblem: "<<time3-time2<<"\n\n";
  msg<<"Greensolver(\""+this->get_name()+"\")::getting Green's function\n"<<std::endl;
  const   vector<std::complex<double> >* M_values = solver.get_eigenvalues();
  unsigned int number_of_eigenvalues = M_values->size();

  unsigned int number_of_vectors_size = inverse_Green->get_num_rows();
  PetscMatrixParallelComplex* Basisfunction = NULL;
  solve_retarded_Green_LRA_initialize_temp_matrix(number_of_vectors_size,number_of_vectors_size,number_of_vectors_size,0,number_of_vectors_size,
      number_of_eigenvalues,Basisfunction);
  cplx temp_val(0.0,0.0);
  for(unsigned int i=0; i<number_of_eigenvalues; i++)
  {
    PetscVectorNemo< std::complex< double > >& vec = solver.get_eigenvector(i);
    int vec_size = vec.get_local_size();

    for(int j=0; j<vec_size; j++)
    {
      temp_val = vec.get(j);
      if(abs(temp_val)>1e-8)
        Basisfunction->set(j,i,temp_val);
    }
  }
  Basisfunction->assemble();//V

  PetscMatrixParallelComplex* inverse_Green_eigen1 = NULL;
  PetscMatrixParallelComplex* inverse_Green_eigen2 = NULL;
  PetscMatrixParallelComplex::mult(*inverse_Green,*Basisfunction,&inverse_Green_eigen1);//invGR_eigen=invGR*V
  Basisfunction->hermitian_transpose_matrix(*Basisfunction,MAT_REUSE_MATRIX);//V'
  PetscMatrixParallelComplex::mult(*Basisfunction,*inverse_Green_eigen1,&inverse_Green_eigen2);//invGR_eigen=V'*invGR*V
  delete inverse_Green_eigen1;
  inverse_Green_eigen1 = NULL;
  solve_retarded_Green_LRA_initialize_temp_matrix(number_of_eigenvalues,number_of_eigenvalues,number_of_eigenvalues,0,number_of_eigenvalues,
      number_of_eigenvalues,inverse_Green_eigen1);
  solve_retarded_Green_LRA_set_matrix_elements(0,number_of_eigenvalues,0,number_of_eigenvalues,0,0,inverse_Green_eigen2,inverse_Green_eigen1);
  PetscMatrixParallelComplex* Green_eigen = NULL;
  PetscMatrixParallelComplex* super_Green_eigen = NULL;
  solve_retarded_Green_LRA_initialize_temp_matrix(number_of_vectors_size,number_of_vectors_size,number_of_eigenvalues,0,number_of_eigenvalues,
      number_of_eigenvalues,super_Green_eigen);
  double time4=NemoUtils::get_time();
  msg<<"\n\n perparing for small GR_inv: "<<time4-time3<<"\n\n";
  if(number_of_eigenvalues<=1)
  {
    std::complex<double> GR_matrix_1(0.0,0.0);
    if(number_of_eigenvalues==1)
      GR_matrix_1 = 1.0/(inverse_Green_eigen1->get(0,0));
    super_Green_eigen->set(0,0,GR_matrix_1);
    super_Green_eigen->assemble();
  }
  else
  {
    PetscMatrixParallelComplex::invert(*inverse_Green_eigen1,&Green_eigen);//GR_eigen=inv(invGR_eigen)
    solve_retarded_Green_LRA_set_matrix_elements(0,number_of_eigenvalues,0,number_of_eigenvalues,0,0,Green_eigen,super_Green_eigen);
  }
  double time5=NemoUtils::get_time();
  msg<<"\n\n GR=inv(GR_inv): "<<time5-time4<<"\n\n";
  PetscMatrixParallelComplex* retGreen1 = NULL;
  PetscMatrixParallelComplex* retGreen2 = NULL;
  PetscMatrixParallelComplex::mult(*super_Green_eigen,*Basisfunction,&retGreen1);//retGR=GR_eigen*V'
  Basisfunction->hermitian_transpose_matrix(*Basisfunction,MAT_REUSE_MATRIX);//V
  PetscMatrixParallelComplex::mult(*Basisfunction,*retGreen1,&retGreen2);//retGR=V*GR_eigen*V'
  result = new PetscMatrixParallelComplex (*retGreen2);
  double time6=NemoUtils::get_time();
  msg<<"\n\n turn into GR with original size: "<<time6-time5<<"\n\n";
  delete Basisfunction;
  delete inverse_Green_eigen1;
  delete inverse_Green_eigen2;
  delete Green_eigen;
  delete super_Green_eigen;
  delete retGreen1;
  delete retGreen2;
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::solve_retarded_Green_LRA_initialize_temp_matrix(const int number_of_rows, const int number_of_cols, const int /*number_of_own_rows*/,
    const int start_own_rows, const int end_own_rows, const int number_of_nonzero_cols, PetscMatrixParallelComplex*& result_matrix)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::solve_retarded_Green_LRA_initialize_temp_matrix ";
  NemoUtils::tic(tic_toc_prefix);

  result_matrix = new PetscMatrixParallelComplex(number_of_rows,number_of_cols,
      get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
  //set matrix pattern
  result_matrix->set_num_owned_rows(number_of_rows);//single cpu
  for(int i=0; i<start_own_rows; i++)
    result_matrix->set_num_nonzeros(i,0,0);
  for(int i=start_own_rows; i<end_own_rows && i<number_of_rows; i++)
    result_matrix->set_num_nonzeros(i,number_of_nonzero_cols,0);
  for(int i=end_own_rows; i<number_of_rows; i++)
    result_matrix->set_num_nonzeros(i,0,0);
  result_matrix->allocate_memory();
  result_matrix->set_to_zero();
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::solve_retarded_Green_LRA_set_matrix_elements(const int start_rows, const int end_rows, const int start_cols, const int end_cols,
    const int source_matrix_rows, const int source_matrix_cols, PetscMatrixParallelComplex*& source_matrix, PetscMatrixParallelComplex*& result_matrix)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::solve_retarded_Green_LRA_set_matrix_elements ";
  NemoUtils::tic(tic_toc_prefix);

  cplx temp_val(0.0,0.0);
  for(int i=start_rows; i<end_rows; i++)
    for(int j=start_cols; j<end_cols; j++)
    {
      temp_val=source_matrix->get(i-source_matrix_rows,j-source_matrix_cols);
      if(abs(temp_val)>1e-8)
        result_matrix->set(i,j,source_matrix->get(i-source_matrix_rows,j-source_matrix_cols));
    }
  result_matrix->assemble();
  NemoUtils::toc(tic_toc_prefix);
}



void Greensolver::solve_retarded_Green_analytical(Propagator*& output_Propagator,const std::vector<NemoMeshPoint>& /*momentum_point*/,
    PetscMatrixParallelComplex*& /*result*/)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::solve_retarded_Green_analytical ";
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix="Greensolver(\""+this->get_name()+"\")::solve_retarded_Green_analytical(): ";
  //1. get the type of the output_Propagator (either Boson or Fermion retarded self-energy)
  std::map<std::string, NemoPhys::Propagator_type>::const_iterator Propagator_type_c_it=Propagator_types.find(output_Propagator->get_name());
  NEMO_ASSERT(Propagator_type_c_it!=Propagator_types.end(),prefix+"have not found the propagator type of \""+output_Propagator->get_name()+"\"\n");

  NemoUtils::toc(tic_toc_prefix);
}


void Greensolver::do_solve_lesser(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_solve_retarded";
  NemoUtils::tic(tic_toc_prefix);
  activate_regions();
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=output_Propagator->propagator_map.find(momentum_point);
  if(matrix_it!=output_Propagator->propagator_map.end())
    temp_matrix=matrix_it->second;
  do_solve_lesser(output_Propagator,momentum_point,temp_matrix);
  NEMO_ASSERT(is_ready(output_Propagator->get_name(),momentum_point),tic_toc_prefix+"still not ready\n");
  write_propagator(output_Propagator->get_name(),momentum_point, temp_matrix);
  conclude_after_do_solve(output_Propagator,momentum_point);
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::do_solve_lesser(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                  PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_solve_lesser ";
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix="Greensolver(\""+this->get_name()+"\")::do_solve_lesser(): ";

  //1. find the retarded and/or the advanced Green's functions and the set of lesser self-energies in the list of Propagators
  const Propagator* retarded_Green=NULL;
  const Propagator* advanced_Green=NULL;
  if(exact_GR_solver!=NULL)
  {
    PropagatorInterface* temp_source=dynamic_cast<PropagatorInterface*>(exact_GR_solver);
    NEMO_ASSERT(temp_source!=NULL,prefix+exact_GR_solver->get_name()+" is not a PropagatorInterface");
    NemoPhys::Propagator_type temp_Propagator_type=NemoPhys::Fermion_retarded_Green;
    if(!particle_type_is_Fermion)
      temp_Propagator_type=NemoPhys::Boson_retarded_Green;
    temp_source->get_Propagator(retarded_Green, &temp_Propagator_type);
  }
  std::set<const Propagator*> set_lesser_self;
  std::set<Simulation*>::iterator sigmaL_it=contact_sigmaL_solvers.begin();
  for(;sigmaL_it!=contact_sigmaL_solvers.end();++sigmaL_it)
  {
    Simulation* temp_simulation=*sigmaL_it;
    PropagatorInterface* temp_source=dynamic_cast<PropagatorInterface*>(temp_simulation);
    NEMO_ASSERT(temp_source!=NULL,prefix+(*sigmaL_it)->get_name()+" is not a PropagatorInterface");
    const Propagator* temp_Propagator=NULL;
    NemoPhys::Propagator_type temp_Propagator_type=NemoPhys::Fermion_lesser_self;
    if(!particle_type_is_Fermion)
      temp_Propagator_type=NemoPhys::Boson_lesser_self;
    temp_source->get_Propagator(temp_Propagator, &temp_Propagator_type);
    set_lesser_self.insert(temp_Propagator);
  }
  if(scattering_sigmaL_solver!=NULL)
  {
    PropagatorInterface* temp_source=dynamic_cast<PropagatorInterface*>(scattering_sigmaL_solver);
    NEMO_ASSERT(temp_source!=NULL,prefix+scattering_sigmaL_solver->get_name()+" is not a PropagatorInterface");
    const Propagator* temp_Propagator=NULL;
    NemoPhys::Propagator_type temp_Propagator_type=NemoPhys::Fermion_lesser_self;
    if(!particle_type_is_Fermion)
      temp_Propagator_type=NemoPhys::Boson_lesser_self;
    temp_source->get_Propagator(temp_Propagator, &temp_Propagator_type);
    set_lesser_self.insert(temp_Propagator);
  }

  //std::map<std::string, const Propagator*>::iterator it=Propagators.begin();
  //for(; it!=Propagators.end(); it++)
  //{
  //  //1.make sure that the c_it->second is useable
  //  if(it->second==NULL)
  //  {
  //    Simulation* data_source=find_source_of_data(it->first);
  //    data_source->get_data(it->first,it->second);
  //    //pointer_to_Propagator_Constructors->find(it->first)->second->get_data(it->first,it->second);
  //  }

  //  NemoPhys::Propagator_type p_type=Propagator_types.find(it->first)->second;
  //  //1. find all lesser self-energy pointers and store them in a set
  //  if(p_type == NemoPhys::Fermion_lesser_self ||p_type == NemoPhys::Boson_lesser_self)
  //    set_lesser_self.insert(it->second);
  //  //2. find the retarded and/or advanced Green's function
  //  else if(p_type == NemoPhys::Fermion_retarded_Green ||p_type == NemoPhys::Boson_retarded_Green)
  //    retarded_Green=it->second;
  //  else if(p_type == NemoPhys::Fermion_advanced_Green ||p_type == NemoPhys::Boson_advanced_Green)
  //    advanced_Green=it->second;
  //}
  //3. do the matrix-matrix products in another method
  //do_solve_lesser(output_Propagator, momentum_point, result, set_lesser_self, retarded_Green, advanced_Green);
  std::string solve_option=options.get_option(output_Propagator->get_name()+"_solution",std::string("exact"));
  PropagationUtilities::do_solve_lesser(this, solve_option, output_Propagator, momentum_point, result,
                                        set_lesser_self,retarded_Green, advanced_Green);
  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum_point, true);
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::do_init_lesser(void)
{
}



void Greensolver::do_init_retarded(void)
{

}




void Greensolver::do_init_inverse(void)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_init_inverse ";
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix="Greensolver(\""+get_name()+"\")::do_init_inverse: ";

  std::map<std::string, NemoPhys::Momentum_type>::const_iterator c_it=momentum_mesh_types.begin();
  std::string energy_name=std::string("");
  for (; c_it!=momentum_mesh_types.end(); ++c_it)
  {
    //check that there exists only one NemoMesh of type energy
    if(c_it->second==NemoPhys::Energy||c_it->second==NemoPhys::Complex_energy)
    {
      if(energy_name=="")
        energy_name=c_it->first;
      else
        throw std::invalid_argument("Greensolver(\""+get_name()+"\")::do_init_inverse: found more than one energy: "+energy_name+" & "+c_it->first+"\n");
    }
  }
  NEMO_ASSERT(energy_name!=std::string(""),"Greensolver(\""+get_name()+"\")::do_init_inverse: found no energy\n");


  //check that there is only one writeable Propagator
  //std::map<std::string,Propagator*>::const_iterator c_it2=writeable_Propagators.begin();
  //NEMO_ASSERT(c_it2!=writeable_Propagators.end(),prefix+"received empty writeable Propagator list\n");
  //++c_it2;
  //NEMO_ASSERT(c_it2==writeable_Propagators.end(),"Greensolver(\""+get_name()+"\")::do_init_inverse: found more than one writeable Propagator\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Greensolver::get_Hamiltonian_for_inverseG(const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& temp_matrix,
    PetscMatrixParallelComplex*& overlap_matrix)
{
  std::string prefix = "Greensolver(\""+this->get_name()+"\")::get_Hamiltonian_for_inverseG ";
  if(particle_type_is_Fermion)
  {
    NEMO_ASSERT(momentum_point.size()!=0&&momentum_point.size()<=3,prefix+"\"Hamilton_momenta\" is empty or its size is over 3\n");
    std::vector<NemoMeshPoint> sorted_momentum_point;
    
    if(!options.get_option("RF_with_kmesh",false))
    {
      QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum_point,options,momentum_mesh_types,Hamilton_Constructor);
    }
    else
    {
      for(unsigned int ii=0; ii<momentum_point.size(); ii++)
        if(momentum_point[ii].get_dim()==3)
          sorted_momentum_point.push_back(momentum_point[ii]);

      vector<string> Hamilton_momenta;
      options.get_option("Hamilton_momenta",Hamilton_momenta);
      InputOptions& writeable_solver_options = Hamilton_Constructor->get_reference_to_options();
      writeable_solver_options.set_option("quantum_number_order",Hamilton_momenta);
    }

    Hamilton_Constructor->get_data(std::string("Hamiltonian"),temp_matrix,sorted_momentum_point,avoid_copying_hamiltonian,
                                   get_const_simulation_domain()); //result=H
    //temp_matrix->assemble();
    //temp_matrix->save_to_matlab_file("Ham.m");
    Hamilton_Constructor->get_data(string("overlap_matrix"),overlap_matrix,sorted_momentum_point, false,
                                   get_const_simulation_domain()); //use the same get_data as the previous line
  }
  else
  {
    std::set<unsigned int> Hamilton_momentum_indices;
    std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
    NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
    PropagationUtilities::find_Hamiltonian_momenta(this,writeable_Propagator,pointer_to_Hamilton_momentum_indices);
    if(pointer_to_Hamilton_momentum_indices!=NULL)
      temp_NemoMeshPoint=momentum_point[*(pointer_to_Hamilton_momentum_indices->begin())];
    Hamilton_Constructor->get_data(std::string("DynamicalMatrix"),temp_NemoMeshPoint,temp_matrix,avoid_copying_hamiltonian); //result=H
    NEMO_ASSERT(temp_matrix!=NULL,prefix+"received NULL for the Hamiltonian/dynamical Matrix\n");
    delete overlap_matrix;
    overlap_matrix=NULL;
  }
}

void Greensolver::shift_Hamiltonian_with_energy(const std::complex<double>& energy, PetscMatrixParallelComplex*& temp_overlap,
    PetscMatrixParallelComplex*& matrix_to_get_shifted)
{
  if(temp_overlap==NULL)
  {
    if(particle_type_is_Fermion)
    {
      matrix_to_get_shifted->matrix_diagonal_shift(energy);//result=E-H
    }
    else
    {
      matrix_to_get_shifted->matrix_diagonal_shift(energy*energy);//result=E^2-H
    }
  }
  else
  {
    PetscMatrixParallelComplex temp_S=PetscMatrixParallelComplex(*temp_overlap);
    temp_S *= energy; //E*S
    matrix_to_get_shifted->add_matrix(temp_S, DIFFERENT_NONZERO_PATTERN,std::complex<double> (1.0,0.0));//result=ES-H
  }
}

void Greensolver::get_retarded_sigma(const std::string& propagator_name, const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result,
                                     const DOFmapInterface* subdofmap)
{
  Simulation* data_source=find_source_of_data(propagator_name);
  data_source->get_data(propagator_name,&momentum_point,result,subdofmap);
}

void Greensolver::do_solve_inverse(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                   PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::do_solve_inverse2 ";
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix = "Greensolver(\""+this->get_name()+"\")::do_solve_inverse ";

  PetscMatrixParallelComplex* temp_matrix = NULL;
  PetscMatrixParallelComplex* temp_overlap = NULL;

  get_Hamiltonian_for_inverseG(momentum_point, temp_matrix,temp_overlap);
  PetscMatrixParallelComplex* overlap=temp_overlap;
  if(debug_output && overlap!=NULL)
  {
    std::string temp_file_string;
    {
      const std::vector<NemoMeshPoint>* temp_pointer = &(momentum_point);
      PropagationUtilities::translate_momentum_vector(this,temp_pointer, temp_file_string);
    }
    overlap->save_to_matlab_file("overlap"+temp_file_string+".m");
  }
  //temp_matrix->assemble();
  //temp_matrix->save_to_matlab_file("Ham.m");

  if(result!=NULL)
  {
    delete result;
    result=NULL;
  }

  NEMO_ASSERT(temp_matrix!=NULL,prefix+"received NULL for the Hamiltonian/dynamical Matrix\n");
  if(!avoid_copying_hamiltonian)
  {
    delete result;
    result = new PetscMatrixParallelComplex (*temp_matrix);
    output_Propagator->allocated_momentum_Propagator_map[momentum_point]=true;
  }
  else
  {
    delete result;
    result = temp_matrix;
  }

  *result *= std::complex<double>(-1.0,0.0);//result=-H

  std::complex<double> energy=PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point,output_Propagator);
  shift_Hamiltonian_with_energy(energy,overlap,result);

  result->assemble();
  if(debug_output)
    result->save_to_matlab_file(get_name()+std::string("EmGHam.m"));

  //get all self-energies
  if(read_NEGF_object_list)
  //if(contact_sigmaR_solvers.size()==0)
  {
    std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
    for(; c_it!=Propagators.end(); ++c_it)
    {
      std::string temp_name = c_it->first;
      //omit those propagators that are in the writeable_Propagators map (!=inverse_Green)
      if(name_of_writeable_Propagator!=c_it->first)
      {
        if(Propagator_types.find(c_it->first)->second==NemoPhys::Fermion_retarded_self ||
            Propagator_types.find(c_it->first)->second==NemoPhys::Boson_retarded_self) // && add_self_energy_on_this_CPU)
        {
          //NemoUtils::tic(tic_toc_prefix + " getting Sigma");
          PetscMatrixParallelComplex* temp_matrix2=NULL;
          get_retarded_sigma(c_it->first,momentum_point,temp_matrix2,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
          //std::cerr<<prefix<<"using : "<<c_it->first<<"\n";
          //NemoUtils::toc(tic_toc_prefix + " getting Sigma");
          //add the matrix to result
          result->add_matrix(*temp_matrix2, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));//result=E-H-Sigma or result=ES-H-Sigma
        }
        else
        {
          /*ignore non ready and unknown Propagators that are not ret. self-energies for Fermions*/
        }
      }
    }
  }
  else
  {
    std::set<Simulation*>::iterator sigma_it=contact_sigmaR_solvers.begin();
    for(;sigma_it!=contact_sigmaR_solvers.end();++sigma_it)
    {
      std::string temp_name;
      NEMO_ASSERT((*sigma_it)!=NULL,prefix+"received NULL for contact self-energy solver\n");
      (*sigma_it)->get_data("writeable_Propagator",temp_name);
      PetscMatrixParallelComplex* temp_matrix2=NULL;
      //std::cerr<<prefix<<(*sigma_it)->get_name()<<" will give "<<temp_name<<"\n";
      (*sigma_it)->get_data(temp_name,&momentum_point,temp_matrix2,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      //add the matrix to result
      result->add_matrix(*temp_matrix2, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));//result=E-H-Sigma or result=ES-H-Sigma
    }
    if(scattering_sigmaR_solver!=NULL)
    {
      PetscMatrixParallelComplex* temp_matrix2=NULL;
      //std::cerr<<prefix<<scattering_sigmaR_solver->get_name()<<" will give scattering "<<temp_name<<"\n";
      SelfenergyInterface* selfenergy_interface = dynamic_cast<SelfenergyInterface*>(scattering_sigmaR_solver);
      const DOFmapInterface* dofmap = &Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain());
      selfenergy_interface->get_Selfenergy(momentum_point, temp_matrix2, dofmap, dofmap, NemoPhys::Fermion_retarded_self);
      //add the matrix to result
      result->add_matrix(*temp_matrix2, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));//result=E-H-Sigma or result=ES-H-Sigma
    }
    sigma_it=constant_eta_solvers.begin();
    for(;sigma_it!=constant_eta_solvers.end();++sigma_it)
    {
      std::string temp_name;
      NEMO_ASSERT((*sigma_it)!=NULL,prefix+"received NULL for constant eta solver\n");
      (*sigma_it)->get_data("writeable_Propagator",temp_name);
      PetscMatrixParallelComplex* temp_matrix2=NULL;
      //std::cerr<<prefix<<(*sigma_it)->get_name()<<" will give "<<temp_name<<"\n";
      (*sigma_it)->get_data(temp_name,&momentum_point,temp_matrix2,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      //add the matrix to result
      result->add_matrix(*temp_matrix2, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));//result=E-H-Sigma or result=ES-H-Sigma
    }
  }
  //assemble of the matrix
  result->assemble(); //Yu: before call assemble(), it won't work

  Propagator*& inverse_Green_function=output_Propagator; //remember - only one writeable_Propagator allowed for inverse_Green, therefore ".begin()"
  Propagator::PropagatorMap::iterator it;
  it=inverse_Green_function->propagator_map.find(momentum_point);
  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum_point,true);
  NemoUtils::toc(tic_toc_prefix);
}


void Greensolver::solve_retarded_Green_SanchoRubio(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::solve_retarded_Green_SanchoRubio ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Greensolver(\""+this->get_name()+"\")::solve_retarded_Green_SanchoRubio ";

  std::string variable_name=output_Propagator->get_name()+std::string("_lead_domain");
  std::string neighbor_domain_name;
  if (options.check_option(variable_name))
    neighbor_domain_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(prefix+" define \""+variable_name+"\"\n");
  const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);
  PropagationUtilities::Sancho_Rubio_retarded_green(this,output_Propagator,neighbor_domain,momentum_point,result);
  NemoUtils::toc(tic_toc_prefix);

}

void Greensolver::custom_matrix_product_RGF(const int number_of_off_diagonals, PetscMatrixParallelComplex*& retarded_Green, PetscMatrixParallelComplex*& Gamma,
    PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "Greensolver(\""+tic_toc_name+"\")::custom_matrix_product_RGF ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Greensolver(\""+this->get_name()+"\")::custom_matrix_product_RGF ";

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
      get_simulation_domain()->get_communicator() );
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

void Greensolver::set_input_options_map()
{
  Propagation::set_input_options_map();
}
