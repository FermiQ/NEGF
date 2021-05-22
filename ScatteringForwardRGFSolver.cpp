//$Id: ScatteringForwardRGFSolver.cpp $

#include "ScatteringForwardRGFSolver.h"
#include "EOMMatrixInterface.h"
#include "PetscMatrixParallelComplexContainer.h"
#include "QuantumNumberUtils.h"
#include "BlockSchroedingerModule.h"
#include "Propagator.h"
#include "PropagationUtilities.h"
#include "SelfenergyInterface.h"


ScatteringForwardRGFSolver::ScatteringForwardRGFSolver()
{
  repartition_solver=NULL;
  lead_domain=NULL;
  Hamiltonian=NULL;
  use_explicit_blocks=false;
  number_offdiagonal_blocks = 0;
  lesser_green_propagator=NULL;
  lesser_green_propagator_type=NemoPhys::Fermion_lesser_Green;
  L_container = NULL;
  check_inversion_stability = false;
}

ScatteringForwardRGFSolver::~ScatteringForwardRGFSolver()
{
  //this is responsible for the extra propagator
  if(lesser_green_propagator!=NULL)
  {
    delete_propagator_matrices(lesser_green_propagator);
    //delete propagator
    lesser_green_propagator->Propagator::~Propagator();
  }
  delete L_container;
  L_container = NULL;
}

void ScatteringForwardRGFSolver::do_init()
{
  read_NEGF_object_list=false;
  NEMO_ASSERT(options.check_option("external_Hamilton_Constructor"),"ScatteringForwardRGFSolver(\""+get_name()+"\")::do_init please define \"external_Hamilton_Constructor\"\n");
  std::string temp_name=options.get_option("external_Hamilton_Constructor",std::string(""));
  options.set_option("Hamilton_constructor",temp_name);
  Greensolver::do_init();
  //additionally we have g< and HG propagators that must be calculated correctly
  options.get_option("Hamiltonian_indices",Hamiltonian_indices);

  Simulation* temp_simulation=Hamilton_Constructor;
  if(options.check_option("Repartition_solver"))
  {
    std::string name=options.get_option("Repartition_solver",std::string(""));
    Simulation* temp_simulation=find_simulation(name);
    NEMO_ASSERT(temp_simulation!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::do_init have not found simulation \""+name+"\"\n");
  }
  repartition_solver=dynamic_cast<RepartitionInterface*>(temp_simulation);
  NEMO_ASSERT(repartition_solver!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::do_init \""+temp_simulation->get_name()+"\" is not a RepartitionInterface\n");

  if(options.check_option("number_of_offdiagonal_blocks"))
  {
    use_explicit_blocks = true;
    number_offdiagonal_blocks = options.get_option("number_of_offdiagonal_blocks",0);
  }

  if(options.check_option("scattering_self_energies"))
  {
    std::vector<std::string> scattering_self_energies;
    options.get_option("scattering_self_energies",scattering_self_energies);
    for(unsigned int i=0; i<scattering_self_energies.size(); i++)
    {
      NEMO_ASSERT(options.check_option(scattering_self_energies[i]+"_solver"),get_name()+"have not found \""+scattering_self_energies[i]+"+_solver\"\n");
      std::string solver_name=options.get_option(scattering_self_energies[i]+"_solver",std::string(""));
      //opt.set_option(scattering_self_energies[i]+"_solver",solver_name);
      if(options.check_option(scattering_self_energies[i]+"_constructor"))
      {
        std::string constructor_name=options.get_option(scattering_self_energies[i]+"_constructor",std::string(""));
        //opt.set_option(scattering_self_energies[i]+"_constructor",constructor_name);
        if(scattering_self_energies[i].find("retarded")!=std::string::npos)
          scattering_sigmaR_solver=find_simulation(constructor_name);
        if(scattering_self_energies[i].find("lesser")!=std::string::npos)
          scattering_sigmaL_solver=find_simulation(constructor_name);
      }
    }
  }
  check_inversion_stability = options.get_option("check_inversion_stability",false);

}

void ScatteringForwardRGFSolver::do_solve()
{
  initialize_Propagators();
  Propagator::PropagatorMap& temp_map = writeable_Propagator->propagator_map;
  Propagator::PropagatorMap::iterator momentum_it=temp_map.begin();
  PetscMatrixParallelComplex* Matrix = NULL;
  //get_Greensfunction will trigger solving all relevant Propagations
  for(; momentum_it!=temp_map.end(); ++momentum_it)
  {
    get_Greensfunction(momentum_it->first,Matrix,NULL,NULL,NemoPhys::Fermion_retarded_Green);
  }
}

void ScatteringForwardRGFSolver::get_Greensfunction(const std::vector<NemoMeshPoint>& momentum,PetscMatrixParallelComplex*& result,
    const DOFmapInterface* row_dofmap, const DOFmapInterface* column_dofmap,const NemoPhys::Propagator_type& type)
{
  bool data_is_ready=false;
  if(type==NemoPhys::Fermion_retarded_Green||type==NemoPhys::Boson_retarded_Green)
  {
    std::map<std::vector<NemoMeshPoint>, bool>::const_iterator momentum_it=job_done_momentum_map.find(momentum);
    if(momentum_it!=job_done_momentum_map.end())
      data_is_ready=momentum_it->second;
    else
      job_done_momentum_map[momentum]=false;
    if(!data_is_ready)
      run_forward_RGF_for_momentum(momentum);

    Propagator::PropagatorMap& result_prop_map=writeable_Propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it=result_prop_map.find(momentum);

    //extract the requested submatrix
    if(row_dofmap!=NULL||column_dofmap!=NULL)
    {
      std::vector<int> row_indices;
      if(row_dofmap!=NULL)
        Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()), *row_dofmap, row_indices);
      std::vector<int> column_indices=row_indices;
      if(column_dofmap!=NULL)
        Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()), *column_dofmap, column_indices);
      if(row_dofmap==NULL)
        row_indices=column_indices;
      PetscMatrixParallelComplex* temp_matrix_container=prop_it->second;
      NEMO_ASSERT(temp_matrix_container!=NULL,"ScatteringForwardRGFSolver(\""+get_name()+"\")::get_Greensfunction gR is still NULL\n");
      delete result;
      result=NULL;
      temp_matrix_container->get_submatrix(row_indices,column_indices,MAT_REUSE_MATRIX,result);
    }
    else
      result=prop_it->second;
  }
  else if (type==NemoPhys::Fermion_lesser_Green||type==NemoPhys::Boson_lesser_Green)
  {
    //throw std::runtime_error("ScatteringForwardRGFSolver(\""+get_name()+"\")::get_Greensfunction: lesser Green NYI\n");
    std::map<std::vector<NemoMeshPoint>, bool>::const_iterator momentum_it=lesser_propagator_job_done_momentum_map.find(momentum);
    if(momentum_it!=lesser_propagator_job_done_momentum_map.end())
      data_is_ready=momentum_it->second;
    else
      lesser_propagator_job_done_momentum_map[momentum]=false;
    if(!data_is_ready)
      run_forward_RGF_for_momentum(momentum);

    Propagator::PropagatorMap& result_prop_map=lesser_green_propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it=result_prop_map.find(momentum);

    //extract the requested submatrix
    if(row_dofmap!=NULL||column_dofmap!=NULL)
    {
      std::vector<int> row_indices;
      if(row_dofmap!=NULL)
        Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()), *row_dofmap, row_indices);
      std::vector<int> column_indices=row_indices;
      if(column_dofmap!=NULL)
        Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()), *column_dofmap, column_indices);
      if(row_dofmap==NULL)
        row_indices=column_indices;
      PetscMatrixParallelComplex* temp_matrix_container=prop_it->second;
      NEMO_ASSERT(temp_matrix_container!=NULL,"ScatteringForwardRGFSolver(\""+get_name()+"\")::get_Greensfunction gL is still NULL\n");
      delete result;
      result=NULL;
      temp_matrix_container->get_submatrix(row_indices,column_indices,MAT_REUSE_MATRIX,result);
    }
    else
      result=prop_it->second;

  }
  else if (type == NemoPhys::Fermion_lesser_HGreen||type==NemoPhys::Boson_lesser_HGreen)
  {
    throw std::runtime_error("ScatteringForwardRGFSolver(\""+get_name()+"\")::get_Greensfunction: lesser HGreen NYI\n");

  }
  else
  {
    throw std::runtime_error("ScatteringForwardRGFSolver(\""+get_name()+"\")::get_Greensfunction: called with unknown Propagator type\n");
  }

}

void ScatteringForwardRGFSolver::get_data(const std::string& variable, std::map<std::vector<NemoMeshPoint>, int >& momentum_map)
{
  if(variable == "Ek_stable_inversion_map")
  {
    momentum_map = Ek_stable_inversion;
  }

}


void ScatteringForwardRGFSolver::get_data(const std::string& variable, const std::vector<NemoMeshPoint>* momentum,
    PetscMatrixParallelComplex*& Matrix, const DOFmapInterface* row_dof_map, const DOFmapInterface* col_dofmap)
{

  if(variable == "L_container")
    Matrix = L_container;
  else
  {
    NemoPhys::Propagator_type type;
    if (variable == writeable_Propagator->get_name())
      type = writeable_Propagator->get_propagator_type();
    else if (variable == lesser_green_propagator_name)
      type = lesser_green_propagator_type;
    else
      throw std::runtime_error("ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_data variable, momentum, matrix, "
          "row dof, col dof: called with unknown Propagator type\n");

    get_Greensfunction(*momentum, Matrix, row_dof_map, col_dofmap, type);
  }
}

void ScatteringForwardRGFSolver::run_forward_RGF_for_momentum(const std::vector<NemoMeshPoint>& momentum)
{
  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this->get_name() + "\")::run_forward_RGF_for_momentum: ";
  NemoUtils::tic(tic_toc_prefix);

  initialize_Propagators();
  //0. get some general information
  //0.1 get the list of subdomain names
  if(subdomain_names.size()<1)
    repartition_solver->get_subdomain_names(get_const_simulation_domain(),true,subdomain_names);

  //1. get the Hamiltonian block matrices
  EOMMatrixInterface* eom_interface=NULL;
  get_device_Hamiltonian(momentum,Hamiltonian, eom_interface);
  if(debug_output)
    Hamiltonian->save_to_matlab_file("total_H.m");

  //2.1 get the contact self-energy
  PetscMatrixParallelComplex* gR = NULL;
  GreensfunctionInterface* temp_gR_interface=dynamic_cast<GreensfunctionInterface*>(forward_gR_solver);
  NEMO_ASSERT(temp_gR_interface!=NULL,"ScatteringForwardRGFSolver(\""+get_name()+"\")::get_contact_sigma \""+forward_gR_solver->get_name()+"\" is no GreensfunctionInterface\n");
  if(particle_type_is_Fermion)
    temp_gR_interface->get_Greensfunction(momentum,gR,NULL, NULL, NemoPhys::Fermion_retarded_Green);
  else
    temp_gR_interface->get_Greensfunction(momentum,gR,NULL, NULL, NemoPhys::Boson_retarded_Green);

  //this doesn't change for nonlocal scattering
  PetscMatrixParallelComplex* Sigma_contact=NULL;
  get_contact_sigma(momentum,eom_interface, Sigma_contact, subdomain_names[0], lead_domain, Hamiltonian, gR);
  //NemoMath::symmetry_type type_symm = NemoMath::symmetric;
  //PropagationUtilities::symmetrize(this, Sigma_contact,type_symm);

  PetscMatrixParallelComplexContainer* Hamiltonian_container=dynamic_cast<PetscMatrixParallelComplexContainer*>(Hamiltonian);
  const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& H_blocks=Hamiltonian_container->get_const_container_blocks();
  std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator block_cit=H_blocks.begin();
  std::vector<std::pair<std::vector<int>,std::vector<int> > > diagonal_indices;
  std::vector<std::pair<std::vector<int>,std::vector<int> > > offdiagonal_indices;
  EOMMatrixInterface* temp_module=dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
  NEMO_ASSERT(temp_module!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::run_forward_RGF_for_momentum Hamilton_Constructor is not a EOMMatrixInterface\n");
  temp_module->get_ordered_Block_keys(diagonal_indices, offdiagonal_indices);

  if(use_explicit_blocks)
  {
    //L_container allocation
    if(L_container==NULL)
    {
      L_container=new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),Hamiltonian->get_num_cols(),Hamiltonian->get_communicator());
    }
  }

  //2.2 get the lesser contact_self_energy
  PetscMatrixParallelComplex* Sigma_lesser_contact=NULL;
  get_contact_sigma_lesser_equilibrium(momentum,eom_interface,Sigma_lesser_contact, subdomain_names[0],
      lead_domain, Hamiltonian_indices, Hamiltonian, gR);


  //3. loop over the blocks of 1. and solve forward RGF

  for(unsigned int i=0;i<diagonal_indices.size()-1;i++)
  {
    msg.threshold(2) << get_name() << " solving domain " << i << " \n";
    //cerr << " scatteringrgf solving for domain " << i << " \n";

    //solve diagonal gR
    block_cit=H_blocks.find(diagonal_indices[i]);
    NEMO_ASSERT(block_cit!=H_blocks.end(),"ForwardRGFSolver(\""+get_name()+"\")::run_forward_RGF_for_momentum have not found block Hamiltonian\n");
    //PetscMatrixParallelComplex* diagonal_H=new PetscMatrixParallelComplex(*(block_cit->second));
    PetscMatrixParallelComplex* diagonal_H=block_cit->second;//new PetscMatrixParallelComplex(*(block_cit->second));

    PetscMatrixParallelComplex* scattering_sigmaR_matrix = NULL;
    if(scattering_sigmaR_solver!=NULL)
    {
      const DOFmapInterface& dof_map = Hamilton_Constructor->get_dof_map(Domain::get_domain(subdomain_names[i]));
      SelfenergyInterface* selfenergy_interface = dynamic_cast<SelfenergyInterface*>(scattering_sigmaR_solver);
      NEMO_ASSERT(selfenergy_interface, tic_toc_prefix + "scattering sigmaR solver cannot be cast into class type SelfenergyInterface");
      NemoPhys::Propagator_type propagator_type;
      if (particle_type_is_Fermion)
        propagator_type = NemoPhys::Fermion_retarded_self;
      else
        propagator_type = NemoPhys::Boson_retarded_self;
      selfenergy_interface->get_Selfenergy(momentum, scattering_sigmaR_matrix, &dof_map, &dof_map, propagator_type);
    }
    //solve_diagonal_gR will delete diagonal_H and Sigma_contact
    PetscMatrixParallelComplex* block_gR=NULL;
    solve_diagonal_gR(diagonal_indices[i], momentum, diagonal_H,Sigma_contact,block_gR,scattering_sigmaR_matrix);
    //block_gR->save_to_matlab_file("forward_gR_"+subdomain_names[i]+".m");
    //off diagonal gR
    if(use_explicit_blocks && i>0)
    {
      int row_index = i;

      unsigned int actual_requested = number_offdiagonal_blocks;
      //check bounds
      int difference = row_index;//+1;//subdomain_names.size()-(row_index);
      actual_requested = std::min(int(number_offdiagonal_blocks),difference);

      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
          L_container->get_const_container_blocks();
      Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      PetscMatrixParallelComplexContainer* gR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks =
          gR_container->get_const_container_blocks();

      //loop through j which is nonlocality index
      for(unsigned int j = 1; j <= actual_requested; ++j)
        //for (unsigned int j = actual_requested; j >=1; --j)
      {
        int start_index = row_index-j+1;//row_index-j+1;//i-j+1;
        PetscMatrixParallelComplex* result_gr = NULL;
        //PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
        //NEMO_ASSERT(GR_container != NULL,
        //    "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum GR is not received in the container format\n");



        PropagationUtilities::solve_offdiagonal_gR_nonlocal_blocks(this, momentum, diagonal_indices, number_offdiagonal_blocks-1, row_index-j, row_index, start_index,
            subdomain_names, L_blocks, gR_blocks, gR_container, result_gr);
        delete result_gr;
        result_gr = NULL;
      }


      //off diagonal g<
      //solve off diagonal gL
      Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);

      PetscMatrixParallelComplexContainer* gL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);

      int start_index = i-number_offdiagonal_blocks;
      if(start_index < 0)
        start_index = 0;
      int end_index = i-1;
      std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[i].first,diagonal_indices[i].second);
         std::map< std::pair < std::vector<int>, std::vector<int> >,
         PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
         NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
             tic_toc_prefix + " have not found gr diagonal block \n");
         PetscMatrixParallelComplex* gR_row_row = gRblock_cit->second;

      for(int j = start_index; j <= end_index; ++j)
      {

        PetscMatrixParallelComplex* result_gL = NULL;
        PropagationUtilities::solve_offdiagonal_gL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
            number_offdiagonal_blocks, i, j, start_index, subdomain_names, L_blocks, gR_blocks, gR_row_row,
            scattering_sigmaL_solver, gL_container, result_gL);
        delete result_gL;
        result_gL = NULL;
      }
    }


    //solve diagonal g<
    PetscMatrixParallelComplex* scattering_sigmaL_matrix = NULL;
    if(scattering_sigmaL_solver!=NULL)
    {
      PetscMatrixParallelComplex* temp_matrix = NULL;
      //PetscMatrixParallelComplex* temp_matrix2=NULL;
      //std::cerr<<prefix<<scattering_sigmaR_solver->get_name()<<" will give scattering "<<temp_name<<"\n";
      //const DOFmapInterface& dof_map=Hamilton_Constructor->get_dof_map(Domain::get_domain(subdomain_names[i]));
      SelfenergyInterface* selfenergy_interface = dynamic_cast<SelfenergyInterface*>(scattering_sigmaL_solver);
      NEMO_ASSERT(selfenergy_interface, tic_toc_prefix + "scattering sigmaL solver cannot be cast into class type SelfenergyInterface");
      const DOFmapInterface& dof_map = Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain());
      NemoPhys::Propagator_type propagator_type;
      if (particle_type_is_Fermion)
        propagator_type = NemoPhys::Fermion_lesser_self;
      else
        propagator_type = NemoPhys::Boson_lesser_self;
      selfenergy_interface->get_Selfenergy(momentum, temp_matrix, &dof_map, &dof_map, propagator_type);
      //PetscMatrixParallelComplexContainer* sigmaL_container=dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
      //NEMO_ASSERT(sigmaL_container!=NULL,"ScatteringForwardRGFSolver(\""+get_name()+"\")::run_forward_RGF sigmaL is not a container \n");
      //const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& sigmaL_blocks=sigmaL_container->get_const_container_blocks();
      //std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator sigmaL_block_cit=
      //    sigmaL_blocks.find(diagonal_indices[i]);
      scattering_sigmaL_matrix = new PetscMatrixParallelComplex(diagonal_indices[i].first.size(),diagonal_indices[i].second.size(),
          temp_matrix->get_communicator());

      temp_matrix->get_submatrix(diagonal_indices[i].first,diagonal_indices[i].second,MAT_INITIAL_MATRIX,scattering_sigmaL_matrix);
      scattering_sigmaL_matrix->assemble();
      //Domain* name_domain = Domain::get_domain(subdomain_names[i]);
      //scattering_sigmaL_matrix->save_to_matlab_file("sigmaL_"+name_domain->get_name()+".m");
      //scattering_sigmaL_matrix=sigmaL_block_cit->second;
    }
    PetscMatrixParallelComplex* block_gL = NULL;
    if(!use_explicit_blocks || i==0)
    {
      bool real_lead = false;
      if(i==0)
        real_lead = true;
      solve_diagonal_gL(diagonal_indices[i], momentum, Sigma_lesser_contact, block_gR, block_gL, real_lead, scattering_sigmaL_matrix);

      //block_gL->save_to_matlab_file("forward_gL_"+subdomain_names[i]+".m");

      if(debug_output)
      {
        Domain* temp_domain = Domain::get_domain(subdomain_names[i]);
        block_gL->save_to_matlab_file("gL_"+temp_domain->get_name()+".m");
      }
    }
    else
    {
      //TOOD:JC lesser for testing

      int start_index = i-number_offdiagonal_blocks;
      if(start_index < 0)
        start_index = 0;

      Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);
      /*if(prop_it_lesser->second==NULL)
      {
        PetscMatrixParallelComplexContainer* new_container=new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),Hamiltonian->get_num_cols(),Hamiltonian->get_communicator());
        lesser_green_propagator->allocated_momentum_Propagator_map[momentum]=true;
        prop_it_lesser->second=new_container;
      }*/
      PetscMatrixParallelComplexContainer* gL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
          L_container->get_const_container_blocks();
      Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      PetscMatrixParallelComplexContainer* gR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks =
          gR_container->get_const_container_blocks();

      std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[i].first,diagonal_indices[i].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
      NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
          tic_toc_prefix + " have not found gr diagonal block \n");
      PetscMatrixParallelComplex* gR_row_row = gRblock_cit->second;

      PropagationUtilities::solve_diagonal_gL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
          number_offdiagonal_blocks, i, start_index, subdomain_names, L_blocks, gR_blocks, Sigma_lesser_contact, gR_row_row,
          scattering_sigmaL_solver, gL_container, block_gL);

    }
    delete scattering_sigmaL_matrix;
    scattering_sigmaL_matrix = NULL;
    //solve the next contact self-energy
    if(i<diagonal_indices.size()-1)
    {
      //get the right coupling Hamiltonian
      std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator off_block_cit=H_blocks.find(offdiagonal_indices[i]);
      NEMO_ASSERT(off_block_cit!=H_blocks.end(),"ForwardRGFSolver(\""+get_name()+"\")::run_forward_RGF_for_momentum have not found offdiagonal block Hamiltonian\n");
      PetscMatrixParallelComplex* coupling_Hamiltonian=off_block_cit->second;

      if(!use_explicit_blocks)
      {
        get_contact_sigma(momentum,eom_interface, Sigma_contact, subdomain_names[0], lead_domain, Hamiltonian, block_gR,coupling_Hamiltonian);
        //NemoMath::symmetry_type type_symm = NemoMath::symmetric;
        //PropagationUtilities::symmetrize(this,Sigma_contact,type_symm);
        get_contact_sigma(momentum,eom_interface, Sigma_lesser_contact, subdomain_names[0], lead_domain, Hamiltonian,
            block_gL,coupling_Hamiltonian, NULL ,lesser_green_propagator_type);
        NemoMath::symmetry_type type = NemoMath::antihermitian;
        symmetrize(Sigma_lesser_contact,type);

      }
      else
      {
        //get_contact_sigma(momentum,eom_interface, Sigma_contact, subdomain_names[0], lead_domain, Hamiltonian_indices, Hamiltonian, block_gR,coupling_Hamiltonian);
        //       get_contact_sigma(momentum,eom_interface, Sigma_lesser_contact, subdomain_names[0], lead_domain, Hamiltonian_indices, Hamiltonian,
        //           block_gR,coupling_Hamiltonian, sigma_offdiagonal,lesser_green_propagator_type);
        //throw std::runtime_error("ScatteringForwardRGFSolver(\""+get_name()+"\")::run_forward_RGF_for_momentum explicit blocks NYI \n");

        //get gR blocks
         Propagator::PropagatorMap& result_prop_map=writeable_Propagator->propagator_map;
         Propagator::PropagatorMap::iterator prop_it=result_prop_map.find(momentum);
         NEMO_ASSERT(prop_it!=result_prop_map.end(),tic_toc_prefix + " could not find gR container");
         PetscMatrixParallelComplexContainer* gR_container=dynamic_cast<PetscMatrixParallelComplexContainer*> (prop_it->second);


        PropagationUtilities::get_contact_sigmaR_nonlocal(this, momentum, eom_interface, i+1, number_offdiagonal_blocks-1, subdomain_names,
        diagonal_indices, L_container, gR_container, Sigma_contact, scattering_sigmaR_solver);

        NemoMath::symmetry_type type = NemoMath::symmetric;
        PropagationUtilities::symmetrize(this, Sigma_contact,type);

        //this isn't needed for nonlocal scattering. Only the actual contacts the rest of this sigma are included indirectly
        //TODO:JC lesser for testing
        /*get_contact_sigma(momentum,eom_interface, Sigma_lesser_contact, subdomain_names[0], lead_domain, Hamiltonian,
            block_gL,coupling_Hamiltonian, NULL ,lesser_green_propagator_type);
        NemoMath::symmetry_type type = NemoMath::antihermitian;
        symmetrize(Sigma_lesser_contact,type);*/

        delete Sigma_lesser_contact;
        Sigma_lesser_contact=NULL;
      }
    }
    if(debug_output)
    {
     Domain* temp_domain = Domain::get_domain(subdomain_names[i]);
     block_gR->save_to_matlab_file("gR_"+temp_domain->get_name()+".m");
    }
    delete block_gR;
    block_gR = NULL;
    delete block_gL;
    block_gL = NULL;

    //
    Hamiltonian_container->delete_submatrix(diagonal_indices[i].first,diagonal_indices[i].second);
    Hamiltonian_container->delete_submatrix(offdiagonal_indices[i].first,offdiagonal_indices[i].second);
  }

  /*
  if(use_explicit_blocks)
  {
    //off diagonal g<
    //solve off diagonal gL
    Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);

    PetscMatrixParallelComplexContainer* gL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);

    Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

    PetscMatrixParallelComplexContainer* gR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks =
        gR_container->get_const_container_blocks();
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
        L_container->get_const_container_blocks();

    int i = subdomain_names.size()-1;
    std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[i].first,diagonal_indices[i].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >,
    PetscMatrixParallelComplex* >::const_iterator gRblock_cit=gR_blocks.find(temp_pair2);
    NEMO_ASSERT(gRblock_cit != gR_blocks.end(),
        tic_toc_prefix + " have not found gr diagonal block \n");
    PetscMatrixParallelComplex* gR_row_row = gRblock_cit->second;


    int start_index = i-number_offdiagonal_blocks;
    if(start_index < 0)
      start_index = 0;
    int end_index = i-1;

    for(int j = start_index; j <= end_index; ++j)
    {

      PetscMatrixParallelComplex* result_gL = NULL;
      PropagationUtilities::solve_offdiagonal_gL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
          number_offdiagonal_blocks, i, j, start_index, subdomain_names, L_blocks, gR_blocks, gR_row_row, scattering_sigmaL_solver, gL_container,
          result_gL);
      delete result_gL;
      result_gL = NULL;
    }
    int start_index_diag = i-number_offdiagonal_blocks;
    if(start_index_diag < 0)
      start_index_diag = 0;


    PetscMatrixParallelComplex* result_gL = NULL;
    PropagationUtilities::solve_diagonal_gL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
        number_offdiagonal_blocks, i, start_index_diag, subdomain_names, L_blocks, gR_blocks, Sigma_lesser_contact, gR_row_row,
        scattering_sigmaL_solver, gL_container, result_gL);
    delete result_gL;
    result_gL = NULL;

  }
  */

  set_job_done_momentum_map(&name_of_writeable_Propagator,&momentum,true);
  set_job_done_momentum_map(&lesser_green_propagator_name,&momentum,true);
  delete Hamiltonian;
  Hamiltonian=NULL;
  delete Sigma_contact;
  Sigma_contact=NULL;  
  delete Sigma_lesser_contact;
  Sigma_lesser_contact=NULL;

  NemoUtils::toc(tic_toc_prefix);
}

void ScatteringForwardRGFSolver::solve_diagonal_gR(std::pair<std::vector<int>,std::vector<int> >& diagonal_index, const std::vector<NemoMeshPoint>& momentum,
    PetscMatrixParallelComplex*& diagonal_H,PetscMatrixParallelComplex*& Sigma_contact,
    PetscMatrixParallelComplex*& block_gR, PetscMatrixParallelComplex* scattering_sigmaR_matrix)
{
  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_gR: ";
  NemoUtils::tic(tic_toc_prefix);

  //sigma is dense
  diagonal_H->matrix_convert_dense();
  Sigma_contact->matrix_convert_dense();
  //NemoMath::symmetry_type type = NemoMath::symmetric;
  //PropagationUtilities::symmetrize(this, Sigma_contact,type);
  Sigma_contact->add_matrix(*diagonal_H,DIFFERENT_NONZERO_PATTERN,std::complex<double> (1.0,0.0));//result=E-H-Sigma or result=E^2-H-Sigma


  if(scattering_sigmaR_matrix!=NULL)
  {
    scattering_sigmaR_matrix->matrix_convert_dense();
    Sigma_contact->add_matrix(*scattering_sigmaR_matrix, DIFFERENT_NONZERO_PATTERN,std::complex<double> (1.0,0.0));//result=E-H-Sigma or result=ES-H-Sigma
  }

  //sigma contact = H+sigma
  *Sigma_contact *= std::complex<double>(-1.0,0.0);
  //sigma_contact = -H-sigma
  std::complex<double> energy=std::complex<double>(PropagationUtilities::read_energy_from_momentum(this,momentum, writeable_Propagator),0.0);
  PetscMatrixParallelComplex* temp_S_matrix=NULL;
  shift_Hamiltonian_with_energy(energy,temp_S_matrix,Sigma_contact); //diagonal_H=E-H


  options.set_option("invert_solver",options.get_option("invert_solver",std::string("lapack")));
  PropagationUtilities::exact_inversion(this, momentum, Sigma_contact, block_gR);

  if(check_inversion_stability)
  {

    PetscMatrixParallelComplex* isitidentity = NULL;
    PetscMatrixParallelComplex::mult(*Sigma_contact,*block_gR,&isitidentity);
    Sigma_contact->assemble();
    //inverse_Green->save_to_matlab_file("inverse_Green.m");
    block_gR->assemble();
    //result->save_to_matlab_file("result.m");
    isitidentity->assemble();
    //isitidentity->save_to_matlab_file("isitidentity.m");
    {
      PetscMatrixParallelComplex identity =PetscMatrixParallelComplex(isitidentity->get_num_rows(),
          isitidentity->get_num_cols(),isitidentity->get_communicator());
      identity.consider_as_full();
      identity.allocate_memory();
      identity.set_to_zero();
      identity.assemble();
      //temp_transpose.save_to_matlab_file("exact_identity.m");
      identity.matrix_diagonal_shift(cplx(1.0,0.0));//idenity
      identity.assemble();
      //isitidentity->transpose_matrix(temp_transpose,MAT_INITIAL_MATRIX);
      identity.add_matrix(*isitidentity, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));
      identity.assemble();

      std::vector<double> norms;
      identity.get_column_euclidean_norms(&norms);
      double norm_max = *std::max_element(norms.begin(), norms.end());
      double tolerance = options.get_option("inversion_stability_tolerance",double(1E-7));
      if(norm_max > tolerance)
      {
        //cerr << "energy " << energy << " norm compared to identity " <<  norm_max <<  "\n";
        //add to map of offending energies
        Ek_stable_inversion[momentum] = 0;
      }
    }
    delete isitidentity;
  }

  //4. store results in PMPCC
  //if needed create a PetscMatrixParallelComplexContainer

  if(use_explicit_blocks)
  {
    NemoMath::symmetry_type type = NemoMath::symmetric;
    PropagationUtilities::symmetrize(this, block_gR,type);
  }

  Propagator::PropagatorMap& result_prop_map=writeable_Propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it=result_prop_map.find(momentum);
  if(prop_it==result_prop_map.end())
  {
    result_prop_map[momentum]=NULL;
    prop_it=result_prop_map.find(momentum);
  }
  if(prop_it->second==NULL)
  {
    PetscMatrixParallelComplexContainer* new_container=new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),Hamiltonian->get_num_cols(),Hamiltonian->get_communicator());
    writeable_Propagator->allocated_momentum_Propagator_map[momentum]=true;
    prop_it->second=new_container;
  }
  PetscMatrixParallelComplexContainer* temp_container=dynamic_cast<PetscMatrixParallelComplexContainer*> (prop_it->second);
  temp_container->set_block_from_matrix1(*block_gR,diagonal_index.first,diagonal_index.second);
  NemoUtils::toc(tic_toc_prefix);

}


//void ScatteringForwardRGFSolver::get_contact_sigmaR_nonlocal(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum , EOMMatrixInterface* eom_interface, unsigned int i,
//        unsigned int number_of_offdiagonal, std::vector<string> subdomain_names,
//        std::vector<std::pair<std::vector<int>,std::vector<int> > > diagonal_indices,
//        std::vector<std::pair<std::vector<int>,std::vector<int> > > offdiagonal_indices,
//        PetscMatrixParallelComplexContainer*& L_block_container, PetscMatrixParallelComplexContainer* gR_container,
//        PetscMatrixParallelComplex*& Sigma_contact, Simulation* scattering_sigmaR_solver)
//{
//  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this_simulation->get_name() + "\")::get_contact_sigmaR_nonlocal: ";
//  delete Sigma_contact;
//  Sigma_contact = NULL;
//
//  const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& gR_blocks=gR_container->get_const_container_blocks();
//
//  //result wil be stored in container for later use
//  int row_idx = i;
//  int start_idx = i - 1 - number_of_offdiagonal;
//  if(start_idx<0)
//    start_idx = 0;
//  int end_idx = i - 1;
//
//  for(int k = start_idx; k <= end_idx; ++k)
//  {
//    PetscMatrixParallelComplex* sigmaR = NULL;
//    if(scattering_sigmaR_solver != NULL)
//    {
//      //find sigmaR( row_idx,k )
//      std::string row_name = subdomain_names[row_idx];
//      std::string col_name = subdomain_names[k];
//     extract_sigmaR_submatrix(momentum, row_name, col_name, eom_interface, sigmaR);
//    }
//
//    PetscMatrixParallelComplex* L_row_k = NULL;
//    solve_L(this_simulation, momentum, eom_interface, number_of_offdiagonal, subdomain_names, row_idx, k/*col_idx*/,
//        diagonal_indices, offdiagonal_indices, sigmaR, L_row_k, L_container, gR_container, scattering_sigmaR_solver);
//
//    //get gR
//    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator gR_cit = gR_blocks.find(diagonal_indices[k]);
//    PetscMatrixParallelComplex* gR_k_k = gR_cit->second;
//
//    PetscMatrixParallelComplex* temp_matrix = NULL;
//    PetscMatrixParallelComplex::mult(*L_row_k, *gR_k_k, &temp_matrix);
//
//    //transpose L_row_k
//    PetscMatrixParallelComplex L_k_row(L_row_k->get_num_cols(),L_row_k->get_num_rows(),L_row_k->get_communicator());
//    L_row_k->transpose_matrix(L_k_row,MAT_INITIAL_MATRIX);
//    delete L_row_k;
//    L_row_k = NULL;
//    PetscMatrixParallelComplex* temp_matrix2 = NULL;
//    PetscMatrixParallelComplex::mult(*temp_matrix,L_k_row,&temp_matrix2);
//
//    if(Sigma_contact == NULL)
//      Sigma_contact = new PetscMatrixParallelComplex(*temp_matrix2);
//    else
//      Sigma_contact->add_matrix(*temp_matrix2, DIFFERENT_NONZERO_PATTERN, cplx(1.0,0.0));
//    delete temp_matrix;
//    temp_matrix = NULL;
//    delete temp_matrix2;
//    temp_matrix2 = NULL;
//  }
//
//}
//
//void ScatteringForwardRGFSolver::solve_L(Simulation* this_simulation, const std::vector<NemoMeshPoint>& momentum, EOMMatrixInterface* eom_interface,
//    unsigned int number_of_offdiagonal, std::vector<string> subdomain_names,
//    unsigned int row_idx, unsigned int col_idx, std::vector<std::pair<std::vector<int>,std::vector<int> > > diagonal_indices,
//    std::vector<std::pair<std::vector<int>,std::vector<int> > > offdiagonal_indices,
//    PetscMatrixParallelComplex* sigmaR, PetscMatrixParallelComplex*& L_block, PetscMatrixParallelComplexContainer*& L_blocks_container,
//    PetscMatrixParallelComplexContainer*& gR_container, Simulation* scattering_sigmaR_solver)
//{
//  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_gL: ";
//
//  cerr << " row_idx " << row_idx << " col_idx " << col_idx << " \n";
//
//  //get gR blocks
//  //Propagator::PropagatorMap& result_prop_map=writeable_Propagator->propagator_map;
//  //Propagator::PropagatorMap::iterator prop_it=result_prop_map.find(momentum);
//  //NEMO_ASSERT(prop_it!=result_prop_map.end(),tic_toc_prefix + " could not find gR container");
//  //PetscMatrixParallelComplexContainer* gR_container=dynamic_cast<PetscMatrixParallelComplexContainer*> (prop_it->second);
//  const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& gR_blocks=gR_container->get_const_container_blocks();
//
//  //get coupling H
//  //including sigmaR
//  //row is coupling domain
//  Domain* lead_subdomain = Domain::get_domain(subdomain_names[row_idx]);
//  //column is target domain
//  Domain* target_subdomain = Domain::get_domain(subdomain_names[col_idx]);
//  int num_target_rows = diagonal_indices[col_idx].first.size();
//  int num_target_cols = num_target_rows;
//
//  std::vector<NemoMeshPoint> temp_vector;
//  QuantumNumberUtils::sort_quantum_number(momentum,temp_vector,options,get_momentum_mesh_types(),this);
//
//  //find all leads
//  const vector< const Domain*>& leads = target_subdomain->get_all_leads();
//  int num_leads = leads.size();
//
//  for(unsigned int idx = 0; idx < num_leads; ++idx)
//  {
//    if (leads[idx] == lead_subdomain)
//    {
//    PetscMatrixParallelComplex* full_couplingH = NULL;
//    eom_interface->get_EOMMatrix(temp_vector, target_subdomain, lead_subdomain,true,full_couplingH);
//
//    //PetscMatrixParallelComplex* coupling = NULL;
//    PropagationUtilities::extract_coupling_Hamiltonian_RGF(this, num_target_rows, num_target_cols,  full_couplingH,
//        L_block, sigmaR);
//    }
//  }
//
//  if(L_block==NULL && sigmaR != NULL)
//    L_block = new PetscMatrixParallelComplex(*sigmaR);
//
//
//  if(L_block==NULL &&sigmaR==NULL )
//  {
//    //for testing make L_block matrix that is diagonal 0
//    //PetscMatrixParallelComplex fake_0_matrix(diagonal_indices[0].first.size(), diagonal_indices[0].second.size(),
//    //    get_const_simulation_domain()->get_communicator());
//    L_block = new PetscMatrixParallelComplex(diagonal_indices[row_idx].first.size(), diagonal_indices[col_idx].first.size(), get_const_simulation_domain()->get_communicator());
//    int start_row = 0;
//    int end_row = diagonal_indices[row_idx].first.size(); //fix this
//    L_block->set_num_owned_rows(end_row - start_row);
//    for (int i = start_row; i < end_row; i++)
//      L_block->set_num_nonzeros_for_local_row(i, 1, 0);
//    L_block->allocate_memory();
//    L_block->set_to_zero();
//    L_block->assemble();
//  }
//
//
//
//  int start_idx = row_idx/*col_idx*/ - 1 - number_of_offdiagonal;
//  //check bounds
//  if(start_idx < 0)
//    start_idx = 0;
//  int end_idx = col_idx - 1;//1;
//  //check bounds
//  //if(end_idx < 0)
//  //  end_idx = 0;
//
//  for(int k = start_idx; k <= end_idx && k>=0; k++)
//  {
//    //find L(row_idx,k)
//    const std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >& L_blocks=L_container->get_const_container_blocks();
//
//    std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[row_idx].first,diagonal_indices[k].second);
//    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator Lblock_cit=L_blocks.find(temp_pair);
//    //NEMO_ASSERT(Lblock_cit != L_blocks.end(),
//    //      tic_toc_prefix + " have not found L block \n");
//    PetscMatrixParallelComplex* L_row_k = NULL;
//    if(Lblock_cit!=L_blocks.end())
//      L_row_k = Lblock_cit->second;
//    else
//    {  //solve it
//      PetscMatrixParallelComplex* sigmaR_row_k = NULL;
//      if(scattering_sigmaR_solver != NULL)
//      {
//        //find sigmaR( row_idx,k )
//        std::string row_name = subdomain_names[row_idx];
//        std::string col_name = subdomain_names[k];
//        extract_sigmaR_submatrix(momentum, row_name, col_name, eom_interface, sigmaR_row_k);
//      }
//
//      solve_L(this_simulation, momentum, eom_interface, number_of_offdiagonal, subdomain_names, row_idx, k, diagonal_indices,
//          offdiagonal_indices,sigmaR_row_k, L_row_k, L_container, gR_container, scattering_sigmaR_solver);
//      delete sigmaR_row_k;
//      sigmaR_row_k = NULL;
//    }
//
//    //find gR(k,k)
//    std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator gR_cit = gR_blocks.find(diagonal_indices[k]);
//    PetscMatrixParallelComplex* gR_k_k = gR_cit->second;
//    //find L(col_idx,k) -> transpose
//    std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[col_idx].first,diagonal_indices[k].second);
//    Lblock_cit=L_blocks.find(temp_pair2);
//    //NEMO_ASSERT(Lblock_cit != L_blocks.end(),
//   //     tic_toc_prefix + " have not found L block for transpose\n");
//    //PetscMatrixParallelComplex* L_col_k = Lblock_cit->second;
//
//    PetscMatrixParallelComplex* L_col_k = NULL;
//    if(Lblock_cit!=L_blocks.end())
//      L_col_k = Lblock_cit->second;
//    else
//    {  //solve it
//      PetscMatrixParallelComplex* sigmaR_col_k = NULL;
//      if(scattering_sigmaR_solver != NULL)
//      {
//        //find sigmaR( row_idx,k )
//        std::string row_name = subdomain_names[row_idx];
//        std::string col_name = subdomain_names[k];
//        extract_sigmaR_submatrix(momentum, row_name, col_name, eom_interface, sigmaR_col_k);
//
//      }
//      solve_L(this_simulation, momentum, eom_interface, number_of_offdiagonal, subdomain_names, col_idx, k, diagonal_indices,
//          offdiagonal_indices,sigmaR_col_k, L_row_k, L_container, gR_container, scattering_sigmaR_solver);
//    }
//
//    PetscMatrixParallelComplex *temp_matrix = NULL;
//    //multiply L(row_idx,k)*gR(k,k)
//    PetscMatrixParallelComplex::mult(*L_row_k,*gR_k_k, &temp_matrix);
//
//    //transpose
//    PetscMatrixParallelComplex L_k_col(L_col_k->get_num_cols(), L_col_k->get_num_rows(),
//        L_col_k->get_communicator());
//    L_col_k->transpose_matrix(L_k_col,MAT_INITIAL_MATRIX);
//
//    PetscMatrixParallelComplex *temp_matrix2 = NULL;
//    PetscMatrixParallelComplex::mult(*temp_matrix, L_k_col, &temp_matrix2);
//    delete temp_matrix;
//    temp_matrix = NULL;
//    //add result to coupling
//    L_block->add_matrix(*temp_matrix2,DIFFERENT_NONZERO_PATTERN,cplx(-1.0,0.0));
//    delete temp_matrix2;
//    temp_matrix2 = NULL;
//  }
//
//  //store coupling in L block
//  L_container->set_block_from_matrix1(*L_block,diagonal_indices[row_idx].first,diagonal_indices[col_idx].first);
//
//  if(debug_output)
//  {
//    L_block->save_to_matlab_file("L_row_" + subdomain_names[row_idx]+  "_col_" + subdomain_names[col_idx]);
//  }
//}
//
//void ScatteringForwardRGFSolver::extract_sigmaR_submatrix(const std::vector<NemoMeshPoint>& momentum,
//    std::string row_name, std::string col_name, EOMMatrixInterface* eom_interface, PetscMatrixParallelComplex*& sigmaR)
//   {
//     BlockSchroedingerModule* temp_simulation=dynamic_cast<BlockSchroedingerModule*>(eom_interface);
//     Domain* row_domain = Domain::get_domain(row_name);
//     const DOFmapInterface& row_DOFmap=temp_simulation->get_dof_map(row_domain);
//     Domain* col_domain = Domain::get_domain(col_name);
//     const DOFmapInterface& col_DOFmap=temp_simulation->get_const_dof_map(col_domain);
//     //PetscMatrixParallelComplex* temp_matrix = NULL;
//     std::string temp_name;
//     scattering_sigmaR_solver->get_data("writeable_Propagator",temp_name);
//     scattering_sigmaR_solver->get_data(temp_name,&momentum,sigmaR,&row_DOFmap, &col_DOFmap);
//     //    &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
//     if(debug_output)
//     {
//       sigmaR->assemble();
//       sigmaR->save_to_matlab_file("sigmaR_"+row_name+"_"+col_name+".m");
//     }
//
//   }

void ScatteringForwardRGFSolver::solve_diagonal_gL(std::pair<std::vector<int>,std::vector<int> >& diagonal_index, const std::vector<NemoMeshPoint>& momentum,
    PetscMatrixParallelComplex*& Sigma_lesser_contact, PetscMatrixParallelComplex*& diagonal_block_gR, PetscMatrixParallelComplex*& diagonal_block_gL, bool real_lead,
    PetscMatrixParallelComplex* scattering_sigmaL_matrix)
{

  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_gL: ";
  NemoUtils::tic(tic_toc_prefix);


  //1. call core routine for the g<

  if(scattering_sigmaL_matrix!=NULL)
  {
    Sigma_lesser_contact->matrix_convert_dense();
    if(scattering_sigmaL_matrix!=NULL)
      scattering_sigmaL_matrix->matrix_convert_dense();
    Sigma_lesser_contact->add_matrix(*scattering_sigmaL_matrix, DIFFERENT_NONZERO_PATTERN,std::complex<double> (1.0,0.0));
  }

  if(debug_output)
  {
   Sigma_lesser_contact->save_to_matlab_file("summed_lesser_self.m");
   if(scattering_sigmaL_matrix!=NULL)
     scattering_sigmaL_matrix->save_to_matlab_file("sigma_lesser_diag.m");
   diagonal_block_gR->save_to_matlab_file("gR_diag_for_gL.m");
  }


  PropagationUtilities::core_correlation_Green_exact(Sigma_lesser_contact,diagonal_block_gL, diagonal_block_gR);


  //diagonal_block_gL->save_to_matlab_file("gL.m");

  //2. store in container for g<
  Propagator::PropagatorMap& result_prop_map=lesser_green_propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it=result_prop_map.find(momentum);
  if(prop_it==result_prop_map.end())
  {
    result_prop_map[momentum]=NULL;
    prop_it=result_prop_map.find(momentum);
  }
  if(prop_it->second==NULL)
  {
    PetscMatrixParallelComplexContainer* new_container=new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),Hamiltonian->get_num_cols(),Hamiltonian->get_communicator());
    lesser_green_propagator->allocated_momentum_Propagator_map[momentum]=true;
    prop_it->second=new_container;
  }
  PetscMatrixParallelComplexContainer* temp_container=dynamic_cast<PetscMatrixParallelComplexContainer*> (prop_it->second);

  NemoMath::symmetry_type type = NemoMath::antihermitian;
  symmetrize(diagonal_block_gL,type);
  temp_container->set_block_from_matrix1(*diagonal_block_gL,diagonal_index.first,diagonal_index.second);



  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringForwardRGFSolver::do_solve_retarded(Propagator*& /*output_Propagator*/, const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result)
{
  //Propagator* output_Propagator_temp=output_Propagator;
  //output_Propagator_temp=NULL;
  get_Greensfunction(momentum_point,result,NULL,NULL,NemoPhys::Fermion_retarded_Green);
}

//
//void ScatteringForwardRGFSolver::do_reinit()
//{
//  // TODO : Clean variables, reinitialize external libraries, etc ...
//}

void ScatteringForwardRGFSolver::set_writeable_propagator()
{
  NemoPhys::Propagator_type temp_type;
  if(particle_type_is_Fermion)
  {
    name_of_writeable_Propagator=get_name()+"_retarded_Green_Fermion"; 
    Propagator_types[name_of_writeable_Propagator]=NemoPhys::Fermion_retarded_Green;
    temp_type = NemoPhys::Fermion_retarded_Green;
  }
  else
  {
    name_of_writeable_Propagator=get_name()+"_retarded_Green_Boson"; 
    Propagator_types[name_of_writeable_Propagator]=NemoPhys::Boson_retarded_Green;
    temp_type = NemoPhys::Boson_retarded_Green;
  }
  delete writeable_Propagator;
  Propagator* temp_propagator = new Propagator(name_of_writeable_Propagator, temp_type);
  Propagators[name_of_writeable_Propagator]=temp_propagator;
  writeable_Propagator=temp_propagator;

  this_is_combine_Propagation=false; 

  ready_Propagator_map[name_of_writeable_Propagator]=false;
  Propagator_Constructors[name_of_writeable_Propagator]=this;

  type_of_writeable_Propagator=NemoPhys::Inverse_Green;

  //additionally need to create and store propagator for g<
  if(particle_type_is_Fermion)
  {
    lesser_green_propagator_name = get_name()+"_lesser_Green_Fermion";
    lesser_green_propagator_type = NemoPhys::Fermion_lesser_Green;
  }
  else
  {
    lesser_green_propagator_name = get_name()+"_lesser_Green_Boson";
    lesser_green_propagator_type = NemoPhys::Boson_lesser_Green;
  }
  delete lesser_green_propagator;
  lesser_green_propagator = new Propagator(lesser_green_propagator_name, lesser_green_propagator_type);
  lesser_propagator_ready_map[lesser_green_propagator_type]=false;
  Propagator_Constructors[lesser_green_propagator_name]=this;

}


void ScatteringForwardRGFSolver::delete_propagator_matrices(Propagator* input_Propagator, const std::vector<NemoMeshPoint>* momentum, const DOFmap* row_DOFmap,
    const DOFmap* col_DOFmap)
{
  if(input_Propagator==NULL||input_Propagator->get_propagator_type()==writeable_Propagator->get_propagator_type())
  {
    Propagation::delete_propagator_matrices(input_Propagator,momentum, row_DOFmap, col_DOFmap);
    delete_propagator_matrices(lesser_green_propagator, momentum, row_DOFmap, col_DOFmap);
  }

  //delete_propagator_matrices can be called twice if this is NULL so only really delete the lesser propagator on the 2nd call
  else if(input_Propagator->get_propagator_type()==lesser_green_propagator->get_propagator_type())
  {
    Propagator *temp_propagator = lesser_green_propagator;
    if(momentum!=NULL)
    {
      std::map<const std::vector<NemoMeshPoint>,bool>::iterator it=temp_propagator->allocated_momentum_Propagator_map.find(
          *momentum);
      NEMO_ASSERT(it!=temp_propagator->allocated_momentum_Propagator_map.end(),
          "Propagation(\""+get_name()+"\")::delete_propagator_matrices have not found momentum in \""+temp_propagator->get_name()+"\"\n");
      if(it->second)
      {
        Propagator::PropagatorMap::iterator it2=temp_propagator->propagator_map.find(*momentum);
        NEMO_ASSERT(it2!=temp_propagator->propagator_map.end(),
            "Propagation(\""+get_name()+"\")::delete_propagator_matrices have not found momentum in propagator_map of \""
            +temp_propagator->get_name()+"\"\n");
        if(it2->second!=NULL)
        {
          msg<<"ScatteringForwardRGFSolver(\"" << get_name() << "\")::delete_propagator_matrices: "<<it2->second<<"\n";
          if(/*input_Propagator->get_name().find("combine")!=std::string::npos &&*/ row_DOFmap!=NULL)
          {

            const DOFmapInterface& large_DOFmap=Hamilton_Constructor->get_dof_map(get_const_simulation_domain());
            std::vector<int> rows_to_delete;

            Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,*row_DOFmap,rows_to_delete,get_const_simulation_domain());

            std::vector<int> cols_to_delete;
            if(col_DOFmap!=NULL)
            {
              Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,*col_DOFmap,cols_to_delete,get_const_simulation_domain());
            }
            else
              cols_to_delete=rows_to_delete;

            PetscMatrixParallelComplexContainer* temp_container=dynamic_cast<PetscMatrixParallelComplexContainer*>(it2->second);
            if(temp_container!=NULL)
              temp_container->delete_submatrix(rows_to_delete,cols_to_delete);
            else
            {
              delete it2->second;
              it2->second=NULL;
              it->second=false;
            }
            //since this is combine solver, need to quit the function inorder not to mess up with job_done_map etc
            return void();
          }
          else
          {
            delete it2->second;
            it2->second=NULL;
          }
          temp_propagator->propagator_map[it2->first]=NULL;
        }
        it->second=false;

      }
      else {}
      set_job_done_momentum_map(&(temp_propagator->get_name()), momentum, false);
    }
    else
    {
      msg<<"ScatteringForwardRGFSolver(\"" << get_name() << "\")::delete_propagator_matrices: "<<"deleting Propagator \""<<temp_propagator->get_name()<<"\"\n";
      Propagator::PropagatorMap::iterator it2=temp_propagator->propagator_map.begin();
      for(; it2!=temp_propagator->propagator_map.end(); ++it2)
      {
        delete_propagator_matrices(temp_propagator, &(it2->first));
      }
    }
  }
  //else
  //{
  //  {
  //    if(lesser_green_propagator!=NULL)
  //      delete_propagator_matrices(lesser_green_propagator,NULL);
  //  }
  //}
}

void ScatteringForwardRGFSolver::set_job_done_momentum_map(const std::string* Propagator_name, const std::vector<NemoMeshPoint>* momentum_point,
    const bool input_status)
{
  if(Propagator_name==NULL||*Propagator_name==writeable_Propagator->get_name())
    Propagation::set_job_done_momentum_map(Propagator_name,momentum_point,input_status);
  else if(*Propagator_name==lesser_green_propagator_name)
  {
    //bool always_ready=options.get_option("always_ready",false);
    bool input_status2 = always_ready || input_status; //DM:boolean operation is faster..
    {
      if(momentum_point!=NULL)
      {
        //it->second[*momentum_point]=input_status2; //Normally this is faster
        lesser_propagator_job_done_momentum_map[*momentum_point]=input_status2;
      }
      else
      {
        std::map<std::vector<NemoMeshPoint>, bool>::iterator it2=lesser_propagator_job_done_momentum_map.begin();
        for(; it2!=lesser_propagator_job_done_momentum_map.end(); it2++)
          it2->second=input_status2;
      }

    }

  }
}

void ScatteringForwardRGFSolver::get_data(const std::string& variable, Propagator*& Propagator_pointer)
{
  if(variable.find("retarded")!=std::string::npos)
    Propagator_pointer = writeable_Propagator;
  else if(variable.find("lesser")!=std::string::npos)
    Propagator_pointer = lesser_green_propagator;
  else
    throw std::runtime_error("ScatteringForwardRGFSolver(\""+get_name()+"\")::get_data propagator name unknown variable ");

}

void ScatteringForwardRGFSolver::get_start_lead_sigma(const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = "ScatteringForwardRGFSolver::(\"" + this->get_name() + "\")::get_start_lead_sigma: ";
  NemoUtils::tic(tic_toc_prefix);

  //1. get sub domain names
  if (subdomain_names.size() < 1)
    repartition_solver->get_subdomain_names(get_const_simulation_domain(), true, subdomain_names);

  //2. get the Hamiltonian block matrices
  EOMMatrixInterface* eom_interface = NULL;
  get_device_Hamiltonian(momentum, Hamiltonian, eom_interface);

  //3. get the contact self-energy
  get_contact_sigma(momentum, eom_interface, result, subdomain_names[0], lead_domain, Hamiltonian);

  delete Hamiltonian;
  Hamiltonian = NULL;

  NemoUtils::toc(tic_toc_prefix);
}

void ScatteringForwardRGFSolver::set_description()
{
  description = "This solver is the library-kind version of the Forward RGF suitable for scattering";
}

void ScatteringForwardRGFSolver::set_input_options_map()
{
  Greensolver::set_input_options_map();
  //set_input_option_map("OPTION_NAME",InputOptions::Req_Def("OPTION_DESCRIPTION")); //TODO: Set options
  //set_input_option_map("OPTION_NAME",InputOptions::NonReq_Def("DEFAULT_VALUE","OPTION_DESCRIPTION")); //TODO: Set options
}
