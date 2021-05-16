// NEMO5, The Nanoelectronics simulation package.
// This package is a free software.
// It is distributed under the NEMO5 Non-Commercial License (NNCL).
// Purdue Research Foundation, 1281 Win Hentschel Blvd., West Lafayette, IN 47906, USA
//$Id: Propagation.cpp 24117 $
/*  Purpose:    Propagation class = base class for every NEGF-simulation
                irrespective of system's dimensionality or its representation
    Note:       uses the propagator class as container for all propagator related data
 */



#include "Simulation.h"
#include "Propagator.h"
#include "Propagation.h"
#include "Matrix.h"
#include <map>
#include <complex>
#include <stdexcept>
#include <sstream>
#include "NemoUtils.h"
#include "pcl_lapack.h"
#include "PetscMatrixParallel.h"
#include "petscksp.h" // for LU factorization
#include "petscmat.h"
#include "petscerror.h"
#include "PetscMatrixParallelComplex.h"
#include "PetscMatrixParallelComplexContainer.h"
#include "LinearSolverPetscComplex.h"
#include "LinearSolverPetscDouble.h"
#include "LinearSolverPetsc.h"
#include "EigensolverPetscDouble.h"
#include "EigensolverSlepcDouble.h"
#include "NemoMesh.h"
#include "NemoMath.h"
#include "NemoMICUtils.h"
#include "NemoPhys.h"
#include "NemoUtils.h"
#include "Espace.h"
#include "Kspace.h"
#include <mpi.h>
#include "Nemo.h"
#include "DenseMatrixNemo.h"
#include "Domain.h"
#include "HamiltonConstructor.h"
#include "Material.h"
#include "StrainVFF.h"
#include "Transformation.h"
#include "TaskOffload.h"

#include "Atom.h"
#include "AtomStructNode.h"
#include "ActiveAtomIterator.h"
#include <iostream>
#include <fstream>
#include "OutputVTK.h"
#include "OutputXYZ.h"
#include "OutputDX.h"
#include "OutputSilo.h"
#include "libmesh.h"
#include "meshfree_interpolation.h"
#include "radial_basis_interpolation.h"
#include "QuantumNumberUtils.h"
#include "PropagationUtilities.h"
#include "BlockSchroedingerModule.h"
#include "TransformationUtilities.h"
#include "BackwardRGFSolver.h"
#include <omp.h>




#ifndef NO_MKL
#include <mkl.h>
#endif
using NemoUtils::MsgLevel;
using namespace libMesh;



Propagation::~Propagation()
{
  //test_debug();
  //tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::~Propagation ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix = "Propagation(\""+get_name()+"\")::~Propagation() ";
  msg.set_level(MsgLevel(4));

  //debug: put all to comments for debugging -------------------
  std::string tic_toc_prefix1 = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::~Propagation mesh clean ");
  NemoUtils::tic(tic_toc_prefix1);
  //if(!options.get_option("disable_meshclean",false)&&produced_a_mesh)
  if(disable_meshclean==false&&produced_a_mesh)//changed to reduce string operation Pengyu
  {
    std::map<std::string,Simulation*>::iterator it = Mesh_Constructors.begin();
    if(parallelized_by_this())
    {
      //collect all pointers that point to the same object
      //call delete only once and set the remaining pointers to NULL
      std::map<NemoMesh*,std::vector<NemoMesh*> >::iterator it3=Mesh_tree_topdown.begin();
      for(; it3!=Mesh_tree_topdown.end(); ++it3)
      {
        NemoMesh* temp_mesh=it3->first;
        //delete temp_mesh;
        //it3->first=NULL;
        for(unsigned int i=0; i<it3->second.size(); i++)
        {
          temp_mesh=it3->second[i];
          it=Mesh_Constructors.find(temp_mesh->get_name());
          Simulation* temp_sim=it->second;
          if(temp_sim!=NULL)
          {
            InputOptions& mesh_options=temp_sim->get_reference_to_options();
            NEMO_ASSERT(it!=Mesh_Constructors.end(),prefix+"have not found constructor of \""+temp_mesh->get_name()+"\"\n");
            bool testL=false;
            if(mesh_options.check_option("non_rectangular"))
              testL=mesh_options.get_option("non_rectangular",false);
            if((it->second==this||options.check_option(temp_mesh->get_name()+"_parent"))
                &&!testL)
              delete temp_mesh;
          }
          /*else
            std::cerr<<prefix<<"Constructor of \""+temp_mesh->get_name()+"\" is \""+it->second->get_name()+"\"\n";*/
          it3->second[i]=NULL;
        }
      }
    }

    for(it = Mesh_Constructors.begin(); it!=Mesh_Constructors.end(); ++it)
    {
      Simulation* temp_simulation = it->second;
      if(temp_simulation==this)
      {
        msg<<prefix<<"deleting " << it->first<<std::endl;
        std::map<std::string,NemoMesh*>::iterator Mesh_it=Momentum_meshes.find(it->first);
        if(Mesh_it!=Momentum_meshes.end())
        {
          if(Mesh_it->second!=NULL)
            delete Mesh_it->second;
          Mesh_it->second=NULL;
        }
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix1);
  //debug: put all to comments for debugging -------------------
  //
  msg.set_level(MsgLevel(4));
  msg<<prefix<<"all meshes are destroyed\n";
  std::map<std::string,Simulation*>::iterator it2=pointer_to_Propagator_Constructors->begin();

  std::string tic_toc_prefix2 = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::~Propagation iterator on Propagator map ");
  NemoUtils::tic(tic_toc_prefix2);
  for(std::map<std::string, const Propagator*>::iterator temp_it2=Propagators.begin(); temp_it2!=Propagators.end(); temp_it2++)
  {
    std::string tic_toc_prefixfind = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\"):: writeable_Propagators.find() ");
    NemoUtils::tic(tic_toc_prefixfind);
    //std::map<std::string, Propagator*>::iterator temp_it=writeable_Propagators.find(temp_it2->first);
    //test_debug();


    //this section is meant to inform the book keeping that this solver is no longer available
    //do this only if the book keeping systems is available (i.e. as long as Parallelizer is available)
    NemoUtils::toc(tic_toc_prefixfind);
    bool book_keeping_is_alive = true;
    //the following comment lines are correct for central storage
    //std::string name_of_parallelizer;
    //name_of_parallelizer=options.get_option("Parallelizer",std::string(""));
    //Simulation* temp_parallelizer=find_simulation(name_of_parallelizer);
    //if(temp_parallelizer==NULL)
    //  book_keeping_is_alive=false;
    if(book_keeping_is_alive)
    {

      bool if_parallelizer_is_this=false;
      if(Parallelizer==this)
        if_parallelizer_is_this=true;

      if(if_parallelizer_is_this)
      {
        //loop over all known constructors
        std::map<std::string,Simulation*>::iterator it2=pointer_to_Propagator_Constructors->begin();
        //tell each constructor to erase their Propagator matrices
        for(; it2!=pointer_to_Propagator_Constructors->end(); ++it2)
        {
          if(it2->second!=NULL)
          {
            Propagation* temp_prop=dynamic_cast<Propagation*>(it2->second);
            if(temp_prop!=NULL)
            {
              temp_prop->delete_propagator(NULL);
              temp_prop->delete_propagator_matrices(NULL);
            }
          }
        }
      }
      else
      {
        std::map<std::string,Simulation*>::iterator it2=pointer_to_Propagator_Constructors->find(temp_it2->first);

        if(it2!=pointer_to_Propagator_Constructors->end()) //,prefix+"have not found constructor of \""+temp_it2->first+"\"\n");
        {
          //if(temp_it!=writeable_Propagators.end())
          if(name_of_writeable_Propagator==temp_it2->first)
          {
            //if this is a constructor, delete the Propagator and erase this from the constructor map of all solvers and callers
            if(it2->second==this)
            {
              delete_propagator_matrices(writeable_Propagator);
              delete_propagator(&(name_of_writeable_Propagator));
              writeable_Propagator=NULL;
              ////tell all solvers and callers, that this is no longer the constructor...
              //std::string tic_toc_prefixfindcallers = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\"):: Propagator_Callers.find() )";
              //NemoUtils::tic(tic_toc_prefixfindcallers);
              //std::map<std::string, std::set<Simulation*> >::iterator solver_it=Propagator_Callers.find(name_of_writeable_Propagator);
              //if(solver_it!=Propagator_Callers.end())
              //{
              //  std::set<Simulation*>::iterator all_solvers_it=solver_it->second.begin();
              //  for(; all_solvers_it!=solver_it->second.end(); ++all_solvers_it)
              //  {
              //    //tell the caller that this is no longer a constructor
              //    std::map<std::string, Simulation* >* pointer_to_foreign_constructor_list=NULL;
              //    (*all_solvers_it)->get_data("Propagator_Constructors",pointer_to_foreign_constructor_list);
              //    NEMO_ASSERT(pointer_to_foreign_constructor_list!=NULL,prefix+"received NULL as constructor list pointer\n");
              //    std::map<std::string, Simulation* >::iterator temp_list_it=pointer_to_foreign_constructor_list->find(name_of_writeable_Propagator);
              //    if(temp_list_it!=pointer_to_foreign_constructor_list->end())
              //      pointer_to_foreign_constructor_list->erase(temp_list_it);
              //  }
              //  Propagator_Callers.erase(solver_it);
              //}
              //NemoUtils::toc(tic_toc_prefixfindcallers);
              Propagator_is_initialized[name_of_writeable_Propagator]=false;


              //delete this from the Constructor map:
              //it2=pointer_to_Propagator_Constructors->find(temp_it2->first);
              //if(it2!=pointer_to_Propagator_Constructors->end())

              //std::map<std::string,Simulation*>::const_iterator temp_cit_debug=pointer_to_Propagator_Constructors->begin();
              //for(;temp_cit_debug!=pointer_to_Propagator_Constructors->end();++temp_cit_debug)
              //  std::cerr<<prefix<<temp_cit_debug->first<<" "<<temp_cit_debug->second->get_name()<<"\n";

              it2=pointer_to_Propagator_Constructors->find(temp_it2->first);
              if(it2!=pointer_to_Propagator_Constructors->end())
              {
                //std::cerr<<prefix<<"going to erase: "<<it2->first <<" "<< it2->second<<"\n";
                pointer_to_Propagator_Constructors->erase(it2);
              }
            }
            //if this is not a constructor, delete this from the Solver and Caller map of the Propagator_Constructor

            else if(it2!=pointer_to_Propagator_Constructors->end())
            {
              //erase from solvers
              std::string tic_toc_prefixconend = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\"):: pointer_to_Propagator_Constructors->end()");
              NemoUtils::tic(tic_toc_prefixconend);
              NemoUtils::toc(tic_toc_prefixconend);
            }
          }
          else
          {
            //std::string tic_toc_prefixelse = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::~Propagation else ");
            //NemoUtils::tic(tic_toc_prefixelse);
            ////std::cerr<<prefix<<"we should be here \n\n\n";
            //std::map<std::string, std::set<Simulation*> >* caller_map_of_constructor=NULL;
            ////erase from callers
            ////std::cerr<<prefix<<"constructor of \""<<it2->first<<"\" is \""<<it2->second->get_name()<<"\"\n";
            ////if(!ignore_callers_ready_map)
            //{
            //  it2->second->get_data("Propagator_Callers",caller_map_of_constructor);
            //  if(caller_map_of_constructor!=NULL)
            //  {
            //    std::map<std::string, std::set<Simulation*> >::iterator solver_it=caller_map_of_constructor->begin();
            //    for(; solver_it!=caller_map_of_constructor->end(); ++solver_it)
            //    {
            //      std::set<Simulation*>::iterator set_it=solver_it->second.find(this);
            //      if(set_it!=solver_it->second.end())
            //      {
            //        //std::cerr<<prefix<<"going to erase "<<get_name()<<" from caller list of "<<solver_it->first<<" stored in solver "<<it2->second->get_name()<<"\n";
            //        solver_it->second.erase(set_it);
            //      }
            //      //else
            //      //  std::cerr<<prefix<<"have not found \""+get_name()+"\" in the solver list of \""+it2->second->get_name()+"\"\n";
            //    }
            //  }
            //}
            ////else
            ////{
            ////  std::cerr<<prefix<<"have received NULL for the Propagator_Caller map from \""+it2->second->get_name()+"\"\n";
            ////  it2->second->get_data("Propagator_Callers",caller_map_of_constructor);
            ////  throw std::runtime_error("STOP\n");
            ////}
            //NemoUtils::toc(tic_toc_prefixelse);
          }
        }
      }
    }
    /*else
      std::cerr<<prefix+"have not found constructor of \""+temp_it2->first+"\"\n";*/
    /*if(temp_it2->first.find(std::string("Buettiker_probe"))!=std::string::npos)
    {
      std::cerr<<prefix<<"handling BP\n";
      test_debug();
    }*/
  }
  NemoUtils::toc(tic_toc_prefix2);

  std::string tic_toc_prefix3 = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::~Propagation delete everything ") ;
  NemoUtils::tic(tic_toc_prefix3);
  delete temporary_sub_matrix;
  msg.set_level(MsgLevel(4));
  msg<<"all Propagators in \""<<this->get_name()<<"\" destroyed\n";

  delete into_device_propagating_modes;
  delete into_device_decaying_modes;
  delete out_of_device_propagating_modes;
  delete out_of_device_decaying_modes;
  delete into_device_propagating_phase;
  delete into_device_decaying_phase;
  delete out_of_device_propagating_phase;
  delete out_of_device_decaying_phase;

  delete into_device_modes;
  delete into_device_phase;
  delete out_of_device_modes;
  delete out_of_device_phase;

  delete out_of_device_velocity;
  delete into_device_velocity;
  delete out_of_device_propagating_velocity;
  delete into_device_propagating_velocity;
  NemoUtils::toc(tic_toc_prefix3);

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::base_init()
{
  //Propagation::list_of_Propagators.insert(this);
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::base_init ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::base_init: ";
  print_all_mesh_points = options.get_option("print_all_mesh_points",false);

  //set some variable that replace former option-accesses
  if(use_input_options)
  {
    no_file_output = options.get_option("no_file_output",false);
    use_matrix_0_threshold=options.check_option("matrix_0_threshold");
    clean_up=options.get_option("clean_up",false);
    do_outputL=options.get_option("do_output",false);
    one_energy_only=options.get_option("one_energy_only",false);
    debug_output=options.get_option("debug_output",false);
    save_local_sigma=options.get_option("save_local_sigma",false);
    debug_output_job_list=options.get_option("debug_output_job_list",false);
    temperature=options.get_option("temperature",NemoPhys::temperature);
    temperature_in_eV=temperature*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
    chemical_potential=options.get_option("chemical_potential",0.0);
    threshold_energy_for_lead = options.get_option("threshold_energy_for_lead",double(-1E10));
    temperature1=options.get_option("temperature1",NemoPhys::temperature);

    temperature2=options.get_option("temperature2",NemoPhys::temperature);

    chemical_potential1=options.get_option("chemical_potential1",0.0);
    chemical_potential2=options.get_option("chemical_potential2",0.0);
    threshold_energy_for_lead1 = options.get_option("threshold_energy_for_lead1",double(-1E10));
    threshold_energy_for_lead2 = options.get_option("threshold_energy_for_lead2",double(-1E10));
    avoid_copying_hamiltonian=options.get_option("avoid_copying_hamiltonian",false);
    use_analytical_momenta=options.check_option("analytical_momenta");
    sigma_convert_dense=options.get_option("sigma_convert_dense",false);
    solve_on_single_replica=options.get_option("solve_on_single_replica",true);
    std::string temp_string=options.get_option("inversion_method",std::string("exact"));
    inversion_method = get_inversion_method_from_options(temp_string);
    add_combine_solver_matrices = options.get_option("add_combine_solver_matrices",bool(false));
    skip_book_keeping = options.get_option("skip_book_keeping",false);

    store_result_elsewhere=options.check_option("store_result_in");
    disable_meshclean = options.get_option("disable_meshclean",false);

    electron_hole_model = options.get_option("electron_hole_model",false);

    if(store_result_elsewhere)
    {
      const std::string combine_propagator_name = options.get_option("store_result_in",std::string(""));
      combine_solver_for_storage = dynamic_cast<Propagation*> (find_source_of_data(combine_propagator_name));
      /*std::string combine_solver_name=options.get_option("store_result_in",std::string(""));
      combine_solver_for_storage=find_simulation(combine_solver_name);*/
      NEMO_ASSERT(combine_solver_for_storage!=NULL,tic_toc_prefix+"have not found storage-solver for \""+combine_propagator_name+"\"\n");
    }
  }
  temperature1_in_eV=temperature1*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  temperature2_in_eV=temperature2*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;

  msg.set_level(MsgLevel(4));
  msg << prefix+"initializing (base level)"<<std::endl;

  //set the Parallelizer - also used as central storage of book keeping
  std::string who_is_parallelizing;
  if (options.check_option("Parallelizer"))
  {
    who_is_parallelizing=options.get_option("Parallelizer",std::string(""));
    Simulation* temp_simulation = this->find_simulation(who_is_parallelizing);
    NEMO_ASSERT(temp_simulation!=NULL,prefix+"unknown Parallelizer \""+ who_is_parallelizing + "\"\n");
    if (Parallelizer!=NULL)
    {
      NEMO_ASSERT(Parallelizer->get_name()==temp_simulation->get_name(),prefix+"Parallelizer is ambiguously defined\n");
    }
    else
    {
      Parallelizer = temp_simulation;
    }
    //Parallelizer->get_data("Propagator_Constructors",pointer_to_Propagator_Constructors);
  }
  else
  {
    NEMO_ASSERT(Parallelizer!=NULL,prefix+"no Parallelizer defined\n");
  }

  if(!offload_solver_initialized && options.check_option("offload_solver"))
  {
    std::string offload_solver_name = options.get_option("offload_solver", std::string(""));
    offload_solver = dynamic_cast<TaskOffload*>(this->find_simulation(offload_solver_name));
    NEMO_ASSERT(offload_solver!=NULL,prefix+"have not found offload solver \""+offload_solver_name+"\"\n");

    NemoUtils::MsgLevel prev_level = msg.get_level();
    OUT_DEBUG << prefix << this->get_name() << " : Initializing offload solver " << offload_solver_name << "\n";
    msg.set_level(prev_level);

    offload_solver_initialized = true;
  }
  //let the pointer_t0_Propagator_Constructors point to the storage of the Parallelizer (to centralize the book keeping)
  //Parallelizer->get_data("Propagator_Constructors",pointer_to_Propagator_Constructors);


  // -----------------------------------
  // define the possible momenta of the Propagator entities
  // set parallelization first
  // then create NemoMeshes, such as k_space and E_space
  // or get the meshes from another simulation
  // -----------------------------------
  set_momentum_meshes();
  //test_debug();

  // -----------------------------------
  // read in which Green's functions be calculated
  // -----------------------------------
  if(read_NEGF_object_list)
    base_read_function_list();
  else
  {
    set_writeable_propagator();
  }

  //test_debug();
  // -----------------------------------
  // get the Hamilton constructor
  // -----------------------------------
  if(use_input_options&&Hamilton_Constructor==NULL)
  {
    std::string Hamilton_constructor_name;
    if (options.check_option("Hamilton_constructor_source"))
    {
      Hamilton_constructor_name=options.get_option("Hamilton_constructor_source",std::string(""));
      Simulation* temp_simulation = find_simulation(Hamilton_constructor_name);
      NEMO_ASSERT(temp_simulation != NULL, prefix+"have not found simulation \""+Hamilton_constructor_name+"\"\n");
      temp_simulation->get_data("Hamilton_constructor",Hamilton_Constructor);
    }
    else if (options.check_option("Hamilton_constructor"))
    {
      Hamilton_constructor_name=options.get_option("Hamilton_constructor",std::string(""));
      Hamilton_Constructor = this->find_simulation(Hamilton_constructor_name);
    }
    else
      throw std::invalid_argument(prefix+"define \"Hamilton_constructor\", i.e. the simulation that constructs the Hamiltonian\n");
  }


  NEMO_ASSERT(Hamilton_Constructor!=NULL,prefix+"Hamilton_Constructor has not been found!\n");
  //Hamilton_Constructor->set_simulation_domain(this->get_simulation_domain());
  //NEMO_ASSERT(Hamilton_Constructor->get_const_simulation_domain() == get_const_simulation_domain(),
  //            prefix+"Hamilton_Constructor \""+Hamilton_Constructor->get_name()+"\" defined in a wrong domain\n");

  // -----------------------------------
  // initialize the Propagator matrices with the size of the domain on the do_solve level (after dofmap is initialized)
  // -----------------------------------
  //NOTE: do be done in do_solve, because then the momentum meshes are filled!
  //test_debug();
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::fill_momentum_meshes(void)
{

  std::string prefix="Propagation(\""+this->get_name()+"\")::fill_momentum_meshes: ";
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::fill_momentum_meshes ");
  NemoUtils::tic(tic_toc_prefix);

  if(Parallelizer==this)
  {
    std::map<std::string, Simulation*>::iterator it=Mesh_Constructors.begin();
    for (; it!=Mesh_Constructors.end()&& !momentum_meshes_are_set; ++it)
    {
      if(it->second!=this)
      {
        if (momentum_mesh_types.find(it->first)->second==NemoPhys::Momentum_1D||momentum_mesh_types.find(it->first)->second==NemoPhys::Momentum_2D||
            momentum_mesh_types.find(it->first)->second==NemoPhys::Momentum_3D)
        {
          NemoMesh* temp_k_space=NULL; //=new NemoMesh();
          it->second->get_data("k_space_"+it->first,temp_k_space);
          NEMO_ASSERT(temp_k_space!=NULL,prefix+"have received NULL for \""+it->first+"\" from \""+it->first+"\"\n");
          //copy the mesh coordinates of temp_k_space to Momentum_meshes.find(it->first)
          std::vector<NemoMeshPoint*> temp_points=temp_k_space->get_mesh_points();

          msg.set_level(MsgLevel(4));

          it->second->get_data("k_space_"+it->first,Momentum_meshes.find(it->first)->second);

          NEMO_ASSERT(Momentum_meshes.find(it->first)!=Momentum_meshes.end(),prefix+"have not found mesh \""+it->first+"\"\n");
          if(Momentum_meshes.find(it->first)->second==NULL)
            Momentum_meshes.find(it->first)->second=temp_k_space;
          NEMO_ASSERT(Momentum_meshes.find(it->first)->second!=NULL,prefix+"mesh \""+it->first+"\" is NULL\n");
        }
        else if(momentum_mesh_types.find(it->first)->second==NemoPhys::Energy||momentum_mesh_types.find(it->first)->second==NemoPhys::Complex_energy)
        {
          //NemoMesh* temp_e_space=NULL; //=new NemoMesh();
          std::map<std::string, NemoMesh*>::iterator it_tmp = Momentum_meshes.find(it->first);
          NEMO_ASSERT(it_tmp!=Momentum_meshes.end(), prefix + "have not found\""+it->first+"\"\n");

          InputOptions& mesh_options=it->second->get_reference_to_options();
          if(!mesh_options.get_option(std::string("non_rectangular"),false))
            //if(!options.get_option("non_rectangular_"+it->first,false))
            it->second->get_data("e_space_"+it->first,it_tmp->second);
          else
          {
            //1.figure out which meshes to include in the get_data call
            std::vector<std::string> list_of_extra_momenta;
            options.get_option("Hamilton_momenta",list_of_extra_momenta);
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
                temp_vector2[i].set_new_mesh_name("dummy_k");
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
            //3. call get_data with all required momentum information
            it->second->get_data("e_space_"+it->first,temp_vector2,it_tmp->second);
            //it->second->set_job_done_momentum_map(NULL,&temp_vector2,0);
          }


          msg.set_level(MsgLevel(4));

        }

        else if (momentum_mesh_types.find(it->first)->second==NemoPhys::Valley)
        {
          NemoMesh* temp_v_space=NULL; //=new NemoMesh();
          it->second->get_data("v_space_"+it->first,temp_v_space);
          NEMO_ASSERT(temp_v_space!=NULL,prefix+"have received NULL for \""+it->first+"\" from \""+it->first+"\"\n");
          //copy the mesh coordinates of temp_k_space to Momentum_meshes.find(it->first)
          std::vector<NemoMeshPoint*> temp_points=temp_v_space->get_mesh_points();

          std::vector<NemoMeshPoint*> temppoints=temp_v_space->get_mesh_points();

          std::vector<std::vector<double> > temp_coords_points(temp_points.size());
          for(unsigned int i=0; i<temp_points.size(); i++)
            temp_coords_points[i]=temp_points[i]->get_coords();

          it->second->get_data("v_space_"+it->first,Momentum_meshes.find(it->first)->second);

          NEMO_ASSERT(Momentum_meshes.find(it->first)!=Momentum_meshes.end(),prefix+"have not found mesh \""+it->first+"\"\n");
          if(Momentum_meshes.find(it->first)->second==NULL)
            Momentum_meshes.find(it->first)->second=temp_v_space;
          NEMO_ASSERT(Momentum_meshes.find(it->first)->second!=NULL,prefix+"mesh \""+it->first+"\" is NULL\n");


        }

        else
          throw std::invalid_argument(prefix+"there are only Energy, valley and Momentum implemented so far\n");
      }
      else
      {
        create_NemoMeshes(it->first);

      }
    }
    momentum_meshes_are_set=true;
  }
  else
  {
    Parallelizer->get_data("Momentum_meshes",pointer_to_Momentum_meshes);
    momentum_meshes_are_set=true;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::fill_momentum_mesh_tree(const std::string& name, std::vector<std::vector<double> >& coordinates)
{

  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::fill_momentum_mesh_tree ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::fill_momentum_tree: ";
  throw std::runtime_error(prefix+"stop\n");
  //storing all NemoMeshes into a set
  std::set<NemoMesh*> all_meshes;
  std::map<NemoMesh*,NemoMesh* >::const_iterator c_it=Mesh_tree_downtop.begin();
  for(; c_it!=Mesh_tree_downtop.end(); ++c_it)
  {
    all_meshes.insert(c_it->first);
    if(c_it->second!=NULL)
      all_meshes.insert(c_it->second);
  }
  std::set<NemoMesh*>::iterator it=all_meshes.begin();
  for(; it!=all_meshes.end(); ++it)
  {
    if((*it)->get_name()==name)
    {
      NEMO_ASSERT((*it)->get_num_points()==coordinates.size(),prefix+"mismatch in number of points\n");
      (*it)->set_mesh(coordinates);
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}



bool Propagation::is_Propagator_initialized(const std::string Propagator_name)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::is_Propagator_initialized ");
  NemoUtils::tic(tic_toc_prefix);
  //NEMO_ASSERT(pointer_to_Propagator_Constructors->find(Propagator_name)!=pointer_to_Propagator_Constructors->end(),
  //            tic_toc_prefix+"have not found constructor of \""+
  //            Propagator_name+"\"\n");;
  //Simulation* constructor=pointer_to_Propagator_Constructors->find(Propagator_name)->second;
  //NEMO_ASSERT(constructor!=NULL,tic_toc_prefix+"constructor of \""+Propagator_name+"\" is NULL\n");
  Simulation* constructor=this;
  std::map<std::string, bool>* temp_map_pointer=NULL;
  //2. get the Propagator_is_initialized map from the constructor
  constructor->get_data(std::string("Propagator_is_initialized"),temp_map_pointer);
  NEMO_ASSERT(temp_map_pointer!=NULL,tic_toc_prefix+"received NULL for temp_map_pointer\n");
  //3. check or update the Propagator_is_initialized map
  std::map<std::string, bool>::iterator it=temp_map_pointer->begin();
  it=temp_map_pointer->find(Propagator_name);
  if(it!=temp_map_pointer->end())
  {
    NemoUtils::toc(tic_toc_prefix);
    return it->second;
  }
  else
  {
    (*temp_map_pointer)[Propagator_name]=false;
    NemoUtils::toc(tic_toc_prefix);
    return false;
  }
}

void Propagation::fill_all_momenta(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::fill_all_momenta ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::fill_all_momenta: ";

  if(Parallelizer==this)
  {
    std::vector<std::vector<NemoMeshPoint* > > all_momentum_points(number_of_momenta);
    for(unsigned int i=0; i<number_of_momenta; i++)
    {
      std::string temp_mesh_name=Mesh_tree_names[i];
      std::map<std::string, NemoMesh*>::const_iterator c_it2;
      c_it2 = Momentum_meshes.find(temp_mesh_name);
      NEMO_ASSERT(c_it2!=Momentum_meshes.end(),prefix+"have not found the NemoMesh\n");
      NemoMesh* temp_mesh=c_it2->second;
      all_momentum_points[i]=temp_mesh->get_mesh_points();
    }
    //construct the momentum tupel and store it in all_momentum_points
    std::vector<std::vector<NemoMeshPoint*>::const_iterator> const_iterator_vector(number_of_momenta);
    for(unsigned int i=0; i<number_of_momenta; i++)
    {
      NEMO_ASSERT(all_momentum_points[i].size()>0,prefix+"found empty branch of momenta\n");
      const_iterator_vector[i]=all_momentum_points[i].begin();
    }

    //new version:
    //find the topmost Mesh
    all_momenta.clear();
    local_momenta.clear();
    NemoMesh* topmost_mesh=NULL;
    std::map<NemoMesh*,std::vector<NemoMesh*> >::const_iterator Mesh_it=Mesh_tree_topdown.begin();
    for(; Mesh_it!=Mesh_tree_topdown.end(); ++Mesh_it)
    {
      if(Mesh_it->first->get_name()==Mesh_tree_names[0])
      {
        NEMO_ASSERT(topmost_mesh==NULL,prefix+"found more than one topmost mesh\n");
        topmost_mesh=Mesh_it->first;
      }
    }
    std::set<std::vector<NemoMeshPoint> > resulting_points;
    std::map<std::string,double> conditions;
    std::vector<double> temp_vector(3,0.0);
    NemoMeshPoint temp_momentum(0,temp_vector);
    const std::vector<NemoMeshPoint> fake_momentum_point(number_of_momenta,temp_momentum);

    find_specific_momenta(conditions,fake_momentum_point,resulting_points);

    std::set<std::vector<NemoMeshPoint> >::const_iterator momentum_c_it=resulting_points.begin();
    ofstream outfile;
    std::string filename=get_name()+"all_points.dat";
    if(print_all_mesh_points&&!no_file_output)
      outfile.open(filename.c_str());
    for(; momentum_c_it!=resulting_points.end(); ++momentum_c_it)
    {
      //for debug output: print all points into file and on screen
      if(print_all_mesh_points&&!no_file_output)
      {
        for(unsigned int j=0; j<(*momentum_c_it).size(); j++)
        {
          const std::vector<double>& coords=(*momentum_c_it)[j].get_coords();
          outfile<<"("<<coords[0];
          for (unsigned int i=1; i<coords.size(); i++)
            outfile<<", "<<coords[i];
          outfile<<")"<<std::endl;
        }
        msg<<prefix<<"considered momentum tupel: ";
        for(unsigned int j=0; j<(*momentum_c_it).size(); j++)
          (*momentum_c_it)[j].print();
      }

      all_momenta.insert(*momentum_c_it);
    }
    if(print_all_mesh_points&&!no_file_output)
      outfile.close();

    //this is a very similar block as above, but it produces the local points only!
    {

      //find the topmost Mesh
      NemoMesh* topmost_mesh=NULL;
      std::map<NemoMesh*,std::vector<NemoMesh*> >::const_iterator Mesh_it=Mesh_tree_topdown.begin();
      for(; Mesh_it!=Mesh_tree_topdown.end(); ++Mesh_it)
      {
        if(Mesh_it->first->get_name()==Mesh_tree_names[0])
        {
          NEMO_ASSERT(topmost_mesh==NULL,prefix+"found more than one topmost mesh\n");
          topmost_mesh=Mesh_it->first;
        }
      }
      std::set<std::vector<NemoMeshPoint> > resulting_points;
      //this vector keeps track of the index configuration of the respective branch
      std::vector<unsigned int> tree_index(number_of_momenta,0);
      //fill vector<NemoMeshPoint> of topmost mesh
      std::vector<int> my_point_indices = topmost_mesh->get_my_points();
      NemoMeshPoint dummy_point=topmost_mesh->get_point(std::abs(my_point_indices[0]));
      std::vector<NemoMeshPoint> topmost_points=std::vector<NemoMeshPoint>(my_point_indices.size(),dummy_point);
      for(unsigned int i=0; i<my_point_indices.size(); i++)
        topmost_points[i]=topmost_mesh->get_point(std::abs(my_point_indices[i]));

      //iterate until end of topmost points is reached
      //store the resulting "leafs" of the tree in resulting_points
      //for(;const_iterator_vector[0]!=topmost_points.end();)
      std::vector<NemoMesh*> parents(number_of_momenta,topmost_mesh);
      NemoMesh* child;
      std::vector<NemoMeshPoint> resulting_momentum_tupel(number_of_momenta,topmost_points[tree_index[0]]); //temporary storage for this specific "leaf"

      ////identify the appropiate child of this tree branch
      std::map<NemoMesh*,std::vector<NemoMesh*> >::const_iterator temp_c_it; //=Mesh_tree_topdown.find(parent);
      child=topmost_mesh;
      //for(unsigned int dimension_i=1;dimension_i<number_of_momenta;dimension_i++)
      for(unsigned int dimension_i=0; dimension_i<number_of_momenta;)
      {
        parents[dimension_i]=child;

        my_point_indices = parents[dimension_i]->get_my_points();

        //if the index is within the local branch, store the momentum in the tupel and go to the next branch
        if(tree_index[dimension_i]<my_point_indices.size())
        {
          resulting_momentum_tupel[dimension_i]=parents[dimension_i]->get_point(std::abs(my_point_indices[tree_index[dimension_i]]));
          if(dimension_i==number_of_momenta-1)
          {
            tree_index[dimension_i]++;
            resulting_points.insert(resulting_momentum_tupel);

          }
          else
          {
            //identify the next child of this tree branch
            temp_c_it=Mesh_tree_topdown.find(parents[dimension_i]);
            NEMO_ASSERT(temp_c_it!=Mesh_tree_topdown.end(),prefix+"have not found mesh \""+parents[dimension_i]->get_name()+"\"\n");
            child=temp_c_it->second[std::abs(my_point_indices[tree_index[dimension_i]])];//here: child is used as temporary storage for parents...
            dimension_i++;
          }
        }
        //if the index is beyond the local branch, increment the index of the previous branch and set this index=0
        else
        {
          //is there a previous branch, then do as stated above
          if(dimension_i>0)
          {
            tree_index[dimension_i]=0;
            --dimension_i;
            tree_index[dimension_i]++;
            child=parents[dimension_i];
          }
          //if not, then the end of the upper most branch is reached
          else
            break;
        }
      }
      std::set<std::vector<NemoMeshPoint> >::const_iterator momentum_c_it=resulting_points.begin();
      ofstream outfile;
      std::string filename=get_name()+"all_local_points.dat";
      if(print_all_mesh_points&&!no_file_output)
        outfile.open(filename.c_str());
      for(; momentum_c_it!=resulting_points.end(); ++momentum_c_it)
      {
        //for debug output: print all local points into file and on screen
        if(print_all_mesh_points&&!no_file_output)
        {
          for(unsigned int j=0; j<(*momentum_c_it).size(); j++)
          {
            const std::vector<double>& coords=(*momentum_c_it)[j].get_coords();
            outfile<<"("<<coords[0];
            for (unsigned int i=1; i<coords.size(); i++)
              outfile<<", "<<coords[i];
            outfile<<")"<<std::endl;
          }
          msg<<prefix<<"considered local momentum tupel: ";
          for(unsigned int j=0; j<(*momentum_c_it).size(); j++)
            (*momentum_c_it)[j].print();
        }

        local_momenta.insert(*momentum_c_it);
      }
      if(print_all_mesh_points&&!no_file_output)
        outfile.close();
    }
  }
  else
  {
    Parallelizer->get_data("all_momenta",pointer_to_all_momenta);
    Parallelizer->get_data("local_momenta",pointer_to_local_momenta);
  }
  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::find_specific_momenta(const std::map<std::string,double>& conditions, const std::vector<NemoMeshPoint>& momentum_point,
                                        std::set<std::vector<NemoMeshPoint> >& resulting_points, bool absorption) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::find_specific_momenta ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::find_specific_momenta: ";
  resulting_points.clear();
  NemoMesh* topmost_mesh=NULL;
  std::map<NemoMesh*,std::vector<NemoMesh*> >::const_iterator Mesh_it=Mesh_tree_topdown.begin();
  for(; Mesh_it!=Mesh_tree_topdown.end(); ++Mesh_it)
  {
    if(Mesh_it->first->get_name()==Mesh_tree_names[0])
    {
      NEMO_ASSERT(topmost_mesh==NULL,prefix+"found more than one topmost mesh\n");
      topmost_mesh=Mesh_it->first;
    }
  }
  //this vector keeps track of the index configuration of the respective branch
  std::vector<unsigned int> tree_index(number_of_momenta,0);
  //fill vector<NemoMeshPoint> of topmost mesh
  std::vector<int> my_point_indices = topmost_mesh->get_my_points(); //replace with get_mesh_points
  std::vector<NemoMeshPoint*> temp_points=topmost_mesh->get_mesh_points();

  NemoMeshPoint dummy_point=topmost_mesh->get_point(std::abs(my_point_indices[0]));
  std::vector<NemoMeshPoint> topmost_points=std::vector<NemoMeshPoint>(temp_points.size(),*(temp_points[0]));
  for(unsigned int i=0; i<topmost_points.size(); i++)
    topmost_points[i]=*(temp_points[i]);

  //iterate until end of topmost points is reached
  //store the resulting "leafs" of the tree in resulting_points
  NEMO_ASSERT(number_of_momenta>0,prefix+"no momenta found\n");
  std::vector<NemoMesh*> parents(number_of_momenta,topmost_mesh);
  NemoMesh* child;
  std::vector<NemoMeshPoint> resulting_momentum_tupel(number_of_momenta,topmost_points[tree_index[0]]); //temporary storage for this specific "leaf"

  //const Propagator* temp_propagator=writeable_Propagator; //s.begin()->second;
  //double energy1=0;

  //identify the appropiate child of this tree branch
  std::map<NemoMesh*,std::vector<NemoMesh*> >::const_iterator temp_c_it; //=Mesh_tree_topdown.find(parent);
  child=topmost_mesh;
  for(unsigned int dimension_i=0; dimension_i<number_of_momenta;)
  {
    parents[dimension_i]=child;

    std::vector<NemoMeshPoint*> child_temp_points=parents[dimension_i]->get_mesh_points();
    std::map<unsigned int, NemoMeshPoint*> relevant_child_temp_points_map;
    std::vector<NemoMeshPoint*> relevant_child_temp_points(child_temp_points);

    //filter out those points of child_temp_points that are not needed (if a condition applies)
    std::string possible_condition_name=parents[dimension_i]->get_name()+"_interval";
    std::map<std::string,double>::const_iterator it=conditions.find(possible_condition_name);
    if(it!=conditions.end())
    {
      //this is the interval condition section
      double interval = it->second;
      possible_condition_name=parents[dimension_i]->get_name()+"_value";
      it=conditions.find(possible_condition_name);
      NEMO_ASSERT(it==conditions.end(),prefix+"found conditions \""+parents[dimension_i]->get_name()+"_value\""+
                  " and \""+parents[dimension_i]->get_name()+"_interval\", but they exclude each other\n");
      //find the "origin" (the value in the momentum_point[dimension_i]) from which the interval is measured
      NemoMeshPoint origin(momentum_point[dimension_i]);
      std::vector<double> origin_coords=origin.get_coords();
      for(unsigned int i=0; i<child_temp_points.size(); i++)
      {
        std::vector<double> temp_coords=child_temp_points[i]->get_coords();
        NEMO_ASSERT(temp_coords.size()==origin_coords.size(),prefix+"inconsistent dimension of points\n");
        double sqrt_distance=0.0;
        double distance=0.0;
        for(unsigned int ii=0; ii<temp_coords.size(); ii++)
        {
          distance+=(temp_coords[ii]-origin_coords[ii])*(temp_coords[ii]-origin_coords[ii]);
        }

        sqrt_distance=std::sqrt(distance);
        //add those points to relevant_child_temp_points that are within the interval
        if(sqrt_distance<=interval)
          relevant_child_temp_points_map[i]=child_temp_points[i];
      }
      //translate the map into a vector for usage below (for safety, use the same order as originally done by the mesh
      relevant_child_temp_points.clear();
      relevant_child_temp_points.resize(relevant_child_temp_points_map.size(),relevant_child_temp_points_map.begin()->second);
      std::map<unsigned int, NemoMeshPoint*>::iterator relevant_it=relevant_child_temp_points_map.begin();
      unsigned int counter=0;
      for(; relevant_it!=relevant_child_temp_points_map.end(); ++relevant_it)
      {
        relevant_child_temp_points[counter]=relevant_it->second;
        counter++;
      }
    }
    else if(conditions.size()>0)
    {
      possible_condition_name=parents[dimension_i]->get_name()+"_value";
      it=conditions.find(possible_condition_name);
      if(it!=conditions.end())
      {
        double target_value;
        if((absorption == true))
        {

          //this is the value condition section
          //find the NemoMeshPoint that fits best to the target_value
          // double target_value=it->second;
          //translate the NemoMeshPoints into a map of double and NemoMeshPoints to search for the mesh point agreeing best with the target value
          std::map<double,NemoMeshPoint*> temp_point_x_map;
          for(unsigned int i=0; i<child_temp_points.size(); i++)
            temp_point_x_map[child_temp_points[i]->get_x()]=child_temp_points[i];
          relevant_child_temp_points.clear();

          for(unsigned int jj=0; jj<child_temp_points.size(); jj++)
          {
            target_value = child_temp_points[jj]->get_x() - it->second;
            std::map<double,NemoMeshPoint*>::const_iterator temp_x_map_cit_upper=temp_point_x_map.upper_bound(target_value);
            std::map<double,NemoMeshPoint*>::const_iterator best_x_map_cit=temp_x_map_cit_upper;
            if(best_x_map_cit!=temp_point_x_map.begin())
              best_x_map_cit--;
            if(std::abs(best_x_map_cit->first-target_value)>std::abs(temp_x_map_cit_upper->first-target_value))
              best_x_map_cit=temp_x_map_cit_upper;
            if(best_x_map_cit->first==momentum_point[dimension_i].get_x())
            {
              relevant_child_temp_points.push_back(child_temp_points[jj]);
            }
          }

        }
        else
        {
          target_value = it->second;
          //this is the value condition section
          //find the NemoMeshPoint that fits best to the target_value
          // double target_value=it->second;
          //translate the NemoMeshPoints into a map of double and NemoMeshPoints to search for the mesh point agreeing best with the target value
          std::map<double,NemoMeshPoint*> temp_point_x_map;
          for(unsigned int i=0; i<child_temp_points.size(); i++)
            temp_point_x_map[child_temp_points[i]->get_x()]=child_temp_points[i];

          std::map<double,NemoMeshPoint*>::const_iterator temp_x_map_cit_upper=temp_point_x_map.upper_bound(target_value);
          std::map<double,NemoMeshPoint*>::const_iterator best_x_map_cit=temp_x_map_cit_upper;
          if(best_x_map_cit!=temp_point_x_map.begin())
            best_x_map_cit--;
          if(std::abs(best_x_map_cit->first-target_value)>std::abs(temp_x_map_cit_upper->first-target_value))
            best_x_map_cit=temp_x_map_cit_upper;
          relevant_child_temp_points.clear();
          relevant_child_temp_points.resize(1,best_x_map_cit->second);
        }

      }
    }


    //if the index is within the local branch, store the momentum in the tupel and go to the next branch
    if(tree_index[dimension_i]<relevant_child_temp_points.size())
    {
      resulting_momentum_tupel[dimension_i]=*(relevant_child_temp_points[tree_index[dimension_i]]);
      if(dimension_i==number_of_momenta-1)
      {
        tree_index[dimension_i]++;
        resulting_points.insert(resulting_momentum_tupel);
      }
      else
      {
        //identify the next child of this tree branch
        temp_c_it=Mesh_tree_topdown.find(parents[dimension_i]);
        NEMO_ASSERT(temp_c_it!=Mesh_tree_topdown.end(),prefix+"have not found mesh \""+parents[dimension_i]->get_name()+"\"\n");
        child=temp_c_it->second[tree_index[dimension_i]];//here: child is used as temporary storage for parents...
        dimension_i++;
      }
    }
    //if the index is beyond the local branch, increment the index of the previous branch and set this index=0
    else
    {
      //is there a previous branch, then do as stated above
      if(dimension_i>0)
      {
        tree_index[dimension_i]=0;
        --dimension_i;
        tree_index[dimension_i]++;
        child=parents[dimension_i];
      }
      //if not, then the end of the upper most branch is reached
      else
        break;
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}



void Propagation::update_global_job_list(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::update_global_job_list");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::update_global_job_list: ";
  NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"called before \"all_momenta\" are set\n");
  //1.loop over all_momenta
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
  std::vector<int> all_momenta_ranks(pointer_to_all_momenta->size(),0);
  NemoMesh* temp_mesh=Mesh_tree_topdown.begin()->first;
  NEMO_ASSERT(temp_mesh!=NULL,prefix+"have received NULL for the mesh pointer\n");
  unsigned int counter=0;
  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    //2. if momentum is local, add my_rank to result vector
    //2.1 use first Propagator to determine whether *c_it is local or not
    const Propagator* temp_Propagator=*(known_Propagators.begin());
    if(!is_Propagator_initialized(temp_Propagator->get_name()))
      initialize_Propagators(temp_Propagator->get_name());
    Propagator::PropagatorMap::const_iterator momentum_c_it=temp_Propagator->propagator_map.find(*c_it);

    if(momentum_c_it!=temp_Propagator->propagator_map.end())
    {
      //this momentum is local
      int my_local_rank;
      MPI_Comm_rank(temp_mesh->get_global_comm(), &my_local_rank);
      all_momenta_ranks[counter]=my_local_rank;
    }
    //else for nonlocal momenta keep 0 in the result vector
    counter++;
  }
  //3. MPI Sum all results up and distribute to all CPUs
  //MPI_Barrier(holder.one_partition_total_communicator);
  MPI_Barrier(temp_mesh->get_global_comm());
  //MPI_Allreduce(MPI_IN_PLACE,&(all_momenta_ranks[0]),all_momenta_ranks.size(),MPI_INT,MPI_SUM,holder.one_partition_total_communicator);
  MPI_Allreduce(MPI_IN_PLACE,&(all_momenta_ranks[0]),all_momenta_ranks.size(),MPI_INT,MPI_SUM,temp_mesh->get_global_comm());

  //4. store result in Propagation::global_job_list
  counter=0;
  global_job_list.clear();
  c_it=pointer_to_all_momenta->begin();
  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    std::vector<NemoMeshPoint> temp_point(*c_it);
    global_job_list[temp_point]=all_momenta_ranks[counter];
    counter++;
  }

  //debugging: global_job_list-output - if the rank of real space parallelization is 0
  int real_space_rank;
  MPI_Comm_rank(get_simulation_domain()->get_communicator(),&real_space_rank);
  if(debug_output_job_list&&!no_file_output) //&&real_space_rank==0)
  {
    int size;
    MPI_Comm_size(get_simulation_communicator(),&size);
    //NOTE: merge communicator with energy_mesh->get_global_comm()
    std::map<int,int> load_per_rank;
    int myrank;
    NemoMesh* temp_mesh=Mesh_tree_topdown.begin()->first;
    NEMO_ASSERT(temp_mesh!=NULL,prefix+"have received NULL for the mesh pointer\n");
    MPI_Comm_rank(temp_mesh->get_global_comm(),&myrank);
    std::map<std::vector<NemoMeshPoint>, int>::const_iterator temp_cit=global_job_list.begin();
    std::stringstream convert_to_string;
    convert_to_string << myrank <<"_real_space_"<<real_space_rank;
    std::string filename=get_name()+"_job_list_"+convert_to_string.str()+".dat";

    ofstream outfile;
    outfile.open(filename.c_str());
    char MPI_name[BUFSIZ];
    int MPI_name_length=0;
    MPI_Get_processor_name(MPI_name,&MPI_name_length);
    outfile<<MPI_name<<"\n";
    for(; temp_cit!=global_job_list.end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer,data_string,false);
      data_string +="\t";
      std::stringstream convert2;
      convert2<<temp_cit->second;
      data_string +=convert2.str();
      outfile<<data_string<<"\n";

      load_per_rank[temp_cit->second]+=1;
    }
    outfile<<"\n\n";
    //add load per rank
    std::map<int,int>::const_iterator c_it_job=load_per_rank.begin();
    for(; c_it_job!=load_per_rank.end(); ++c_it_job)
      outfile<<"rank: "<<c_it_job->first<<"\thas load of "<<c_it_job->second<<"\n";

    outfile.close();
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::set_global_job_list()
{
  std::string error_prefix = "Propagation(\""+get_name()+"\")::set_global_gob_list ";
  const std::string combine_propagator_name = options.get_option("store_result_in",std::string(""));
  combine_solver_for_storage = dynamic_cast<Propagation*> (find_source_of_data(combine_propagator_name));
  NEMO_ASSERT(combine_solver_for_storage!=NULL,error_prefix+"no combine solver set\n");
  Propagation* combine_solver = dynamic_cast<Propagation*> (combine_solver_for_storage);
  NEMO_ASSERT(combine_solver!=NULL,error_prefix+"dynamic cast of the solver that solves \""+combine_solver_for_storage->get_name()+"\"into a Propagation failed\n");

  std::map<std::vector<NemoMeshPoint>, int > momentum_map;
  combine_solver->get_data("global_job_list",momentum_map);
  global_job_list =  momentum_map ;

}



void Propagation::get_data(const std::string& variable, std::map<std::vector<NemoMeshPoint>, int >& momentum_map)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data global_job_list: ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix="Propagation(\""+this->get_name()+"\")::get_data global_job_list: ";
  if(variable == "global_job_list")
  {
    if(!Propagation_is_initialized)
      initialize_Propagation();
    if(global_job_list.empty())
      update_global_job_list();
    NEMO_ASSERT(global_job_list.size()>0, prefix + ": global_job_list has no entries.");
    momentum_map = global_job_list;
  }
  else if (variable == "update_global_job_list")
  {
    global_job_list.clear();
    update_global_job_list();
    NEMO_ASSERT(global_job_list.size()>0, prefix + ": global_job_list has no entries.");
    momentum_map = global_job_list;
  }

  else
    throw std::runtime_error(tic_toc_prefix+"variable is not defined for this get_data\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::set_momentum_meshes(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::set_momentum_meshes");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));

  std::string prefix="Propagation(\""+this->get_name()+"\")::set_momentum_meshes: ";

  // -----------------------------------
  // define the possible momenta of the Propagator entities
  // set parallelization first
  // then create NemoMeshes, such as k_space and E_space
  // or get set the NemoMesh to NULL and get the meshes in do_solve from another simulation
  // -----------------------------------
  std::vector<std::string> temp_momentum_names;
  if (!options.check_option("momentum_names")) throw std::invalid_argument(prefix+"define momentum_names, i.e. the mesh types\n");
  options.get_option("momentum_names",temp_momentum_names);
  for(unsigned int i=0; i<temp_momentum_names.size(); i++)
    msg<<temp_momentum_names[i]<<std::endl;

  number_of_momenta=(temp_momentum_names.size()>1) ? temp_momentum_names.size() : 1;

  analytical_momentum_integration.resize(number_of_momenta,false);
  // no dependencies allowed so far
  // for later dependence: dependence hierarchy agrees with the parallelization hierarchy!
  momentum_dependencies.resize(number_of_momenta);


  //find the mesh constructors
  for(unsigned int i=0; i<temp_momentum_names.size(); i++)
  {
    std::string variable_name;
    variable_name=temp_momentum_names[i]+"_constructor";
    std::string constructor_name;
    if(options.check_option(variable_name))
      constructor_name=options.get_option(variable_name,std::string(""));
    else
      throw std::invalid_argument(prefix+"define mesh_constructor "+variable_name+"\n");

    Mesh_Constructors[temp_momentum_names[i]]=this->find_simulation(constructor_name);
    NEMO_ASSERT(this->find_simulation(constructor_name)!=NULL, prefix+"have not found simulation \""+constructor_name+"\"\n");
    if (Mesh_Constructors.find(temp_momentum_names[i])->second == NULL)
      throw std::invalid_argument(std::string(prefix+"unknown mesh constructor for mesh \""+temp_momentum_names[i]+"\": "+constructor_name+"\n"));
  }

  for (unsigned int i=0; i<number_of_momenta; i++)
  {
    // check that each momentum name appears only once:
    NEMO_ASSERT(momentum_mesh_types.find(temp_momentum_names[i])==momentum_mesh_types.end(),
                prefix+"found momentum name \""+temp_momentum_names[i]+"\" twice\n");

    //read in number of mesh points
    unsigned int momentum_points=0;
    if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
    {
      std::string variable_name=temp_momentum_names[i]+"_points";
      if(options.check_option(variable_name))
        momentum_points=options.get_option(variable_name,10);
      else
        throw std::invalid_argument(prefix+"define " + variable_name + "\n");
    }

    if(temp_momentum_names[i].find("momentum_1D")!=std::string::npos)
    {
      momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Momentum_1D;
      Momentum_meshes[temp_momentum_names[i]]=NULL;
      if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
      {
        Momentum_meshes[temp_momentum_names[i]]=new Kspace;
        Momentum_meshes[temp_momentum_names[i]]->set_name(temp_momentum_names[i]);
        Momentum_meshes[temp_momentum_names[i]]->set_num_points(momentum_points);
      }
    }
    else if (temp_momentum_names[i].find("momentum_2D")!=std::string::npos)
    {
      momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Momentum_2D;
      Momentum_meshes[temp_momentum_names[i]]=NULL;
      if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
      {
        Momentum_meshes[temp_momentum_names[i]]=new Kspace;
        Momentum_meshes[temp_momentum_names[i]]->set_name(temp_momentum_names[i]);
        Momentum_meshes[temp_momentum_names[i]]->set_num_points(momentum_points);
      }
    }
    else if (temp_momentum_names[i].find("momentum_3D")!=std::string::npos)
    {
      momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Momentum_3D;
      Momentum_meshes[temp_momentum_names[i]]=NULL;
      if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
      {
        Momentum_meshes[temp_momentum_names[i]]=new Kspace;
        Momentum_meshes[temp_momentum_names[i]]->set_name(temp_momentum_names[i]);
        Momentum_meshes[temp_momentum_names[i]]->set_num_points(momentum_points);
      }
    }
    else if (temp_momentum_names[i].find("energy")!=std::string::npos)
    {
      if(temp_momentum_names[i].find("complex_energy")==std::string::npos)
        momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Energy;
      else
        momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Complex_energy;
      Momentum_meshes[temp_momentum_names[i]]=NULL;
      if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
      {
        Momentum_meshes[temp_momentum_names[i]]=new Espace;
        Momentum_meshes[temp_momentum_names[i]]->set_name(temp_momentum_names[i]);
        Momentum_meshes[temp_momentum_names[i]]->set_num_points(momentum_points);
      }
    }
    else if (temp_momentum_names[i].find("spin")!=std::string::npos)
    {
      momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Spin;
      Momentum_meshes[temp_momentum_names[i]]=NULL;
      if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
      {
        Momentum_meshes[temp_momentum_names[i]]=new NemoMesh;
        Momentum_meshes[temp_momentum_names[i]]->set_name(temp_momentum_names[i]);
        Momentum_meshes[temp_momentum_names[i]]->set_num_points(momentum_points);
      }
    }
    else if (temp_momentum_names[i].find("valley")!=std::string::npos)
    {
      momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Valley;
      Momentum_meshes[temp_momentum_names[i]]=NULL;
      if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
      {
        Momentum_meshes[temp_momentum_names[i]]=new Kspace;
        Momentum_meshes[temp_momentum_names[i]]->set_name(temp_momentum_names[i]);
        Momentum_meshes[temp_momentum_names[i]]->set_num_points(momentum_points);
      }
    }
    else if (temp_momentum_names[i].find("angular_momentum")!=std::string::npos)
    {
      momentum_mesh_types[temp_momentum_names[i]]=NemoPhys::Angular_Momentum;
      Momentum_meshes[temp_momentum_names[i]]=NULL;
      if(Mesh_Constructors.find(temp_momentum_names[i])->second == this)
      {
        Momentum_meshes[temp_momentum_names[i]]=new Kspace;
        Momentum_meshes[temp_momentum_names[i]]->set_name(temp_momentum_names[i]);
        Momentum_meshes[temp_momentum_names[i]]->set_num_points(momentum_points);
      }
    }
    else
      throw std::invalid_argument(prefix+"unknown momentum_type\n");

    if (Mesh_Constructors.find(temp_momentum_names[i])->second==this)
    {
      // set the number of mesh points for momentum[i]
      unsigned int ui_temp;
      std::string variable_name=temp_momentum_names[i] + std::string("_points");
      if(options.check_option(variable_name))
        ui_temp = options.get_option(variable_name,10);
      else
        throw std::invalid_argument(std::string(prefix+"define the number of mesh points for "+temp_momentum_names[i]+" (i.e. " + variable_name+ ")\n"));
      Momentum_meshes.find(temp_momentum_names[i])->second->set_num_points(ui_temp);
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::do_solve()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve1 ");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));
  std::string prefix="Propagation(\""+this->get_name()+"\")::do_solve: ";
  std::map<std::string,Simulation*>::const_iterator temp_cit_debug=pointer_to_Propagator_Constructors->begin();
  for(; temp_cit_debug!=pointer_to_Propagator_Constructors->end(); ++temp_cit_debug)
    msg<<prefix<<temp_cit_debug->first<<" "<<temp_cit_debug->second->get_name()<<"\n";

  //check that the Propagation is initialized...
  if(!Propagation_is_initialized)
    initialize_Propagation(get_name());

  //std::map<std::string,Propagator*>::iterator it=writeable_Propagators.begin();
  //for(; it!=writeable_Propagators.end(); ++it)
  if(writeable_Propagator!=NULL)
  {
    msg<<prefix<<"solving propagator \""<<name_of_writeable_Propagator<<"\"\n";
    //std::string temp_name = name_of_writeable_Propagator;

    //check that the Propagator is initialized
    if(!is_Propagator_initialized(name_of_writeable_Propagator))
    {
      ////find the constructor and let him initialize the Propagator
      //std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator);
      //NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),prefix+"Constructor of propagator \""+name_of_writeable_Propagator+"\" not found\n");
      //Propagation* temp_propagation= dynamic_cast<Propagation*> (temp_it->second);
      //NEMO_ASSERT(temp_propagation!=NULL,prefix+"dynamic cast failed\n");
      //temp_propagation->
      initialize_Propagators(name_of_writeable_Propagator);
      msg<<prefix<<"Constructor \""<<get_name()<<"\" initialized \""<<name_of_writeable_Propagator<<"\n";
      NEMO_ASSERT(is_Propagator_initialized(name_of_writeable_Propagator),
                  prefix+"Propagator \""+name_of_writeable_Propagator+"\" is not initialized after initialize_Propagators is called\n");
      //temp_propagation->
      get_data(name_of_writeable_Propagator,writeable_Propagator);
    }

    //check that the Propagator is not a scattering self-energy (which requires a special solution)
    if(name_of_writeable_Propagator.find(std::string("scattering"))!=std::string::npos && name_of_writeable_Propagator.find(std::string("self"))!=std::string::npos)
    {
      std::string scattering_type;
      if(options.check_option("scattering_type"))
        scattering_type = options.get_option("scattering_type",std::string(""));
      else
        throw std::invalid_argument(prefix+"please define \"scattering_type\"\n");

      if(scattering_type=="lambda_G_scattering")
      {
        //here, the scattering self-energy is directly proportional to the Green's function - no MPI communication required...
        Propagator::PropagatorMap::iterator momentum_it=writeable_Propagator->propagator_map.begin();
        for(; momentum_it!=writeable_Propagator->propagator_map.end(); momentum_it++)
          if(!is_ready(name_of_writeable_Propagator,momentum_it->first)) //do_solve only when really required...
            do_solve(writeable_Propagator,momentum_it->first);
          else
            msg<<prefix<<"job already done\n";
      }
      else
      {
        //do the MPI-special treatment...
        do_full_MPI_solve(writeable_Propagator);
      }
    }
    else //not a scattering self-energy
    {
      Propagator::PropagatorMap::iterator momentum_it=writeable_Propagator->propagator_map.begin();
      NEMO_ASSERT(writeable_Propagator->propagator_map.size()>0,prefix+"empty propagator_map found\n");
      //msg<<prefix<<"size of propagator map: "<<it->second->propagator_map.size()<<"\n";
      for(; momentum_it!=writeable_Propagator->propagator_map.end(); momentum_it++)
        if(!is_ready(name_of_writeable_Propagator,momentum_it->first)) //do_solve only when really required...
          do_solve(writeable_Propagator,momentum_it->first);
        else
          msg<<prefix<<"job already done\n";
    }
  }
  if(options.check_option("density_of"))
  {
    calculate_density();
    if(options.get_option("calculate_Jacobian",false))
      calculate_density_Jacobian();
  }
  //test_debug();
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::do_full_MPI_solve(Propagator*& result)
{
  std::string prefix="Propagation(\""+this->get_name()+"\")::do_full_MPI_solve";
  throw std::invalid_argument(std::string(prefix+"(Propagator*&) solver for \""+result->get_name()+"\" not implemented, yet.\n"));
}

void Propagation::do_solve(Propagator*& result, const std::vector<NemoMeshPoint>& input_momentum)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve3");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));
  activate_regions();
  std::string error_prefix = "Propagation(\""+get_name()+"\")::do_solve ";
  //msg<<error_prefix<< "solving momentum:\n";
  //for(unsigned int i=0; i<input_momentum.size(); i++)
  //  input_momentum[i].print();

  const std::string Propagator_name = result->get_name();

  NemoPhys::Propagator_type p_type=get_Propagator_type(Propagator_name);
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=result->propagator_map.find(input_momentum);
  if(matrix_it!=result->propagator_map.end())
    temp_matrix=matrix_it->second;

  //check the type of this Propagator and call the appropriate version of do_solve_...
  //if(Propagator_name.find(std::string("combined"))!=std::string::npos)
  if(this_is_combine_Propagation)
    do_solve_combine(result, input_momentum,
                     temp_matrix); //this is a kind of neutral case - solve is valid for all kinds of Propagators (required for storage purposes)...
  else if(p_type==NemoPhys::Inverse_Green)
    do_solve_inverse(result, input_momentum, temp_matrix);
  else if(p_type==NemoPhys::Fermion_retarded_Green || p_type==NemoPhys::Boson_retarded_Green || p_type==NemoPhys::Fermion_retarded_self ||
          p_type==NemoPhys::Boson_retarded_self)
    do_solve_retarded(result,input_momentum, temp_matrix);
  else if(p_type==NemoPhys::Fermion_advanced_Green || p_type==NemoPhys::Boson_advanced_Green || p_type==NemoPhys::Fermion_advanced_self ||
          p_type==NemoPhys::Boson_advanced_self)
    do_solve_advanced(result,input_momentum,temp_matrix);
  else if(p_type==NemoPhys::Fermion_lesser_Green || p_type==NemoPhys::Boson_lesser_Green || p_type==NemoPhys::Fermion_lesser_self ||
          p_type==NemoPhys::Boson_lesser_self)
    do_solve_lesser(result,input_momentum,temp_matrix);
  else if(p_type==NemoPhys::Fermion_lesser_HGreen || p_type==NemoPhys::Boson_lesser_HGreen)
    do_solve_H_lesser(result,input_momentum,temp_matrix);
  else if(p_type==NemoPhys::Fermion_greater_Green || p_type==NemoPhys::Boson_greater_Green || p_type==NemoPhys::Fermion_greater_self ||
          p_type==NemoPhys::Boson_greater_self)
    do_solve_greater(result,input_momentum,temp_matrix);
  else if(p_type==NemoPhys::Fermion_spectral_Green ||p_type==NemoPhys::Boson_spectral_Green ||p_type==NemoPhys::Fermion_spectral_self ||
          p_type==NemoPhys::Boson_spectral_self)
    do_solve_spectral(result,input_momentum,temp_matrix);
  else
    throw std::runtime_error(error_prefix+"for this function type is not implemented, yet\n");

  NEMO_ASSERT(is_ready(Propagator_name,input_momentum),error_prefix+"still not ready\n");
  write_propagator(Propagator_name,input_momentum, temp_matrix);
  conclude_after_do_solve(result,input_momentum);
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::conclude_after_do_solve(Propagator*& result, const std::vector<NemoMeshPoint>& input_momentum)
{
  std::string error_prefix = "Propagation(\""+get_name()+"\")::conclude_after_do_solve ";
  std::string Propagator_name=result->get_name();
  NEMO_ASSERT(is_ready(Propagator_name,input_momentum),error_prefix+"still not ready\n");

  if(do_outputL)
    print_Propagator(result,&input_momentum);
  if(clean_up)
    delete_propagator_matrices(result,&input_momentum);

  //make other solvers free memory when their results are no longer required - controlled via inputdeck option "free_memory_of"
  std::vector<std::string> list_of_propagators_to_delete;
  options.get_option(std::string("free_memory_of"),list_of_propagators_to_delete);
  for(unsigned int i=0; i<list_of_propagators_to_delete.size(); i++)
  {
    std::string propagator_name = list_of_propagators_to_delete[i];
    //find the constructor of propagator_name
    //std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(propagator_name);
    Simulation* delete_solver = find_source_of_data(propagator_name);
    NEMO_ASSERT(delete_solver!=NULL,error_prefix+"have not found the solver of \""+propagator_name+"\"\n");

    Propagation* temp_solver=dynamic_cast<Propagation*>(delete_solver);
    if(temp_solver!=NULL)
    {
      Propagator* temp_propagator;
      temp_solver->get_data(propagator_name,temp_propagator);
      //check if the full matrix or only a specific portion is to be deleted
      std::string dofmap_solver=options.get_option("free_DOFmap_solver",std::string(""));
      if(/*propagator_name.find("combine")!=std::string::npos &&*/ dofmap_solver.find("None")==std::string::npos)
      {
        Domain* row_domain=NULL;
        if(options.check_option("free_row_domain"))
        {
          std::string domain_name=options.get_option("free_row_domain",std::string(""));
          row_domain=Domain::get_domain(domain_name);
        }
        std::string variable_name = "free_";
        const DOFmapInterface& temp_DOFmap=get_special_const_dof_map(variable_name,row_domain);
        Domain* col_domain=NULL;
        if(options.check_option("free_column_domain"))
        {
          std::string domain_name=options.get_option("free_column_domain",std::string(""));
          col_domain=Domain::get_domain(domain_name);
        }
        std::string variable_name_col = "free_col_";
        const DOFmapInterface& temp_DOFmap_col=get_special_const_dof_map(variable_name_col,col_domain);
        temp_solver->delete_propagator_matrices(temp_propagator,&input_momentum,&temp_DOFmap,&temp_DOFmap_col);
      }
      else
        temp_solver->delete_propagator_matrices(temp_propagator,&input_momentum);
    }

    if (delete_solver->get_type().find("Transformation") == std::string::npos)// && temp_solver->get_type().find("QTBM") != std::string::npos)
    {
      //if so, the "results" should remain available, i.e. the job is still done for this momentum
      delete_solver->set_job_done_momentum_map(&propagator_name,&input_momentum,true);
    }
  }
  //if Propagator is not in the list_of_propagators_to_delete, make sure that it remains ready
  bool let_Propagator_be_ready = true;
  for(unsigned int i=0; i<list_of_propagators_to_delete.size() && let_Propagator_be_ready; i++)
    if(list_of_propagators_to_delete[i]==Propagator_name)
      let_Propagator_be_ready = false;
  if(let_Propagator_be_ready&&list_of_propagators_to_delete.size()>0)
    set_job_done_momentum_map(&Propagator_name,&input_momentum,true);


  //make other solvers condense their memory when their results are only partially required - controlled via inputdeck option "condense_memory_of"
  std::vector<std::string> list_of_propagators_to_condense;
  options.get_option(std::string("condense_memory_of"),list_of_propagators_to_condense);
  for(unsigned int i=0; i<list_of_propagators_to_condense.size(); i++)
  {
    std::string propagator_name = list_of_propagators_to_condense[i];
    //find the constructor of propagator_name
    std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(propagator_name);
    NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),error_prefix+"have not found the constructor of \""+propagator_name+"\"\n");
    Propagation* temp_solver=dynamic_cast<Propagation*>(temp_it->second);
    if(temp_solver!=NULL)
    {
      Propagator* temp_propagator;
      temp_solver->get_data(propagator_name,temp_propagator);
    }
  }
}

void Propagation::do_load(int sc_it_counter ,  std::string load_poisson_it_number_str, std::string file_name)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_load ");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));
  std::string prefix="Propagation(\""+this->get_name()+"\")::do_load: ";
  std::map<std::string,Simulation*>::const_iterator temp_cit_debug=pointer_to_Propagator_Constructors->begin();
  for(; temp_cit_debug!=pointer_to_Propagator_Constructors->end(); ++temp_cit_debug)
    msg<<prefix<<temp_cit_debug->first<<" "<<temp_cit_debug->second->get_name()<<"\n";

  //check that the Propagation is initialized...
  if(!Propagation_is_initialized)
    initialize_Propagation(get_name());

  //std::map<std::string,Propagator*>::iterator it=writeable_Propagators.begin();
  //for(; it!=writeable_Propagators.end(); ++it)
  if(writeable_Propagator!=NULL)
  {
    msg<<prefix<<"solving propagator \""<<name_of_writeable_Propagator<<"\"\n";
    std::string temp_name = name_of_writeable_Propagator;
    //check that the Propagator is initialized
    if(!is_Propagator_initialized(name_of_writeable_Propagator))
    {
      //find the constructor and let him initialize the Propagator
      std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator);
      NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),prefix+"Constructor of propagator \""+name_of_writeable_Propagator+"\" not found\n");
      Propagation* temp_propagation= dynamic_cast<Propagation*> (temp_it->second);
      NEMO_ASSERT(temp_propagation!=NULL,prefix+"dynamic cast failed\n");
      temp_propagation->initialize_Propagators(name_of_writeable_Propagator);
      msg<<prefix<<"Constructor \""<<temp_propagation->get_name()<<"\" initialized \""<<name_of_writeable_Propagator<<"\n";
      NEMO_ASSERT(is_Propagator_initialized(name_of_writeable_Propagator),
                  prefix+"Propagator \""+name_of_writeable_Propagator+"\" is not initialized after initialize_Propagators is called\n");
      temp_propagation->get_data(name_of_writeable_Propagator,writeable_Propagator);
    }
    Propagator::PropagatorMap::iterator momentum_it=writeable_Propagator->propagator_map.begin();
    NEMO_ASSERT(writeable_Propagator->propagator_map.size()>0,prefix+"empty propagator_map found\n");

    //check that the Propagator is not a scattering self-energy (which requires a special solution)
    if(name_of_writeable_Propagator.find(std::string("scattering"))!=std::string::npos && name_of_writeable_Propagator.find(std::string("self"))!=std::string::npos)
    {
      std::string scattering_type;
      if(options.check_option("scattering_type"))
        scattering_type = options.get_option("scattering_type",std::string(""));
      else
        throw std::invalid_argument(prefix+"please define \"scattering_type\"\n");

      if(scattering_type=="lambda_G_scattering")
      {
        int mom_counter = 0;
        //here, the scattering self-energy is directly proportional to the Green's function - no MPI communication required...
        for(; momentum_it!=writeable_Propagator->propagator_map.end(); momentum_it++,mom_counter++)
        {
          do_load(writeable_Propagator,momentum_it->first,sc_it_counter,load_poisson_it_number_str,mom_counter,file_name);
        }
      }
      else
      {
        //do the MPI-special treatment...
        //std::cerr<<"do_load"<<file_name;
        do_full_MPI_load(writeable_Propagator, sc_it_counter, load_poisson_it_number_str,file_name);
      }
    }
    else //not a scattering self-energy
    {
      int mom_counter = 0;
      //msg<<prefix<<"size of propagator map: "<<it->second->propagator_map.size()<<"\n";
      for(; momentum_it!=writeable_Propagator->propagator_map.end(); momentum_it++,mom_counter++)
      {
        do_load(writeable_Propagator,momentum_it->first,sc_it_counter,load_poisson_it_number_str,mom_counter,file_name);
      }
    }
  }
  if(options.check_option("density_of"))
  {
    calculate_density();
    if(options.get_option("calculate_Jacobian",false))
      calculate_density_Jacobian();
  }
  //test_debug();
  NemoUtils::toc(tic_toc_prefix);
}

//This function is not used can be removed and defined empty with an assert
void Propagation::do_load(Propagator*& result, const std::vector<NemoMeshPoint>& input_momentum,int sc_it_counter,  std::string load_poisson_it_number_str,
                          int mom_counter,std::string file_name)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_load");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));
  activate_regions();
  std::string error_prefix = "Propagation(\""+get_name()+"\")::do_load ";
  msg<<error_prefix<< "solving momentum:\n";
  msg<<error_prefix<< "This code is not well tested:\n";
  for(unsigned int i=0; i<input_momentum.size(); i++)
    input_momentum[i].print();

  const std::string Propagator_name = result->get_name();

  //Propagator_type p_type=get_Propagator_type(Propagator_name);
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=result->propagator_map.find(input_momentum);
  if(matrix_it!=result->propagator_map.end())
    temp_matrix=matrix_it->second;

  int my_local_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_local_rank);
  std::string my_local_rank_str;
  ostringstream convert_rank;
  convert_rank<<my_local_rank;
  my_local_rank_str = convert_rank.str();

  string mom_str;
  ostringstream convert_mom;
  convert_mom<<mom_counter;
  mom_str = convert_mom.str();

  string sc_it_counter_str;
  ostringstream convert_sc_it;
  convert_sc_it<<sc_it_counter;
  sc_it_counter_str = convert_sc_it.str();
  if(sc_it_counter==-1)
  {
    temp_matrix->Read_Mat_From_Binary_File(
      file_name+"_"+my_local_rank_str+"_"+mom_str,
      get_simulation_domain()->get_communicator() );
  }
  else
  {
    temp_matrix->Read_Mat_From_Binary_File(
      file_name+"_"+my_local_rank_str+"_"+mom_str+"_"+sc_it_counter_str+"_"+load_poisson_it_number_str,
      get_simulation_domain()->get_communicator() );
  }

  NEMO_ASSERT(is_ready(Propagator_name,input_momentum),error_prefix+"still not ready\n");
  write_propagator(Propagator_name,input_momentum, temp_matrix);

  if(do_outputL)
    print_Propagator(result,&input_momentum);
  if(clean_up)
    delete_propagator_matrices(result,&input_momentum);

  //make other solvers free memory when their results are no longer required - controlled via inputdeck option "free_memory_of"
  std::vector<std::string> list_of_propagators_to_delete;
  options.get_option(std::string("free_memory_of"),list_of_propagators_to_delete);
  for(unsigned int i=0; i<list_of_propagators_to_delete.size(); i++)
  {
    std::string propagator_name = list_of_propagators_to_delete[i];
    //find the constructor of propagator_name
    //std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(propagator_name);
    Simulation* delete_solver = find_source_of_data(propagator_name);
    NEMO_ASSERT(delete_solver!=NULL,error_prefix+"have not found the solver of \""+propagator_name+"\"\n");

    Propagation* temp_solver=dynamic_cast<Propagation*>(delete_solver);
    if(temp_solver!=NULL)
    {
      Propagator* temp_propagator;
      temp_solver->get_data(propagator_name,temp_propagator);
      //check if the full matrix or only a specific portion is to be deleted
      std::string dofmap_solver=options.get_option("free_DOFmap_solver",std::string(""));
      if(propagator_name.find("combine")!=std::string::npos && dofmap_solver.find("None")==std::string::npos)
      {
        //const DOFmap& temp_DOFmap = get_const_dof_map(get_const_simulation_domain());
        std::string variable_name = "free_";
        const DOFmapInterface& temp_DOFmap=get_special_const_dof_map(variable_name);
        std::string variable_name_col = "free_col_";
        const DOFmapInterface& temp_DOFmap_col=get_special_const_dof_map(variable_name_col);
        temp_solver->delete_propagator_matrices(temp_propagator,&input_momentum,&temp_DOFmap,&temp_DOFmap_col);
      }
      else
        temp_solver->delete_propagator_matrices(temp_propagator,&input_momentum);
    }

    if (delete_solver->get_type().find("Transformation") == std::string::npos)// && temp_solver->get_type().find("QTBM") != std::string::npos)
    {
      //if so, the "rlesults" should remain available, i.e. the job is still done for this momentum
      delete_solver->set_job_done_momentum_map(&propagator_name,&input_momentum,true);
    }
  }
  //if Propagator is not in the list_of_propagators_to_delete, make sure that it remains ready
  bool let_Propagator_be_ready = true;
  for(unsigned int i=0; i<list_of_propagators_to_delete.size() && let_Propagator_be_ready; i++)
    if(list_of_propagators_to_delete[i]==Propagator_name)
      let_Propagator_be_ready = false;
  if(let_Propagator_be_ready&&list_of_propagators_to_delete.size()>0)
    set_job_done_momentum_map(&Propagator_name,&input_momentum,true);

  //make other solvers condense their memory when their results are only partially required - controlled via inputdeck option "condense_memory_of"
  std::vector<std::string> list_of_propagators_to_condense;
  options.get_option(std::string("condense_memory_of"),list_of_propagators_to_condense);
  for(unsigned int i=0; i<list_of_propagators_to_condense.size(); i++)
  {
    std::string propagator_name = list_of_propagators_to_condense[i];
    //find the constructor of propagator_name
    std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(propagator_name);
    NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),error_prefix+"have not found the constructor of \""+propagator_name+"\"\n");
    Propagation* temp_solver=dynamic_cast<Propagation*>(temp_it->second);
    if(temp_solver!=NULL)
    {
      Propagator* temp_propagator;
      temp_solver->get_data(propagator_name,temp_propagator);
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::do_full_MPI_load(Propagator*& result, int sc_it_counter,std::string load_poisson_it_number_str,std::string file_name)
{
  std::string prefix="Propagation(\""+this->get_name()+"\")::do_full_MPI_load";
  std::ostringstream err_str;
  err_str << prefix << "(Propagator*&) sc_it_counter = " << sc_it_counter << ", " <<
          " load_poisson_it_number_str = " << load_poisson_it_number_str << ", file_name = " <<
          file_name << " solver for \"" << result->get_name() << "\" not implemented, yet.\n";
  throw std::invalid_argument(err_str.str());
}

void Propagation::do_solve_inverse(Propagator*&, const std::vector<NemoMeshPoint>&,
                                   PetscMatrixParallelComplex*&)
{
  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_inverse: not implemented\n");
}

void Propagation::do_solve_retarded(Propagator*&, const std::vector<NemoMeshPoint>&,
                                    PetscMatrixParallelComplex*&)
{
  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_retarded: not implemented\n");
}

void Propagation::do_solve_advanced(Propagator*&, const std::vector<NemoMeshPoint>&,
                                    PetscMatrixParallelComplex*&)
{
  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_advanced: not implemented\n");
}

void Propagation::do_solve_greater(Propagator*&, const std::vector<NemoMeshPoint>&,
                                   PetscMatrixParallelComplex*&)
{
  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_greater: not implemented\n");
}

void Propagation::do_solve_lesser(Propagator*&, const std::vector<NemoMeshPoint>&,
                                  PetscMatrixParallelComplex*&)
{
  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_lesser: not implemented\n");
}

void Propagation::do_solve_spectral(Propagator*&, const std::vector<NemoMeshPoint>&,
                                    PetscMatrixParallelComplex*&)
{
  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_spectral: not implemented\n");
}

void Propagation::make_ready(const Propagator* input_Propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::make_ready ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::make_ready: ";

  //std::map<std::string, std::set<Simulation*> >* Solver_map=NULL;
  //get_Solver_list(input_Propagator->get_name(),Solver_map);

  //std::set<Simulation*> solvers=Solver_map->find(input_Propagator->get_name())->second;
  Simulation* solver=pointer_to_Propagator_Constructors->find(input_Propagator->get_name())->second;
  //check that all solver have finished their work
  //std::set<Simulation*>::const_iterator solver_c_it=solvers.begin();
  NEMO_ASSERT(solver!=NULL,prefix+"solver found to be NULL\n");
  //for(; solver_c_it!=solvers.end(); ++solver_c_it)
  {
    Propagation* temp_propagation= dynamic_cast<Propagation*> (solver);
    NEMO_ASSERT(temp_propagation!=NULL,prefix+"dynamic cast failed\n");
    if(!temp_propagation->is_ready(input_Propagator->get_name()))
    {
      msg<<std::string(prefix+"Simulator(\""
                       +temp_propagation->get_name()+"\") reports Propagator \""+input_Propagator->get_name()+"\" is not ready\n");
      msg<<std::string(prefix+"call Simulator(\""
                       +temp_propagation->get_name()+"\").solve()\n");

      //here we will do multithreading
#ifndef NO_MKL
      int num_threads = holder.get_global_options().get_option("self_energy_mkl_threads",1);
      if(holder.get_global_options().check_option("self_energy_mkl_threads"))
      {
        mkl_set_dynamic(false);
        mkl_set_num_threads(num_threads);
      }
#endif
      solver->solve(); //solve self_energy to make it usable
#ifndef NO_MKL
      if(holder.get_global_options().check_option("self_energy_mkl_threads"))
      {
        mkl_set_num_threads(1);
        mkl_set_dynamic(true);
      }
#endif
      NEMO_ASSERT(temp_propagation->is_ready(input_Propagator->get_name()),std::string(prefix+"Simulator(\""
                  +temp_propagation->get_name()+"\") still reports Propagator \""+input_Propagator->get_name()+"\" is not ready. STOP\n"));
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::do_output(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_output ");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));
  msg<<"Propagation(\""<<get_name()<<"\")::do_output\n";
  //throw std::runtime_error("mist schon wieder\n");
  //gather_solutions();
  if(options.get_option("current",false) || current_output)
  {
    calculate_current();
    calculate_energy_interpolated_current();
  }
  if(options.get_option("density_atom_energy_interpolated_output",false))
  {
    calculate_energy_interpolated_atom_density();
  }

  if(options.get_option("local_output_current",false))
  {
    if(local_current_x.size()==0 || local_current_y.size()==0 || local_current_z.size()==0)
    {
      calculate_local_current();
    }
    int myrank;
    MPI_Comm_rank(Mesh_tree_topdown.begin()->first->get_global_comm(),&myrank);
    if(myrank==0)
    {
      print_atomic_map(local_current_x,get_name()+"_local_current_x_"+get_output_suffix("0"));
      print_atomic_map(local_current_y,get_name()+"_local_current_y_"+get_output_suffix("0"));
      print_atomic_map(local_current_z,get_name()+"_local_current_z_"+get_output_suffix("0"));
    }
  }
  if(options.get_option("spatial_resolved_current",false))
    calculate_spatial_current();
  if(options.get_option("energy_current",false))
    calculate_energy_current();
  if(options.get_option("slab_resolved_current",false))
    calculate_slab_resolved_current();
  if((options.get_option("transmission",false))&&!current_output&&!options.get_option("current", false))
    if(momentum_transmission.empty() || options.get_option("momentum_transmission_map_filled_in_backward",false))
      calculate_transmission();
  if(options.get_option("density_output",false))
  {
    if(density.size()==0 || ( (options.get_option("energy_resolved_output",false) || options.get_option("k_resolved_output",false) ) && !energy_resolved_density_ready ))
    {
      calculate_density();
      calculate_energy_interpolated_density();
    }
    int myrank;
    MPI_Comm_rank(Mesh_tree_topdown.begin()->first->get_global_comm(),&myrank);
    if(myrank==0)
    {
      print_atomic_map(density,get_name()+"_density");
    }
  }
  if(options.get_option("density_Jacobian_output",false))
  {
    if(density_Jacobian.size()==0)
      calculate_density_Jacobian();
    int myrank;
    MPI_Comm_rank(Mesh_tree_topdown.begin()->first->get_global_comm(),&myrank);
    if(myrank==0)
    {
      print_atomic_map(density_Jacobian,get_name()+"_density_Jacobian");
    }
  }
  if(options.get_option("calculate_optical_absorption",false))
    calculate_optical_absorption();
  if(debug_output)
  {
    //std::map<std::string, Propagator*>::const_iterator c_it=writeable_Propagators.begin();
    if(writeable_Propagator!=NULL)
      //for(; c_it!=writeable_Propagators.end(); c_it++)
      print_Propagator(writeable_Propagator);
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_format_and_extension(const std::string format,
    const AtomisticDomain* domain,
    NemoOutput*& output,
    std::string& extension)
{
  if (format=="VTK")
  {
    output = new OutputVTK(domain,true);
    extension=".vtk";
  }
  else if(format=="xyz")
  {
    output = new OutputXYZ(domain,true);
    extension=".xyz";
  }
  else if (format=="Silo")
  {
    output = new OutputSilo(domain,true);
    extension=".silo";
  }
  else if (format=="DX")
  {
    output = new OutputDX(domain, true);
    extension=".dx";
  }
  else
    throw std::invalid_argument("[Propagation::get_format_and_extension] Called with unknown format \""+format+"\"\n");
}



void Propagation::print_atomic_maps_per_file(const std::map<vector<double>,std::map<unsigned int,double> >& input_map,const std::string& file_prefix)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_atomic_maps_per_file");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix = "Propagation(\""+get_name()+"\")::print_atomic_maps_per_file";

  //1. create the NemoOutput object depending on the format defined in the inputoptions
  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain_for_output());
  std::string format=options.get_option("output_format",std::string("VTK"));
  NemoOutput* pointer_for_writing_NemoOutput=NULL;
  std::string format_file_ending;
  //2. loop over all maps
  std::map<vector<double>,std::map<unsigned int,double> >::const_iterator map_cit=input_map.begin();
  for(; map_cit!=input_map.end(); ++map_cit)
  {
    Propagation::get_format_and_extension(format,domain,
        pointer_for_writing_NemoOutput,format_file_ending);
    std::stringstream translate_double;
    translate_double.precision(10);
    std::vector<double> temp_vector = map_cit->first;
    for (unsigned int j=0; j<temp_vector.size(); j++)
    {
      if (temp_vector[j]>=0.0)
        translate_double << fabs(temp_vector[j])<<"x";
      else
        translate_double << "m"<<fabs(temp_vector[j])<<"x";
    }
    std::string map_label = translate_double.str();
    //search and replace the "." with "c"
    std::string::iterator string_it=map_label.begin();
    for(; string_it!=map_label.end(); ++string_it)
    {
      if((*string_it)==*(std::string(".").c_str()))
        map_label.replace(string_it,string_it+1,std::string("c").c_str());
    }
    //2.1 call print_single_atomic_map
    std::string filename = output_collector.get_file_path(
        file_prefix + map_label + format_file_ending,
        "print atomic output, e.g. density, DOS, etc",
        NemoFileSystem::RESULTS);
    print_single_atomic_map(map_cit->second, map_label, pointer_for_writing_NemoOutput,input_map.size(),true);
    pointer_for_writing_NemoOutput->write_to_file(filename);
    delete pointer_for_writing_NemoOutput;

  }
  NemoUtils::toc(tic_toc_prefix);

}



void Propagation::print_atomic_maps(const std::map<vector<double>,std::map<unsigned int,double> >& input_map,const std::string& filename)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_atomic_maps ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+get_name()+"\")::print_atomic_maps ";

  //1. create the NemoOutput object depending on the format defined in the inputoptions
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain_for_output());
    std::string format=options.get_option("output_format",std::string("VTK"));
    NemoOutput* pointer_for_writing_NemoOutput=NULL;
    std::string format_file_ending;
    Propagation::get_format_and_extension(format,domain,
                                          pointer_for_writing_NemoOutput,format_file_ending);

    //2. if requested, define a line of projected output
    bool line_corner_file_output=options.check_option("output_along_line_file_output");
    if(line_corner_file_output)
    {

      //prepare atoms for output
      //prepare_atoms_for_line_output
      std::vector<libMesh::Point> atoms_position;
      std::vector<libMesh::Point> points_along_line;
      get_atom_positions_for_line_output(atoms_position, points_along_line);

      //2,4 create labels
      std::vector<std::string> field_labels(input_map.size(),"");
      unsigned int counter = 0;
      std::map<vector<double>,std::map<unsigned int,double> >::const_iterator map_cit=input_map.begin();
      for(; map_cit!=input_map.end(); ++map_cit)
      {
        std::stringstream translate_double;
        translate_double.precision(10);
        std::vector<double> temp_vector = map_cit->first;
        for (unsigned int j=0; j<temp_vector.size(); j++)
        {
          if (temp_vector[j]>=0.0)
            translate_double << fabs(temp_vector[j])<<"x";
          else
            translate_double << "m"<<fabs(temp_vector[j])<<"x";
        }
        std::string map_label = translate_double.str();
        //search and replace the "." with "c"
        std::string::iterator string_it=map_label.begin();
        for(; string_it!=map_label.end(); ++string_it)
        {
          if((*string_it)==*(std::string(".").c_str()))
            map_label.replace(string_it,string_it+1,std::string("c").c_str());
        }
        field_labels[counter] = map_label + " ";
        counter++;
      }

      //2.5 collect data
      int number_of_atoms = atoms_position.size();
      std::vector<double> atoms_data(field_labels.size()*number_of_atoms);
      map_cit=input_map.begin();
      int counter_kpoints = 0;
      for(; map_cit!=input_map.end(); ++map_cit)
      {
        int counter_atom = 0;
        std::map<unsigned int,double>::const_iterator map_cit2=map_cit->second.begin(); //atom-resolved density
        for(; map_cit2!=map_cit->second.end(); ++map_cit2)
        {
          atoms_data[counter_atom*field_labels.size() + counter_kpoints]=map_cit2->second;
          counter_atom++;
        }
        NEMO_ASSERT(counter_atom == (int) number_of_atoms,
                    prefix+"The number of atoms is not the same as the number of values in the map for energy:"+field_labels[counter_kpoints]);
        counter_kpoints++;
      }
      //TODO: Move interpolation functionality to a NemoUtils.
      //2.6 create the interpolation object of libmesh
      int number_of_interpolation_points = options.get_option("number_of_interpolation_points",4);
      int interpolation_power = options.get_option("interpolation_power",2);

      /* Create an InverDistanceInterpolation class named idi.
        Interpolation_helper is the abbreviation of inverse distance interpolation.
        The second argument is the number of surrounding points used to calculate the interpolating values.
        The third argument is the power in the formula of calculating the weight of the surrounding points.
       */
      std::vector<double> tgt_vals(points_along_line.size()*input_map.size());
      MeshfreeInterpolation* interpolation_helper = new InverseDistanceInterpolation<3> (libMesh::Parallel::Communicator(MPI_COMM_SELF), number_of_interpolation_points,
          interpolation_power);
      interpolation_helper->add_field_data (field_labels, atoms_position, atoms_data);
      interpolation_helper->prepare_for_use();
      interpolation_helper->interpolate_field_data (field_labels, points_along_line, tgt_vals);
      delete interpolation_helper;

      std::string filename2 = filename+"_output_along_a_line.dat";
      std::ofstream outfile;
      outfile.open(filename2.c_str());
      outfile << std::left << std::setw(10) << "% x[nm]"
              << std::left << std::setw(10) << "y[nm]"
              << std::left << std::setw(10) << "z[nm]";
      for(unsigned int j=0; j<field_labels.size(); j++)
        outfile << std::left << std::setw(15) << field_labels[j];
      outfile << std::endl;
      for(unsigned int i=0; i<points_along_line.size(); i++)
      {
        outfile << std::left << std::setw(10) << points_along_line[i](0)
                << std::left << std::setw(10) <<points_along_line[i](1)
                << std::left << std::setw(10) << points_along_line[i](2);
        for(unsigned int j=0; j<input_map.size(); j++)
          outfile << std::left << std::setw(15) << tgt_vals[j + i*input_map.size()];
        outfile << std::endl;
      }
      outfile.close();
    }

    //3. loop over all maps
    std::map<vector<double>,std::map<unsigned int,double> >::const_iterator map_cit=input_map.begin();
    for(; map_cit!=input_map.end(); ++map_cit)
    {
      std::stringstream translate_double;
      translate_double.precision(10);
      std::vector<double> temp_vector = map_cit->first;
      for (unsigned int j=0; j<temp_vector.size(); j++)
      {
        if (temp_vector[j]>=0.0)
          translate_double << fabs(temp_vector[j])<<"x";
        else
          translate_double << "m"<<fabs(temp_vector[j])<<"x";
      }
      std::string map_label = translate_double.str();
      //search and replace the "." with "c"
      std::string::iterator string_it=map_label.begin();
      for(; string_it!=map_label.end(); ++string_it)
      {
        if((*string_it)==*(std::string(".").c_str()))
          map_label.replace(string_it,string_it+1,std::string("c").c_str());
      }
      //2.1 call print_single_atomic_map
      print_single_atomic_map(map_cit->second, map_label, pointer_for_writing_NemoOutput,input_map.size(),true);
    }
    pointer_for_writing_NemoOutput->write_to_file(filename + format_file_ending);
    delete pointer_for_writing_NemoOutput;

  NemoUtils::toc(tic_toc_prefix);
}

template<typename T>
void Propagation::print_atomic_maps(const std::map<T,std::map<unsigned int,double>,Compare_double_or_complex_number >& input_map,const std::string& filename, const std::string& file_describe)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_atomic_maps ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix = "Propagation(\""+get_name()+"\")::print_atomic_maps ";

  //1. create the NemoOutput object depending on the format defined in the inputoptions
  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain_for_output());
  std::string format=options.get_option("output_format",std::string("VTK"));
  NemoOutput* pointer_for_writing_NemoOutput=NULL;
  std::string format_file_ending;
  Propagation::get_format_and_extension(format,domain,
      pointer_for_writing_NemoOutput,format_file_ending);

  //2. if requested, define a line of projected output
  bool along_line_file_output=options.get_option("output_along_line_file_output",false);
  bool line_output=options.check_option("output_along_line");

  if(along_line_file_output || line_output)
  {

    //prepare atoms for output
    std::vector<libMesh::Point> atoms_position;
    std::vector<libMesh::Point> points_along_line;
    get_atom_positions_for_line_output(atoms_position, points_along_line);

    //2,4 create labels
    std::vector<std::string> field_labels(input_map.size(),"");
    unsigned int counter = 0;
    typename std::map<T,std::map<unsigned int,double>,Compare_double_or_complex_number >::const_iterator map_cit=input_map.begin();
    for(; map_cit!=input_map.end(); ++map_cit)
    {
      field_labels[counter] = NemoUtils::nemo_to_string(map_cit->first)+"[eV]";
      counter++;
    }
    //2.5 collect data
    int number_of_atoms = atoms_position.size();
    std::vector<double> atoms_data(field_labels.size()*number_of_atoms);
    map_cit=input_map.begin();
    int counter_energy = 0;
    for(; map_cit!=input_map.end(); ++map_cit)
    {
      int counter_atom = 0;
      std::map<unsigned int,double>::const_iterator map_cit2=map_cit->second.begin(); //atom-resolved density
      for(; map_cit2!=map_cit->second.end(); ++map_cit2)
      {
        atoms_data[counter_atom*field_labels.size() + counter_energy]=map_cit2->second;
        counter_atom++;
      }
      cerr << " counter atom " << counter_atom << " number of atoms " << number_of_atoms << " \n";

      NEMO_ASSERT(counter_atom == (int) number_of_atoms,
          prefix+"The number of atoms is not the same as the number of values in the map for energy:"+field_labels[counter_energy]);
      counter_energy++;
    }
    //TODO: Move interpolation functionality to a NemoUtils.
    //2.6 create the interpolation object of libmesh
    int number_of_interpolation_points = options.get_option("number_of_interpolation_points",4);
    int interpolation_power = options.get_option("interpolation_power",2);

    /* Create an InverDistanceInterpolation class named idi.
          Interpolation_helper is the abbreviation of inverse distance interpolation.
          The second argument is the number of surrounding points used to calculate the interpolating values.
          The third argument is the power in the formula of calculating the weight of the surrounding points.
     */
    std::vector<double> tgt_vals(points_along_line.size()*input_map.size());

    MeshfreeInterpolation* interpolation_helper = new InverseDistanceInterpolation<3> (libMesh::Parallel::Communicator(MPI_COMM_SELF),
                                                                                       number_of_interpolation_points,
                                                                                       interpolation_power);
    interpolation_helper->add_field_data (field_labels, atoms_position, atoms_data);
    interpolation_helper->prepare_for_use();
    interpolation_helper->interpolate_field_data (field_labels, points_along_line, tgt_vals);
    delete interpolation_helper;
    std::string suffix = get_output_suffix();
    std::string filename2;
    if(!suffix.empty())
    {
      filename2 = filename+"_output_along_a_line_"+get_output_suffix()+".dat";
    }
    else
    {
      filename2 = filename+"_output_along_a_line.dat";
    }
    std::ofstream outfile;
    output_collector.get_file_path( filename2, file_describe ,NemoFileSystem::RESULTS);
    outfile.open(filename2.c_str());
    outfile << std::left << std::setw(10) << "% x[nm]"
        << std::left << std::setw(10) << "y[nm]"
        << std::left << std::setw(10) << "z[nm]";
    for(unsigned int j=0; j<field_labels.size(); j++)
      outfile << std::left << std::setw(15) << field_labels[j];
    outfile << std::endl;
    for(unsigned int i=0; i<points_along_line.size(); i++)
    {
      outfile << std::left << std::setw(10) << points_along_line[i](0)
                      << std::left << std::setw(10) <<points_along_line[i](1)
                      << std::left << std::setw(10) << points_along_line[i](2);
      for(unsigned int j=0; j<input_map.size(); j++)
        outfile << std::left << std::setw(15) << tgt_vals[j + i*input_map.size()];
      outfile << std::endl;
    }
    outfile.close();
  }

  //3. loop over all maps
  typename std::map<T,std::map<unsigned int,double>,Compare_double_or_complex_number >::const_iterator map_cit=input_map.begin();
  for(; map_cit!=input_map.end(); ++map_cit)
  {
    std::stringstream translate_double;
    translate_double.precision(18);
    translate_double<<map_cit->first;
    std::string map_label=translate_double.str();
    //2.1 call print_single_atomic_map
    print_single_atomic_map(map_cit->second, map_label, pointer_for_writing_NemoOutput,input_map.size());
  }
  pointer_for_writing_NemoOutput->write_to_file(filename + format_file_ending);
  delete pointer_for_writing_NemoOutput;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::print_atomic_map(const std::map<unsigned int,double>& input,const std::string& filename,const std::string& file_describe)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_atomic_map ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix = "Propagation(\""+get_name()+"\")::print_atomic_map ";


  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain_for_output());
  if(!no_file_output)
  {
    if(atomic_output_only)
    {
      activate_regions_domain_for_output();
      std::ofstream density_file(std::string(filename+".dat").c_str(),std::ios_base::out | std::ios_base::trunc);

      const AtomicStructure&  atoms  = domain->get_atoms();
      ConstActiveAtomIterator it     = atoms.active_atoms_begin();
      ConstActiveAtomIterator end    = atoms.active_atoms_end();

      unsigned int atom_index=atoms.local_active_atoms_size();


      string tmp = filename + ".atid";
      ofstream out_file(tmp.c_str());

      std::vector<double> vtk_density(atom_index,0.0);
      atom_index=0;
      // fill the vector atom by atom
      for ( ; it != end; ++it)
      {
        const AtomStructNode& nd      = it.node();
        const unsigned int    atom_id = it.id();

        std::map<unsigned int ,double >::const_iterator c_it=input.find(atom_id);
        NEMO_ASSERT(c_it!=input.end(),prefix+"have not found the atom\n");
        double temp=c_it->second;
        out_file << it.id() << " " << temp << endl;
        density_file << nd.position[0] << "  " << nd.position[1] << "  " << nd.position[2] << "  ";
        density_file << temp << "\n";
        vtk_density[atom_index]=temp;
        atom_index++;
      }
      out_file.close();
      density_file.close();
      std::string format=options.get_option("output_format",std::string("VTK"));
      if (format=="VTK")
      {
        OutputVTK out(domain, true); // active atoms only
        out.add_dataset(filename,&vtk_density);
        out.write_to_file(filename + ".vtk");
        output_collector.get_file_path( filename+".vtk", file_describe ,NemoFileSystem::RESULTS);

        OutputXYZ xyzout(domain, true);
        xyzout.add_dataset(filename,&vtk_density);
        output_collector.get_file_path( filename+".xyz", file_describe ,NemoFileSystem::RESULTS);
        xyzout.write_to_file(filename + ".xyz");
      }
      //.dx write to file, same usage as VTK
      else if (format=="DX")
      {
        OutputDX dxout(domain, true);
        dxout.add_dataset(filename,&vtk_density);
        dxout.write_to_file(filename + ".dx");
      }
      else if (format=="Silo")
      {
        OutputSilo siloout(domain,true);
        siloout.add_dataset(filename,&vtk_density);
        siloout.write_to_file(filename + ".silo");
      }
    }
    else
    {
      const AtomicStructure&  atoms  = domain->get_atoms();
      ConstActiveAtomIterator it     = atoms.active_atoms_begin();
      ConstActiveAtomIterator end    = atoms.active_atoms_end();

      const DOFmapInterface& dof_map = get_const_dof_map(get_const_simulation_domain());


      //find the maximum implemented number of orbitals and their names
      unsigned int n_orbital=dof_map.get_n_vars();
      unsigned int n_atoms=atoms.local_active_atoms_size();

      std::vector<std::string> orbital_namesV= dof_map.get_variables();
      std::vector<std::vector<double> > orbital_resolved_result(n_orbital,std::vector<double>(n_atoms,0.0));
      unsigned int atom_counter=0;
      const Material* last_material = NULL;
      for(it=atoms.active_atoms_begin(); it != end; ++it)
      {


        //store the result in respective vectors of orbital_resolved_result
        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(it.id());
        std::map<short, unsigned int>::const_iterator it =   atom_dofs->begin();

        for(; it!=atom_dofs->end(); it++)
        {
          //std::map<short, unsigned int>::const_iterator it2 = atom_dofs->find((short)i);
          unsigned int j=it->second;
          const std::map<unsigned int,double>::const_iterator c_it = input.find(j);
          NEMO_ASSERT(c_it!=input.end(),prefix+"have not found result index\n");
          orbital_resolved_result.at(it->first).at(atom_counter)=c_it->second;
        }
        atom_counter++;
      }
      //create the Output objects according to the user defined output format
      //VTK format
      OutputVTK out(domain, true); // active atoms only
      //add the datasets and names to the outputobjects, then write to file
      for(unsigned int i=0; i<orbital_namesV.size(); i++)
      {
        out.add_dataset(orbital_namesV[i], &(orbital_resolved_result[i]));
        //out.write_to_file(filename + ".vtk");
      }
      out.write_to_file(filename + ".vtk");


      OutputXYZ xyzout(domain, true);
      for(unsigned int i=0; i<orbital_namesV.size(); i++)
      {
        xyzout.add_dataset(orbital_namesV[i],&(orbital_resolved_result[i]));
        //cerr<<"i"<<i<<orbital_namesV[i]<<orbital_resolved_result[i].size()<<"\n";
      }
      xyzout.write_to_file(filename + ".xyz");
    }
  }
  //in the case of orbital resolved output: do like in Schroedinger - get the orbital names from Hamilton_constructor and add more datafields to the output class

  //for the output, the indices will be given as DOFmap indices - use the translational map as in assemble_hamiltonian of Schroedinger
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_atom_positions_for_line_output(std::vector<libMesh::Point>& atoms_position, std::vector<libMesh::Point>& points_along_line)
{

  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_atom_positions_for_line_output ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+get_name()+"\")::get_atom_positions_for_line_output ";

  //2.1 read the line from inputdeck
  std::vector<std::vector<double> > output_line_points;
  options.get_option("output_along_line",output_line_points);
  NEMO_ASSERT(output_line_points.size()>1,prefix+"please define at least 2 points for \"output_line_points\"\n");
  //2.2 find the set of relevant atom ids
  //2.2.1 sample the line and create points along the line
  //read the number of sample points
  std::vector<unsigned int> line_resolution;
  options.get_option("output_along_line_resolution",line_resolution);
  NEMO_ASSERT(line_resolution.size()==output_line_points.size()-1,prefix+"please define exactly "+NemoUtils::nemo_to_string(output_line_points.size()-1)
  +" values in \"output_along_line_resolution\"\n");
  //create the vector of coordinates
  int total_number_of_line_points = 0;
  for(unsigned int i = 0; i < line_resolution.size(); ++i)
    total_number_of_line_points += line_resolution[i] + 1;
  //std::vector<libMesh::Point> points_along_line(total_number_of_line_points);
  points_along_line.resize(total_number_of_line_points);
  int counter = 0;
  for(unsigned int i = 0; i < output_line_points.size() - 1; ++i)
  {
    for(unsigned int j = 0; j <= line_resolution[i]; ++j)
    {
      std::vector<double> tmp_point(3,0.0);
      double distance = double(j)/double(line_resolution[i]);
      for(unsigned int jj = 0; jj < 3; ++jj)
      {
        tmp_point[jj] = output_line_points[i][jj]+(output_line_points[i+1][jj]-output_line_points[i][jj])*distance;
      }
      points_along_line[counter] = libMesh::Point(tmp_point[0],tmp_point[1],tmp_point[2]);
      counter++;
    }
  }
  //2.3 Get atoms position as libmesh points
  //get all points of the current mesh
  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
  const AtomicStructure& atoms   = domain->get_atoms();
  ConstActiveAtomIterator active_atoms_iterator = atoms.active_atoms_begin();
  ConstActiveAtomIterator active_atoms_iterator_end = atoms.active_atoms_end();
  unsigned int number_of_atoms = 0;
  for ( ; active_atoms_iterator != active_atoms_iterator_end; ++active_atoms_iterator)
    number_of_atoms++;

  //NEMO_ASSERT(input_map.begin()->second.size() == number_of_atoms,prefix+"line-output does not support orbital resolved data yet\n");

  active_atoms_iterator  = atoms.active_atoms_begin();
  //std::vector<libMesh::Point> atoms_position(number_of_atoms);
  atoms_position.resize(number_of_atoms);
  counter = 0;
  for ( ; active_atoms_iterator != active_atoms_iterator_end; ++active_atoms_iterator)
  {
    const AtomStructNode& nd = active_atoms_iterator.node();
    atoms_position[counter] = libMesh::Point(nd.position[0],nd.position[1],nd.position[2]);
    counter++;
  }
  NemoUtils::toc(tic_toc_prefix);

}


void Propagation::print_single_atomic_map(const std::map<unsigned int,double>& input, const std::string& map_label, NemoOutput* output_object,
    const unsigned int& size_of_vector, bool k_resolved)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_single_atomic_map ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix = "Propagation(\""+get_name()+"\")::print_single_atomic_map ";
  std::string filename="";
  NEMO_ASSERT(input.size()>0,prefix+"called with empty map\n");

  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain_for_output());
  if(atomic_output_only)
  {
    const AtomicStructure&  atoms  = domain->get_atoms();
    ConstActiveAtomIterator it     = atoms.active_atoms_begin();
    ConstActiveAtomIterator end    = atoms.active_atoms_end();

    unsigned int atom_index=0;
    for ( ; it != end; ++it)
      atom_index+=1;

    it=atoms.active_atoms_begin();

    std::vector<double> vtk_density(atom_index,0.0);
    atom_index=0;
    // fill the vector atom by atom
    for ( ; it != end; ++it)
    {
      //const AtomStructNode& nd      = it.node();
      const unsigned int    atom_id = it.id();

      std::map<unsigned int ,double >::const_iterator c_it=input.find(atom_id);
      NEMO_ASSERT(c_it!=input.end(),prefix+"have not found the atom\n");
      double temp=c_it->second;
      vtk_density[atom_index]=temp;
      atom_index++;
    }
    //the pointer to the data has to point to the appropriate entry of energy_orbital_resolved_result
    if(!k_resolved)
    {
      if(energy_orbital_resolved_result.size()==0)
      {
        std::vector< std::vector<double> > temp_vector(1,vtk_density);
        energy_orbital_resolved_result.resize(size_of_vector,temp_vector);
      }
      energy_orbital_resolved_result[output_counter][0]=vtk_density;
      output_counter++;

      std::string format=options.get_option("output_format",std::string("VTK"));
      output_object->add_dataset(map_label+"_"+filename,&(energy_orbital_resolved_result[output_counter-1][0]));
    }
    else
    {
      if(k_orbital_resolved_result.size()==0)
      {
        std::vector< std::vector<double> > temp_vector(1,vtk_density);
        k_orbital_resolved_result.resize(size_of_vector,temp_vector);
      }
      k_orbital_resolved_result[output_counter][0]=vtk_density;
      output_counter++;

      std::string format=options.get_option("output_format",std::string("VTK"));
      output_object->add_dataset(map_label+"_"+filename,&(k_orbital_resolved_result[output_counter-1][0]));
    }
  }
  else
  {
    const AtomicStructure&  atoms  = domain->get_atoms();
    ConstActiveAtomIterator it     = atoms.active_atoms_begin();
    ConstActiveAtomIterator end    = atoms.active_atoms_end();

    const DOFmapInterface& dof_map = get_const_dof_map(get_const_simulation_domain());

    ConstActiveAtomIterator maximum_orbitals_atom_it = it;
    //find the maximum implemented number of orbitals and their names
    unsigned int n_orbital=0;
    unsigned int n_atoms=0;
    std::vector<std::string> orbital_namesV;
    for (it=atoms.active_atoms_begin(); it != end; ++it)
    {
      if(particle_type_is_Fermion)
      {
        const AtomStructNode& nd        = it.node();
        const Atom*           atom      = nd.atom;
        HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (Hamilton_Constructor->get_material_properties(atom->get_material()));
        NEMO_ASSERT(tb_ham!=NULL, prefix+"Hamilton constructor pointer is NULL\n");
        if(n_orbital<tb_ham->get_number_of_orbitals(atom))
        {
          n_orbital = tb_ham->get_number_of_orbitals(atom);
          maximum_orbitals_atom_it=it;
          //get the names of these orbitals
          orbital_namesV = tb_ham->get_orbital_names();
        }
      }
      else
      {
        const AtomStructNode& nd        = it.node();
        const Atom*           atom      = nd.atom;
        StrainVFF* temp_strainvff = dynamic_cast<StrainVFF*>(Hamilton_Constructor);
        NEMO_ASSERT(temp_strainvff!=NULL, prefix+"DOF resolved output NYI for Bosons that are not phonons");
        n_orbital = temp_strainvff->get_number_of_orbitals(atom);
        orbital_namesV = temp_strainvff->get_orbital_names();
      }

      n_atoms++;
    }
    //we assume that every atom has the same number of orbitals (if not, then 0 is used for the missing orbital(s))
    //create storage for the individual orbital densities
    std::vector<std::vector<double> > orbital_resolved_result(n_orbital,std::vector<double>(n_atoms,0.0));
    unsigned int atom_counter=0;
    for(it=atoms.active_atoms_begin(); it != end; ++it)
    {
      //store the result in respective vectors of orbital_resolved_result
      const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(it.id());
      for(unsigned int i=0; i<n_orbital; i++)
      {
        std::map<short, unsigned int>::const_iterator it2 = atom_dofs->find((short)i);
        unsigned int j=it2->second;
        const std::map<unsigned int,double>::const_iterator c_it = input.find(j);
        NEMO_ASSERT(c_it!=input.end(),prefix+"have not found result index\n");
        orbital_resolved_result[i][atom_counter]=c_it->second;
      }
      atom_counter++;
    }
    NEMO_ASSERT(n_atoms==atom_counter,prefix+"inconsistent number of atoms\n");
    //create the Output objects according to the user defined output format
    //VTK format
    //add the datasets and names to the outputobjects, then write to file
    //hereby, the pointer to the data has to point to the appropriate entry of energy_orbital_resolved_result

    if(!k_resolved)
    {
      if(energy_orbital_resolved_result.size()==0)
        energy_orbital_resolved_result.resize(size_of_vector,orbital_resolved_result);

      int storage_index=output_counter;//energy_orbital_resolved_result.size()-1;
      energy_orbital_resolved_result[storage_index]=orbital_resolved_result;
      for(unsigned int i=0; i<orbital_namesV.size(); i++)
      {
        std::vector<double>* temp_pointer=&(energy_orbital_resolved_result[storage_index][i]);
        NEMO_ASSERT(temp_pointer!=NULL,prefix+"NULL received\n");
        output_object->add_dataset(map_label+"_"+orbital_namesV[i], temp_pointer);
      }
      output_counter++;
    }
    else
    {
      if(k_orbital_resolved_result.size()==0)
        k_orbital_resolved_result.resize(size_of_vector,orbital_resolved_result);

      int storage_index=output_counter;//energy_orbital_resolved_result.size()-1;
      k_orbital_resolved_result[storage_index]=orbital_resolved_result;
      for(unsigned int i=0; i<orbital_namesV.size(); i++)
      {
        std::vector<double>* temp_pointer=&(k_orbital_resolved_result[storage_index][i]);
        NEMO_ASSERT(temp_pointer!=NULL,prefix+"NULL received\n");
        output_object->add_dataset(map_label+"_"+orbital_namesV[i], temp_pointer);
      }
      output_counter++;
    }
  }
  //in the case of orbital resolved output: do like in Schroedinger - get the orbital names from Hamilton_constructor and add more datafields to the output class

  //for the output, the indices will be given as DOFmap indices - use the translational map as in assemble_hamiltonian of Schroedinger
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::clean_before_reinit()
{}

void Propagation::do_init()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_init ");
  NemoUtils::tic(tic_toc_prefix);
  //Propagation::list_of_Propagators.insert(this);
  always_ready = options.get_option("always_ready",false);
  ignore_callers_ready_map =  options.get_option("ignore_callers_ready_map",true);
  base_init();
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::do_reinit()
{

  std::string prefix = "Propagation(\""+get_name()+"\")::do_reinit ";
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_reinit ");
  NemoUtils::tic(tic_toc_prefix);
  always_ready = options.get_option("always_ready",false);
  ignore_callers_ready_map =  options.get_option("ignore_callers_ready_map",true);
  if(!options.get_option("clean_all_in_reinit",false))
  {
    base_read_function_list();
    //1. delete all Propagators
    //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.begin();
    if(writeable_Propagator!=NULL)
      //for(; it!=writeable_Propagators.end(); it++)
    {
      //do not delete the combine-propagators, except explicitly told to do so
      if(name_of_writeable_Propagator.find("combine")==std::string::npos || options.get_option("delete_combine",false))
        delete_propagator_matrices(writeable_Propagator);
      //2. set_job_done_momentum to false
      set_job_done_momentum_map(&name_of_writeable_Propagator,NULL,false);
    }

    //3. read from the input deck whether there are new readable Propagators, or existing should be ignored
    update_readable_propagator_list();
    //3. read from input deck whether there are new Proapagations
    //update_solvers_list();

    //4. check that the Hamilton constructor has the same domain
    std::string temp_name=options.get_option("Hamilton_constructor",std::string(""));
    Hamilton_Constructor=this->find_simulation(temp_name);
    NEMO_ASSERT(Hamilton_Constructor!=NULL,prefix+"Hamilton_constructor("+'\"'+temp_name+
                '\"'+") was not found\n");
    NEMO_ASSERT(get_const_simulation_domain()==Hamilton_Constructor->get_const_simulation_domain(),
                prefix +"Hamilton_constructor(\" "+Hamilton_Constructor->get_name()+"\") has domain" +
                " \""+Hamilton_Constructor->get_const_simulation_domain()->get_name()+"\" instead of the required \"" + get_const_simulation_domain()->get_name()
                +"\"\n");

    delete out_of_device_decaying_modes;
    delete out_of_device_decaying_phase;
    delete out_of_device_propagating_modes;
    delete out_of_device_propagating_phase;
    delete into_device_decaying_modes;
    delete into_device_decaying_phase;
    delete into_device_propagating_modes;
    delete into_device_propagating_phase;
    delete into_device_modes;
    delete out_of_device_modes;
    delete into_device_phase;
    delete out_of_device_phase;
    delete into_device_velocity;
    delete out_of_device_velocity;
    delete into_device_propagating_velocity;
    delete out_of_device_propagating_velocity;
    if(temporary_sub_matrix!=NULL)
      delete temporary_sub_matrix;
    temporary_sub_matrix=NULL;
    into_device_propagating_modes = NULL;
    into_device_decaying_modes = NULL;
    out_of_device_propagating_modes = NULL;
    out_of_device_decaying_modes = NULL;
    into_device_propagating_phase = NULL;
    into_device_decaying_phase = NULL;
    out_of_device_propagating_phase = NULL;
    out_of_device_decaying_phase = NULL;
    into_device_modes = NULL;
    into_device_phase = NULL;
    out_of_device_modes = NULL;
    out_of_device_phase = NULL;
    into_device_velocity = NULL;
    out_of_device_velocity = NULL;
    into_device_propagating_velocity = NULL;
    out_of_device_propagating_velocity = NULL;
    energy_orbital_resolved_result.clear();
    output_counter = 0;
  }
  else
  {
    clean_all();
    do_init();
  }

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::base_read_function_list (void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::base_read_function_list ");
  NemoUtils::tic(tic_toc_prefix);
  // -----------------------------------
  // read in which Propagator-objects are considered
  // read in what Propagator type these objects are
  // read in which simulations are constructing these objects
  // read in which simulations are solving these objects
  // -----------------------------------

  msg.set_level(MsgLevel(4));
  std::string prefix="Propagation(\""+this->get_name()+"\")::base_read_function_list: ";
  //test_debug();
  // read in which Propagator-objects are considered
  std::vector<string> job_list;
  options.get_option("NEGF_object_list",job_list);
  if (job_list.size()>0)
  {
    for (unsigned int loop_jobs=0; loop_jobs<job_list.size(); loop_jobs++)
    {
      std::string name=job_list[loop_jobs];

      //for Compression: before adding this propagator to the list and creating it...
      //...check that its MPI-rank corresponds to the one of this CPU (if the rank is defined
      std::string option_name=name+"_MPI_rank";
      if(options.check_option(name+"_MPI_rank"))
      {
        int propagator_rank = options.get_option(name+"_MPI_rank",-1);
        int myrank;
        MPI_Comm_rank(get_simulation_domain()->get_communicator(),&myrank);
        if(myrank==propagator_rank)
          fill_Propagator_type_map(name);
        //else: ignore this Propagator
      }
      else
        fill_Propagator_type_map(name);
    }
  }
  else
  {
    NemoUtils::msg << prefix<<"[Propagation] nothing to be done; quit Propagation ("+this->get_name()+")\n";
    NemoUtils::toc(tic_toc_prefix);
    return;
  }
  //test_debug();
  // -----------------------------------
  // read in which simulation "deals" with
  // the propagators
  // -----------------------------------
  std::map<std::string,NemoPhys::Propagator_type>::const_iterator it=Propagator_types.begin();
  for (; it!=Propagator_types.end(); ++it)
  {
    Simulation* constructor=NULL;
    //read in where the user-defined NEGF-objects are constructed
    std::string parameter_name=it->first+std::string("_constructor");
    if(options.check_option(parameter_name))
    {

      std::string Constructor_name=options.get_option(parameter_name,std::string(""));
      Simulation* temp_simulation1 = this->find_simulation(Constructor_name);
      NEMO_ASSERT(temp_simulation1!=NULL, prefix+"simulation \""+Constructor_name+"\" has not been found\n");
      //Propagation* temp_simulation=dynamic_cast<Propagation*> (temp_simulation1);
      if(pointer_to_Propagator_Constructors->find(it->first)!=pointer_to_Propagator_Constructors->end())
      {
        Simulation* temp_solver = pointer_to_Propagator_Constructors->find(it->first)->second;
        if(temp_solver!=NULL)
        {
          std::string temp_name = temp_solver->get_name();
          if(pointer_to_Propagator_Constructors->find(it->first)->second!=this)
            NEMO_ASSERT(temp_simulation1==pointer_to_Propagator_Constructors->find(it->first)->second,
                        prefix+"ambiguous definition for constructors of \""+it->first+
                        "\": solvers \""+Constructor_name+"\" and \""+temp_name+"\" found as constructors\n");
        }
        else
          (*pointer_to_Propagator_Constructors)[it->first]=temp_simulation1;
      }
      else
      {
        //test_debug();
        (*pointer_to_Propagator_Constructors)[it->first]=temp_simulation1;
        //test_debug();
      }
      constructor=pointer_to_Propagator_Constructors->find(it->first)->second;
      if (constructor==this)
      {
        msg<< get_name()<< " is constructing "<<it->first<<std::endl;
        //construct/allocate the Propagator
        std::string temp_name = it->first;
        NemoPhys::Propagator_type temp_type = it->second;
        //test_debug();
        if(Propagators.find(temp_name)==Propagators.end())
        {
          Propagator* temp_propagator = new Propagator(temp_name, temp_type);
          //test_debug();
          Propagators[it->first]=temp_propagator;
          //writeable_Propagators[it->first]=temp_propagator;
          writeable_Propagator=temp_propagator;
          name_of_writeable_Propagator=temp_propagator->get_name();
          
          type_of_writeable_Propagator=Propagator_types.find(name_of_writeable_Propagator)->second;

          if(name_of_writeable_Propagator.find("combine")!=std::string::npos)
            this_is_combine_Propagation=true;
          else
            this_is_combine_Propagation=false;
          if(name_of_writeable_Propagator.find("Boson")!=std::string::npos)
            particle_type_is_Fermion=false;
          ready_Propagator_map[it->first]=false; //will be set to true in "this"->do_solve
          //test_debug();
        }
        //test_debug();

      }
      else
      {
        if(Propagators.find(it->first)==Propagators.end())
          Propagators[it->first]=NULL;
      }
    }
    else //if (pointer_to_Propagator_Constructors->find(it->first)==pointer_to_Propagator_Constructors->end())
      throw std::invalid_argument(prefix+"parameter \""+parameter_name+"\" is not defined\n");

    //if(constructor!=this)
    //{
    //  //test_debug();
    //  std::set<Simulation*> temp_set;
    //  temp_set.insert(this);
      //Propagator_Callers[it->first]=temp_set;
      //test_debug();
      //corrected version get the caller set from the constructor and add this to it
      //std::map<std::string, std::set<Simulation*> >* pointer_to_caller_list=NULL;
      //test_debug();
      //get_Caller_list(it->first, pointer_to_caller_list);
      //if(pointer_to_caller_list!=NULL) //,prefix+"pointer to the solver list is NULL\n");
      //{
      //  //test_debug();
      //  if(pointer_to_caller_list->find(it->first)!=pointer_to_caller_list->end())
      //    pointer_to_caller_list->find(it->first)->second.insert(this);
      //  else
      //  {
      //    std::set<Simulation*> temp_set;
      //    temp_set.insert(this);
      //    (*pointer_to_caller_list)[it->first]=temp_set;
      //  }
      //}
      //test_debug();
    //}
    //test_debug();
    interpret_NEGF_object(it->first,constructor);
  }
  //test_debug();
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::interpret_NEGF_object(const std::string& name, Simulation* constructor)
{
  NEMO_ASSERT(constructor!=NULL,"Propagation(\""+get_name()+"\")::interpret_NEGF_object received NULL for constructor\n");
  NemoPhys::Propagator_type type=get_Propagator_type(name);
  if(type==NemoPhys::Inverse_Green)
    inverse_GR_solver=constructor;
  else if(type==NemoPhys::Fermion_retarded_Green||type==NemoPhys::Boson_retarded_Green)
  {
    exact_GR_solver=constructor;
  }
  else if(type==NemoPhys::Fermion_retarded_self||type==NemoPhys::Boson_retarded_self)
  {
    if(name.find("contact")!=std::string::npos)
      contact_sigmaR_solvers.insert(constructor);
    else
      scattering_sigmaR_solver=constructor;
  }
  else if(type==NemoPhys::Fermion_lesser_Green||type==NemoPhys::Boson_lesser_Green)
  {
    exact_GL_solver=constructor;
  }
  else if(type==NemoPhys::Fermion_lesser_self||type==NemoPhys::Boson_lesser_self)
  {
    if(name.find("contact")!=std::string::npos)
      contact_sigmaL_solvers.insert(constructor);
    else
      scattering_sigmaL_solver=constructor;
  }
}

void Propagation::update_readable_propagator_list(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::update_readable_propagator_list ");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));
  std::string prefix="Propagation(\""+this->get_name()+"\")::update_readable_propagator_list: ";
  //add new propagators:
  std::string variable_name = "add_readable_Propagators";
  if(options.check_option(variable_name))
  {
    std::vector<std::string> new_propagators;
    options.get_option(variable_name,new_propagators);
    for(unsigned int i=0; i<new_propagators.size(); i++)
    {
      //check that the new Propagator is known in the static maps and that there is a constructor for it
      std::map<std::string,Simulation*>::const_iterator c_it=pointer_to_Propagator_Constructors->find(new_propagators[i]);
      if(c_it==pointer_to_Propagator_Constructors->end())
      {
        NEMO_ASSERT(options.check_option(new_propagators[i]+"_constructor"),prefix+"please define \""+new_propagators[i]+"_constructor\"\n");
        std::string constructor_name=options.get_option(new_propagators[i]+"_constructor",std::string(""));
        Simulation* temp_simulation = find_simulation(constructor_name);
        NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+constructor_name+"\"\n");
        (*pointer_to_Propagator_Constructors)[new_propagators[i]]=temp_simulation;
        c_it=pointer_to_Propagator_Constructors->find(new_propagators[i]);
      }
      NEMO_ASSERT(c_it!=pointer_to_Propagator_Constructors->end(),prefix+"constructor of propagator \""+new_propagators[i]+"\" is not known\n");
      //check that it is not in the writeable Propagator list
      //std::map<std::string, Propagator*>::const_iterator c_it2=writeable_Propagators.find(new_propagators[i]);
      //NEMO_ASSERT(c_it2==writeable_Propagators.end(),prefix+"propagator \""+new_propagators[i]+"\" is already part of the list of writeable propagators\n");
      NEMO_ASSERT(new_propagators[i]!=name_of_writeable_Propagator,prefix+"propagator \""+new_propagators[i]+"\" is already writeable propagator\n");
      //then add to readable Propagators
      msg<<prefix<<"adding propagator \""<<new_propagators[i]<<"\" to the readable propagatorlist\n";
      const Propagator* temp_propagator=NULL;
      c_it->second->get_data(new_propagators[i],temp_propagator);
      Propagators[new_propagators[i]]=temp_propagator;
      //add the Propagator type into the Propagator_type map
      fill_Propagator_type_map(new_propagators[i]);
      //enter "this" to Propagator_Callers of the constructor
      /*std::map<std::string, std::set<Simulation*> >* pointer_to_caller_map=NULL;
      get_Caller_list(new_propagators[i],pointer_to_caller_map);
      NEMO_ASSERT(pointer_to_caller_map!=NULL,prefix+"have received NULL for pointer to caller map\n");
      std::map<std::string, std::set<Simulation*> >::iterator it=pointer_to_caller_map->find(new_propagators[i]);
      if(it==pointer_to_caller_map->end())
      {
        std::set<Simulation*> temp_set;
        temp_set.insert(this);
        (*pointer_to_caller_map)[new_propagators[i]]=temp_set;
      }
      else
        it->second.insert(this);*/
      ////add this to the callers of the Propagator constructor
      //std::map<std::string,Simulation*>::iterator constructor_it=pointer_to_Propagator_Constructors->find(new_propagators[i]);
      //NEMO_ASSERT(constructor_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found constructor of \""+new_propagators[i]+"\"\n");
      //std::map<std::string,std::set<Simulation*> >* pointer_to_caller_list=NULL;
      //constructor_it->second->get_data("Propagator_Callers",pointer_to_caller_list);
      //NEMO_ASSERT(pointer_to_caller_list!=NULL,prefix+"received NULL for the caller list\n");
      //std::map<std::string,std::set<Simulation*> >::iterator map_it=pointer_to_caller_list->find(new_propagators[i]);
      //if(map_it==pointer_to_caller_list->end())
      //{
      //  std::set<Simulation*> temp_list;
      //  temp_list.insert(this);
      //  (*pointer_to_caller_list)[new_propagators[i]]=temp_list;
      //}
      //else
      //{
      //  NEMO_ASSERT(map_it!=pointer_to_caller_list->end(),prefix+"have not found propagator \""+new_propagators[i]+
      //              "\" in the caller list of \""+constructor_it->second->get_name()+"\"\n");
      //  msg<<prefix<<"inserting \""<<this->get_name()<<"\" into the caller list of \""<<constructor_it->second->get_name()<<"\"\n";
      //  map_it->second.insert(this);
      //}
    }
  }
  //delete old propagators:
  variable_name = "ignore_readable_Propagators";
  if(options.check_option(variable_name))
  {
    std::vector<std::string> old_propagators;
    options.get_option(variable_name,old_propagators);
    for(unsigned int i=0; i<old_propagators.size(); i++)
    {
      //check that the Propagator is in the readable propagator list
      std::map<std::string, const Propagator*>::iterator it=Propagators.find(old_propagators[i]);
      NEMO_ASSERT(it!=Propagators.end(),prefix+"propagator \""+old_propagators[i]+"\" is not part of the list of readable propagators\n");
      //then delete it from the list
      msg<<prefix<<"erasing propagator \""<<old_propagators[i]<<"\" from the readable propagatorlist\n";
      Propagators.erase(it);
      ////delete "this" from the list of Propagator_Callers and of the Propagator_Caller list of the constructor
      //std::map<std::string, std::set<Simulation*> >::iterator it2=Propagator_Callers.find(old_propagators[i]);
      //NEMO_ASSERT(it2!=Propagator_Callers.end(),prefix+"have not found \""+old_propagators[i]+"\" in the Propagator_Callers map\n");
      //std::set<Simulation*>::iterator it3=it2->second.find(this);
      //NEMO_ASSERT(it3!=it2->second.end(),prefix+"this simulation has already been deleted from the Propagator_Caller list\n");
      //it2->second.erase(it3);

      //std::map<std::string,Simulation*>::iterator constructor_it=pointer_to_Propagator_Constructors->find(old_propagators[i]);
      //NEMO_ASSERT(constructor_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found constructor of \""+old_propagators[i]+"\"\n");
      //std::map<std::string,std::set<Simulation*> >* pointer_to_caller_list=NULL;
      //constructor_it->second->get_data("Propagator_Callers",pointer_to_caller_list);
      //NEMO_ASSERT(pointer_to_caller_list!=NULL,prefix+"received NULL for the caller list\n");
      //std::map<std::string,std::set<Simulation*> >::iterator map_it=pointer_to_caller_list->find(old_propagators[i]);
      //NEMO_ASSERT(map_it!=pointer_to_caller_list->end(),prefix+"have not found propagator \""+old_propagators[i]+
      //            "\" in the caller list of \""+constructor_it->second->get_name()+"\"\n");
      //std::set<Simulation*>::iterator caller_it=map_it->second.find(this);
      //if(caller_it!=map_it->second.end())
      //{
      //  msg<<prefix<<"erasing \""<<this->get_name()<<"\" from the caller list of \""<<constructor_it->second->get_name()<<"\"\n";
      //  map_it->second.erase(caller_it);
      //}
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::create_NemoMeshes(const std::string& mesh_name)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::create_NemoMeshes ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::create_NemoMeshes: ";

  {
    std::vector<std::vector<double> > corner_points;
    std::string variable_name=mesh_name+"_mesh_corners";
    //read mesh corner points from input deck
    if(options.check_option(variable_name))
      options.get_option(variable_name,corner_points);
    else
      throw std::invalid_argument(prefix+"define " + variable_name + " when using the build-in mesh construction\n");

    // check consistent dimensionality of the corner points
    for (unsigned int j=0; j<corner_points.size()-1; j++)
      if (corner_points[j].size()!=corner_points[j+1].size())
        throw std::invalid_argument(prefix+"define " + variable_name + " with a constant dimensionality\n");

    //read in number of mesh points
    unsigned int momentum_points;
    variable_name=mesh_name+"_points";
    if(options.check_option(variable_name))
      momentum_points=options.get_option(variable_name,10);
    else
      throw std::invalid_argument(prefix+"define " + variable_name + " when using the build-in mesh construction\n");

    //if (momentum_mesh_types[momentum_number]==Momentum_1D||momentum_mesh_types[momentum_number]==Momentum_2D||momentum_mesh_types[momentum_number]==Momentum_3D)
    if (momentum_mesh_types.find(mesh_name)->second==NemoPhys::Momentum_1D||momentum_mesh_types.find(mesh_name)->second==NemoPhys::Momentum_2D||
        momentum_mesh_types.find(mesh_name)->second==NemoPhys::Momentum_3D || momentum_mesh_types.find(mesh_name)->second==NemoPhys::Angular_Momentum)
    {
      Kspace* temp_kspace= dynamic_cast<Kspace*> (Momentum_meshes.find(mesh_name)->second);
      temp_kspace->set_mesh_boundary(corner_points);


      //--------------------------------------------------------
      //preliminary implementation of creating a k-mesh!
      //NOTE: to be changed! This allows for debugging, only
      if(momentum_mesh_types.find(mesh_name)->second==NemoPhys::Momentum_1D || momentum_mesh_types.find(mesh_name)->second==NemoPhys::Angular_Momentum)
      {
        temp_kspace->set_segmental_mesh(true);
        //temp_kspace->set_segmental_num_points(0, 2*momentum_points+1);
        temp_kspace->set_segmental_num_points(0, momentum_points);
      }
      if(momentum_mesh_types.find(mesh_name)->second==NemoPhys::Momentum_2D)
      {
        temp_kspace->set_tensorial_mesh(true);
        const vector<vector<double> >& reciprocal_vectors = get_simulation_domain()->get_reciprocal_vectors();

        temp_kspace->set_2D_mesh_of_3D_points(true, reciprocal_vectors[0], reciprocal_vectors[1]);
        const double radius = 1.0;
        temp_kspace->set_tensorial_mesh_refinement(0, -radius, radius, radius/momentum_points); //change: here, non-periodicity has to be in z-direction!
        temp_kspace->set_tensorial_mesh_refinement(1, -radius, radius, radius/momentum_points); //change: here, non-periodicity has to be in z-direction!
      }
      temp_kspace->calculate_mesh();
      Momentum_meshes.find(mesh_name)->second=temp_kspace; //upcasting to NemoMesh
      //--------------------------------------------------------
    }
    else if(momentum_mesh_types.find(mesh_name)->second==NemoPhys::Energy)
    {
      Espace* temp_espace= dynamic_cast<Espace*> (Momentum_meshes.find(mesh_name)->second);
      //NEMO_ASSERT(corner_points.size()==2,prefix+"Define a minimum and a maximum energy\n");
      if(corner_points.size()==2)
      {
        NEMO_ASSERT(corner_points[0].size()==1 && corner_points[1].size()==1,prefix+"wrong dimensionality for maximum/minimum of "+mesh_name+"\n");
        double Emin = std::min(corner_points[0][0],corner_points[1][0]);
        double Emax = std::max(corner_points[0][0],corner_points[1][0]);

        if(!options.check_option(mesh_name+"_exponent"))
        {
          temp_espace->set_Emin(Emin);
          temp_espace->set_Emax(Emax);
          temp_espace->calculate_new_energy_grid();
          energy_integration_weight=(Emax-Emin)/(1.0*momentum_points);
        }
        else
        {
          //energy_integration_weight=(Emax-Emin)/(1.0*momentum_points);
          double exponent=options.get_option(mesh_name+"_exponent",1.0);
          PropagationUtilities::generate_homogeneous_energy_mesh(Emin, Emax, momentum_points, exponent, energy_integration_weight_map, NULL, temp_espace);
        }
      }
      else if(corner_points[0].size()==1)
      {
        NEMO_ASSERT(corner_points[0].size()==1,prefix+"wrong dimensionality of "+mesh_name+"\n");
        temp_espace->set_mesh_boundary(corner_points);
        temp_espace->set_segmental_mesh(true);
        temp_espace->calculate_mesh();

        //temp_espace->add_point(corner_points[0]);
        //   temp_espace->calculate_new_energy_grid();
      }
      else
        throw std::runtime_error(prefix+"Define a minimum and a maximum energy or just a single energy point\n");

      Momentum_meshes.find(mesh_name)->second=temp_espace; //upcasting to NemoMesh
    }
    else
      throw std::invalid_argument(prefix+"there are only Energy and Momentum implemented so far\n");

  }
  produced_a_mesh = true;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::map<std::string, bool>*& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data vector_string ");
  NemoUtils::tic(tic_toc_prefix);
  if(variable=="Propagator_is_initialized")
    data=&Propagator_is_initialized;
  else
    throw std::runtime_error(tic_toc_prefix+"called with unknown variable \""+variable+"\"\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::vector<std::string>& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data vector_string ");
  NemoUtils::tic(tic_toc_prefix);

  NEMO_ASSERT(Parallelizer==this,tic_toc_prefix+"called while this is not the parallelizer");
  NEMO_ASSERT(variable=="mesh_tree_names",tic_toc_prefix+"called unknown variable \""+variable+"\"\n");
  //make sure that the parallelization happened:
  if(!Propagation_is_initialized)
    initialize_Propagation();
  data=Mesh_tree_names;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::map<NemoMesh*, NemoMesh* >& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data map_NemoMesh ");
  NemoUtils::tic(tic_toc_prefix);

  NEMO_ASSERT(Parallelizer==this,tic_toc_prefix+"called while this is not the parallelizer");
  NEMO_ASSERT(variable=="mesh_tree_downtop",tic_toc_prefix+"called unknown variable \""+variable+"\"\n");
  //make sure that the parallelization happened:
  if(!Propagation_is_initialized)
    initialize_Propagation();
  data=Mesh_tree_downtop;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::map<NemoMesh*,std::vector<NemoMesh*> >& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data map_NemoMesh ");
  NemoUtils::tic(tic_toc_prefix);

  NEMO_ASSERT(Parallelizer==this,tic_toc_prefix+"called while this is not the parallelizer");
  NEMO_ASSERT(variable=="mesh_tree_topdown",tic_toc_prefix+"called unknown variable \""+variable+"\"\n");
  //make sure that the parallelization happened:
  if(!Propagation_is_initialized)
    initialize_Propagation();
  data=Mesh_tree_topdown;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const string& variable, NemoMesh*& Mesh_pointer)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data1 ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix="Propagation(\""+this->get_name()+"\")::get_data: ";

  if(!Propagation_is_initialized)
    initialize_Propagation();

  fill_momentum_meshes();
  //check whether there is a momentum with name variable
  std::map<std::string,NemoMesh*>::const_iterator c_it=Momentum_meshes.begin();
  for(; c_it!=Momentum_meshes.end(); ++c_it)
  {
    if(variable.find(c_it->first)!=std::string::npos)
    {
      NEMO_ASSERT(c_it->second!=NULL,prefix+"momentum \""+variable+"\" is not set, yet. Handing over NULL\n");
      Mesh_pointer=c_it->second;
      NemoUtils::toc(tic_toc_prefix);
      return;
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, Propagator*& Propagator_pointer)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data3 ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix="Propagation(\""+this->get_name()+"\")::get_data: ";

  if(!is_Propagation_initialized())
    initialize_Propagation();

  if(name_of_writeable_Propagator.size()>0)
  {
    if(!is_Propagator_initialized(name_of_writeable_Propagator))
      initialize_Propagators(name_of_writeable_Propagator);
    Propagator_pointer=writeable_Propagator;
  }
  else if(Propagators.find(variable)!=Propagators.end())
  {
    std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(variable);
    NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found constructor of \""+variable+"\"\n");
    Simulation* solver = temp_it->second;
    NEMO_ASSERT(solver!=NULL,prefix+"have not found constructor of \""+variable+"\"\n");
    solver->get_data(variable, Propagator_pointer);
  }
  else
    throw std::runtime_error(prefix+"(const std::string& variable, Propagator *& Propagator_pointer) called with unknown parameter \""+variable+"\"\n");
  //if(name_of_writeable_Propagator.size()>0)
  //{
  //  if(!is_Propagator_initialized(name_of_writeable_Propagator))
  //    initialize_Propagators(name_of_writeable_Propagator);
  //  Propagator_pointer=writeable_Propagator;
  //}
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, const Propagator*& Propagator_pointer)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data4 ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::get_data: ";
  if(!Propagation_is_initialized)
    initialize_Propagation();
  if (Propagators.find(variable)!=Propagators.end())
  {
    if(!is_Propagator_initialized(variable))
      initialize_Propagators(variable);
    Propagator_pointer=Propagators.find(variable)->second;
  }
  else
    throw std::runtime_error(prefix+"(const std::string& variable, const Propagator *& Propagator_pointer) called with unknown parameter \""+variable
                             +"\"\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::map<std::vector<NemoMeshPoint>, double>& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data(momentum_resolved_transmission) ");
  NemoUtils::tic(tic_toc_prefix);

  NEMO_ASSERT(variable.find("transmission")!=std::string::npos,"Propagation(\""+get_name()+"\")::get_data called with unknown variable \""+variable+"\"\n");
  if(result.size() != 0)
    result.clear();
  if (momentum_transmission.empty())
  {
    calculate_momentum_resolved_transmission(result);
  }
  else
  {
    result = momentum_transmission;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, const NemoMeshPoint& momentum, double& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data5 ");
  NemoUtils::tic(tic_toc_prefix);

  std::string prefix="Propagation(\""+this->get_name()+"\")::get_data: ";

  if(variable=="integration_weight")
  {
    if(energy_integration_weight_map.size()==0)
      result=energy_integration_weight;
    else
    {
      double this_energy=momentum.get_x();
      NEMO_ASSERT(energy_integration_weight_map.find(this_energy) != energy_integration_weight_map.end(), prefix + "energy not found in energy_integration_weight_map\n");
      result = energy_integration_weight_map[this_energy];

      /*
      //find the name of the energy mesh
      std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
      std::string energy_name=std::string("");
      for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
        if(momentum_name_it->second==NemoPhys::Energy)
          energy_name=momentum_name_it->first;

      std::vector<std::vector<double> > corner_points;
      std::string variable_name=energy_name+"_mesh_corners";
      if(options.check_option(variable_name))
        options.get_option(variable_name,corner_points);//Emin=corner_points[0]...
      //get the exponent
      double exponent=options.get_option(energy_name+"_exponent",1.0);
      unsigned int momentum_points = 10;
      variable_name=energy_name+"_points";
      if(options.check_option(variable_name))
        momentum_points=options.get_option(variable_name,10);
      double temp_index=std::pow((this_energy-corner_points[0][0])/(corner_points[1][0]-corner_points[0][0]),exponent)*momentum_points;

      unsigned int uint_index=(unsigned int) temp_index;
      unsigned int final_index=uint_index;
      if(std::abs(uint_index-temp_index)>std::abs(uint_index-temp_index+1))
        final_index++;
      if(final_index > (energy_integration_weight_vector.size() - 1))
        final_index = energy_integration_weight_vector.size()-1;
      result=energy_integration_weight_vector[final_index];
      */
    }
  }
  else
    throw std::runtime_error(prefix+"called with unknown data-string\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, const std::vector<NemoMeshPoint>* momentum, PetscMatrixParallelComplex*& Matrix,
                           const DOFmapInterface* row_dof_map, const DOFmapInterface* col_dofmap)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data6");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::get_data: ";
  std::string temp_text;
  const std::vector<NemoMeshPoint>* temp_momentum1=momentum;
  translate_momentum_vector(temp_momentum1,temp_text);
  msg<<prefix+temp_text<<"\n";

  //check that the Propagation is initialized...
  if(!Propagation_is_initialized)
    initialize_Propagation(get_name());

  //check that the Propagator is initialized
  if(!is_Propagator_initialized(name_of_writeable_Propagator))
  {
    initialize_Propagators(name_of_writeable_Propagator);
    msg<<prefix<<"Constructor initialized\""<<variable<<"\n";
    NEMO_ASSERT(is_Propagator_initialized(name_of_writeable_Propagator),prefix+"Propagator \""+variable+"\" is not initialized after initialize_Propagators is called\n");
    //temp_propagation->get_data(variable,it->second);
  }

  //2. check that all solvers have done their job - only if the object is not a combined-propagator (used for storage)
  //if(variable.find("combined")==std::string::npos&&!is_ready(variable,*momentum))
  if(!this_is_combine_Propagation&&!is_ready(name_of_writeable_Propagator,*momentum))
  {
    do_solve(writeable_Propagator,*momentum);
  }
  //4. when ready, copy Matrix to the result
  std::vector<NemoMeshPoint> temp_momentum=*momentum;
  std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator momentum_it=writeable_Propagator->propagator_map.find(
        temp_momentum);
  NEMO_ASSERT(momentum_it!=writeable_Propagator->propagator_map.end(),prefix+"momentum not found in propagator \""+variable+"\"\n");
  Matrix=momentum_it->second;
  NEMO_ASSERT(Matrix!=NULL,prefix+"received NULL for the Propagator matrix\n");

  //5. if one or both DOFmap pointer are not NULL, only a submatrix is queried
  //if(row_dof_map!=NULL && row_dof_map!=&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())))
  if(row_dof_map!=NULL && row_dof_map->get_number_of_dofs()!=Hamilton_Constructor->get_dof_map(get_const_simulation_domain()).get_number_of_dofs())
  {
    const DOFmapInterface& large_DOFmap=Hamilton_Constructor->get_dof_map(get_const_simulation_domain());
    //5.1 get the map between this DOFmap and the row_dof_map
    std::vector<int> rows;
    Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,*row_dof_map,rows,get_const_simulation_domain());
    if(rows.size()<=0)
    {
      msg<<prefix+"row indices vector is empty. No place to read out.\nAssuming the full matrix is queried (typical for lead self-energies)\n";
      //large_DOFmap.get_sub_DOF_index_map(&(this->get_const_dof_map(get_const_simulation_domain())),rows_subindex_map);
      Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,(this->get_const_dof_map(get_const_simulation_domain())),rows,get_const_simulation_domain());

    }
    std::vector<int> cols;
    //if col_dofmap is given:
    if(col_dofmap!=NULL)
    {
      Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,*col_dofmap,cols,get_const_simulation_domain());
    }
    else
      cols=rows;

    //5.5 get submatrix according to the input DOFmaps
    delete temporary_sub_matrix;

    if(/*variable.find("combined")!=std::string::npos*/this_is_combine_Propagation && !(Matrix->is_ready()))
    {
      Matrix->get_submatrix(rows,cols, MAT_INITIAL_MATRIX, temporary_sub_matrix);
    }
    else
    {
      temporary_sub_matrix = new PetscMatrixParallelComplex(rows.size(),cols.size(),get_simulation_domain()->get_communicator());
      temporary_sub_matrix->set_num_owned_rows(rows.size());

      for(unsigned int i=0; i<rows.size(); i++)
      {
        if(cols.size()<=rows.size())
          temporary_sub_matrix->set_num_nonzeros_for_local_row(i,cols.size(),0);
        else
          temporary_sub_matrix->set_num_nonzeros_for_local_row(i,rows.size(),cols.size()-rows.size());
      }
      temporary_sub_matrix->allocate_memory();
      Matrix->get_submatrix(rows,cols, MAT_INITIAL_MATRIX, temporary_sub_matrix);
    }
    Matrix=temporary_sub_matrix;

    if(sigma_convert_dense)
      Matrix->matrix_convert_dense();

  }

  NemoUtils::toc(tic_toc_prefix);
}

PetscMatrixParallelComplex* Propagation::check_momentum_do_solve(PetscMatrixParallelComplex*& modes_matrix_pointer, const std::vector<NemoMeshPoint>* momentum)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::check_momentum_do_solve");
  NemoUtils::tic(tic_toc_prefix);
  std::string error_prefix = "Propagation(\""+this->get_name()+"\")::check_momentum_do_solve: ";
  //std::map<std::string, Propagator*>::iterator prop_cit=writeable_Propagators.begin();
  //for(; prop_cit!=writeable_Propagators.end(); prop_cit++)
  {
    //1.1check that the Propagation is initialized...
    if(!Propagation_is_initialized)
      initialize_Propagation(get_name());

    //1.2check that the Propagator is initialized
    //if(!is_Propagator_initialized(prop_cit->first))
    if(!is_Propagator_initialized(name_of_writeable_Propagator))
    {
      //find the constructor and let him initialize the Propagator
      std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator);
      NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),error_prefix+"Constructor of propagator \""+name_of_writeable_Propagator+"\" not found\n");
      Propagation* temp_propagation= dynamic_cast<Propagation*> (temp_it->second);
      NEMO_ASSERT(temp_propagation!=NULL,error_prefix+"dynamic cast failed\n");
      temp_propagation->initialize_Propagators(name_of_writeable_Propagator);
      msg<<error_prefix<<"Constructor initialized\""<<name_of_writeable_Propagator<<"\n";
      NEMO_ASSERT(is_Propagator_initialized(name_of_writeable_Propagator),
                  error_prefix+"Propagator \""+name_of_writeable_Propagator+"\" is not initialized after initialize_Propagators is called\n");
      //temp_propagation->get_data(prop_name,prop_cit->second);
    }
    //1.3 execute do_solve(momentum) if required
    if(!is_ready(name_of_writeable_Propagator,*momentum) || modes_matrix_pointer==NULL ) do_solve(writeable_Propagator,*momentum);
  }
  NemoUtils::toc(tic_toc_prefix);
  return modes_matrix_pointer;
}

void Propagation::get_transfer_matrix_element(PetscMatrixParallelComplex*& output_matrix, const NemoPhys::transfer_matrix_type* type)
{
  if(*type==NemoPhys::into_device_propagating_modes)
    output_matrix=into_device_propagating_modes;
  else if(*type==NemoPhys::into_device_decaying_modes)
    output_matrix=into_device_decaying_modes;
  else if(*type==NemoPhys::out_of_device_propagating_modes)
    output_matrix=out_of_device_propagating_modes;
  else if(*type==NemoPhys::out_of_device_decaying_modes)
    output_matrix=out_of_device_decaying_modes;
  else if(*type==NemoPhys::into_device_propagating_phase)
    output_matrix=into_device_propagating_phase;
  else if(*type==NemoPhys::into_device_decaying_phase)
    output_matrix=into_device_decaying_phase;
  else if(*type==NemoPhys::out_of_device_propagating_phase)
    output_matrix=out_of_device_propagating_phase;
  else if(*type==NemoPhys::out_of_device_decaying_phase)
    output_matrix=out_of_device_decaying_phase;
  else if(*type==NemoPhys::into_device_modes)
    output_matrix=into_device_modes;
  else if(*type==NemoPhys::into_device_phase)
    output_matrix=into_device_phase;
  else if(*type==NemoPhys::out_of_device_modes)
    output_matrix=out_of_device_modes;
  else if(*type==NemoPhys::out_of_device_phase)
    output_matrix=out_of_device_phase;
  else if(*type==NemoPhys::into_device_velocity)
    output_matrix=into_device_velocity;
  else if(*type==NemoPhys::into_device_propagating_velocity)
    output_matrix=into_device_propagating_velocity;
  else if(*type==NemoPhys::out_of_device_velocity)
    output_matrix=out_of_device_velocity;
  else if(*type==NemoPhys::out_of_device_propagating_velocity)
    output_matrix=out_of_device_propagating_velocity;
  else
    throw std::invalid_argument("Propagation(\""+get_name()+"\")::get_transfer_matrix_element called with unknown transfer_matrix_type\n");
}

void Propagation::set_transfer_matrix_element(PetscMatrixParallelComplex*& input_matrix, const NemoPhys::transfer_matrix_type* type)
{
  if(*type==NemoPhys::into_device_propagating_modes)
    into_device_propagating_modes=input_matrix;
  else if(*type==NemoPhys::into_device_decaying_modes)
    into_device_decaying_modes=input_matrix;
  else if(*type==NemoPhys::out_of_device_propagating_modes)
    out_of_device_propagating_modes=input_matrix;
  else if(*type==NemoPhys::out_of_device_decaying_modes)
    out_of_device_decaying_modes=input_matrix;
  else if(*type==NemoPhys::into_device_propagating_phase)
    into_device_propagating_phase=input_matrix;
  else if(*type==NemoPhys::into_device_decaying_phase)
    into_device_decaying_phase=input_matrix;
  else if(*type==NemoPhys::out_of_device_propagating_phase)
    out_of_device_propagating_phase=input_matrix;
  else if(*type==NemoPhys::out_of_device_decaying_phase)
    out_of_device_decaying_phase=input_matrix;
  else if(*type==NemoPhys::into_device_modes)
    into_device_modes=input_matrix;
  else if(*type==NemoPhys::into_device_phase)
    into_device_phase=input_matrix;
  else if(*type==NemoPhys::out_of_device_modes)
    out_of_device_modes=input_matrix;
  else if(*type==NemoPhys::out_of_device_phase)
    out_of_device_phase=input_matrix;
  else if(*type==NemoPhys::into_device_velocity)
    into_device_velocity=input_matrix;
  else if(*type==NemoPhys::into_device_propagating_velocity)
    into_device_propagating_velocity=input_matrix;
  else if(*type==NemoPhys::out_of_device_velocity)
    out_of_device_velocity=input_matrix;
  else if(*type==NemoPhys::out_of_device_propagating_velocity)
    out_of_device_propagating_velocity=input_matrix;
  else
    throw std::invalid_argument("Propagation(\""+get_name()+"\")::set_transfer_matrix_element called with unknown transfer_matrix_type\n");
}

void Propagation::get_data(const std::string& variable, const std::vector<NemoMeshPoint>* momentum, const PetscMatrixParallelComplex*& Matrix,
                           const DOFmapInterface* row_dof_map, const DOFmapInterface* col_dofmap)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data7");
  NemoUtils::tic(tic_toc_prefix);
  std::string error_prefix = "Propagation(\""+this->get_name()+"\")::get_data(const): ";
  PetscMatrixParallelComplex* temp_Matrix = NULL;
  if(variable == "into_device_propagating_modes")
    temp_Matrix = check_momentum_do_solve(into_device_propagating_modes, momentum);
  else if(variable == "into_device_decaying_modes")
    temp_Matrix = check_momentum_do_solve(into_device_decaying_modes, momentum);
  else if(variable == "out_of_device_propagating_modes")
    temp_Matrix = check_momentum_do_solve(out_of_device_propagating_modes, momentum);
  else if(variable == "out_of_device_decaying_modes")
    temp_Matrix = check_momentum_do_solve(out_of_device_decaying_modes, momentum);
  else if(variable == "into_device_propagating_phase")
    temp_Matrix = check_momentum_do_solve(into_device_propagating_phase, momentum);
  else if(variable == "into_device_decaying_phase")
    temp_Matrix = check_momentum_do_solve(into_device_decaying_phase, momentum);
  else if(variable == "out_of_device_propagating_phase")
    temp_Matrix = check_momentum_do_solve(out_of_device_propagating_phase, momentum);
  else if(variable == "out_of_device_decaying_phase")
    temp_Matrix = check_momentum_do_solve(out_of_device_decaying_phase, momentum);
  else if(variable == "into_device_modes")
    temp_Matrix = check_momentum_do_solve(into_device_modes, momentum);
  else if(variable == "into_device_phase")
    temp_Matrix = check_momentum_do_solve(into_device_phase, momentum);
  else if(variable == "out_of_device_modes")
    temp_Matrix = check_momentum_do_solve(out_of_device_modes, momentum);
  else if(variable == "out_of_device_phase")
    temp_Matrix = check_momentum_do_solve(out_of_device_phase, momentum);
  else if(variable == "into_device_velocity")
    temp_Matrix = check_momentum_do_solve(into_device_velocity, momentum);
  else if(variable == "into_device_propagating_velocity")
    temp_Matrix = check_momentum_do_solve(into_device_propagating_velocity, momentum);
  else if(variable == "out_of_device_velocity")
    temp_Matrix = check_momentum_do_solve(out_of_device_velocity, momentum);
  else if(variable == "out_of_device_propagating_velocity")
    temp_Matrix = check_momentum_do_solve(out_of_device_propagating_velocity, momentum);
  else
  {
    std::string temp_text;
    const std::vector<NemoMeshPoint>* temp_momentum1=momentum;
    translate_momentum_vector(temp_momentum1,temp_text);
    msg<<error_prefix+temp_text<<"\n";
    std::map<std::string,Simulation*>::iterator it=pointer_to_Propagator_Constructors->find(variable);
    NEMO_ASSERT(it!=pointer_to_Propagator_Constructors->end(),error_prefix+"no constructor of propagator \""+variable+"\" found\n");
    it->second->get_data(variable,momentum, temp_Matrix,row_dof_map,col_dofmap);
  }
  Matrix = dynamic_cast<const PetscMatrixParallelComplex*> (temp_Matrix);
  NEMO_ASSERT(Matrix!=NULL,error_prefix+"trying to return NULL pointer (" + variable + ")\n");
  NemoUtils::toc(tic_toc_prefix);
  return;
}

void Propagation::allocate_propagator_matrices(const DOFmapInterface* defining_DOFmap, const std::vector<NemoMeshPoint>* momentum)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::allocate_propagator_matrices ");
  NemoUtils::tic(tic_toc_prefix);
  //std::map<std::string, Propagator*>::const_iterator c_it=writeable_Propagators.begin();
  //for(; c_it!=writeable_Propagators.end(); ++c_it)
  {
    //only the respective constructor of each propagator is allowed to set the storage format
    if(pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator)->second==this)
    {
      //how should this Propagator be stored?
      std::string parameter_name=name_of_writeable_Propagator+std::string("_store");
      //writeable_Propagators.find(name_of_writeable_Propagator)->second->set_storage_type(options.get_option(parameter_name,std::string("dense")));

      msg<<"Propagation(\""+this->get_name()+"\"): Propagator \""<<name_of_writeable_Propagator<<"\" will be stored as "<<Propagators.find(
           name_of_writeable_Propagator)->second->get_storage_type()<<std::endl;

      vector<int> local_nonzeros;
      vector<int> nonlocal_nonzeros;
      vector<int> local_rows;
      defining_DOFmap->calculate_non_zeros(&local_nonzeros, &nonlocal_nonzeros );

      defining_DOFmap->get_local_row_indexes(&local_rows);
      unsigned int number_of_rows = defining_DOFmap->get_global_dof_number();
      unsigned int local_number_of_rows =  defining_DOFmap->get_number_of_dofs();

      msg << "Propagation(\""+this->get_name()+"\"): problem size (DOFs): " << number_of_rows << std::endl;
      NEMO_ASSERT(number_of_rows>0, "Propagation(\""+this->get_name()+"\": seem to have 0 DOFs (empty matrix). There is something wrong. Aborting.");

      if(momentum==NULL)
      {
        // ------------------------------
        // allocate all Propagator matrices
        // ------------------------------
        Propagator::PropagatorMap::iterator it=writeable_Propagator->propagator_map.begin();
        for (; it!=writeable_Propagator->propagator_map.end(); ++it)
        {

          it->second = new PetscMatrixParallelComplex(number_of_rows,number_of_rows,
              get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
          it->second->set_num_owned_rows(local_number_of_rows);
          if(writeable_Propagator->get_storage_type()==std::string("dense"))
          {
            it->second->consider_as_full();
          }
          else if(writeable_Propagator->get_storage_type()==std::string("diagonal"))
          {
            for (unsigned int i = 0; i < local_number_of_rows; i++)
              it->second->set_num_nonzeros(local_rows[i],1,0);
          }
          else
            throw std::invalid_argument("Propagation(\""+this->get_name()+"\")::allocate_propagator_matrices: unknown storage format\n");
          msg << "Propagation(\""+this->get_name()+"\"): allocating memory\n";
          it->second->allocate_memory();
          std::map<const std::vector<NemoMeshPoint>,bool>::iterator alloc_it=writeable_Propagator->allocated_momentum_Propagator_map.find(
                it->first);
          NEMO_ASSERT(alloc_it!=writeable_Propagator->allocated_momentum_Propagator_map.end(),"Propagation(\""+name_of_writeable_Propagator+
                      "\")::allocate_propagator_matrices have not found momentum in allocated_momentum_Propagator_map\n");
          alloc_it->second=true;
        }
      }
      else
      {
        for(unsigned int i=0; i<momentum->size(); i++)
          (*momentum)[i].print();
        // ------------------------------
        // allocate just one Propagator matrix
        // ------------------------------
        Propagator::PropagatorMap::iterator it=writeable_Propagator->propagator_map.find(*momentum);
        if(it==writeable_Propagator->propagator_map.end())
        {
          writeable_Propagator->propagator_map[*momentum]=NULL;
          it=writeable_Propagator->propagator_map.find(*momentum);
        }

        it->second = new PetscMatrixParallelComplex(number_of_rows,number_of_rows,
            get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
        it->second->set_num_owned_rows(local_number_of_rows);
        if(writeable_Propagator->get_storage_type()==std::string("dense"))
        {
          it->second->consider_as_full();
        }
        else if(writeable_Propagator->get_storage_type()==std::string("diagonal"))
        {
          for (unsigned int i = 0; i < local_number_of_rows; i++)
            it->second->set_num_nonzeros(local_rows[i],1,0);
        }
        else if(writeable_Propagator->get_storage_type()==std::string("empty"))
        {
          for (unsigned int i = 0; i < local_number_of_rows; i++)
            it->second->set_num_nonzeros(local_rows[i],0,0);
        }
        else
          throw std::invalid_argument("Propagation(\""+this->get_name()+"\")::allocate_propagator_matrices: unknown storage format\n");
        msg << "Propagation(\""+this->get_name()+"\"): allocating memory\n";
        it->second->allocate_memory();
        std::map<const std::vector<NemoMeshPoint>,bool>::iterator alloc_it=writeable_Propagator->allocated_momentum_Propagator_map.find(
              it->first);
        NEMO_ASSERT(alloc_it!=writeable_Propagator->allocated_momentum_Propagator_map.end(),"Propagation(\""+name_of_writeable_Propagator+
                    "\")::allocate_propagator_matrices have not found momentum in allocated_momentum_Propagator_map\n");
        alloc_it->second=true;
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::delete_propagator(const std::string* propagator_name)
{
  //std::string prefix="Propagation(\""+get_name()+"\")::delete_propagator ";
  //purpose of this method is to deallocate the propagator matrices and call the propagator destructor
  if(propagator_name==NULL)
  {
    //loop over writeable_Propagators
    //std::map<std::string,Propagator*>::iterator it=writeable_Propagators.begin();
    //for(; it!=writeable_Propagators.end(); ++it)
    //{
    //  delete_propagator(&(it->first));
    //  writeable_Propagators.erase(it);
    //}
    delete_propagator(&name_of_writeable_Propagator);
    writeable_Propagator=NULL;
    name_of_writeable_Propagator=std::string("");
  }
  else
  {
    //if this is constructor, call destructor of Propagator
    //std::map<std::string,Simulation*>::iterator prop_it=pointer_to_Propagator_Constructors->find(*propagator_name);
    //if(prop_it!=pointer_to_Propagator_Constructors->end())
    //  if(prop_it->second==this)
    {
      if(writeable_Propagator!=NULL)
      {
        delete_propagator_matrices(writeable_Propagator,NULL);
        writeable_Propagator->Propagator::~Propagator();
      }
      ////find the Propagator in the writeable_Propagator list
      //std::map<std::string,Propagator*>::iterator it=writeable_Propagators.find(*propagator_name);
      //if(it!=writeable_Propagators.end())
      //{
      //  delete_propagator_matrices(it->second,NULL);
      //  it->second->Propagator::~Propagator();
      //}
    }
  }
}

void Propagation::delete_propagator_matrices(Propagator* input_Propagator, const std::vector<NemoMeshPoint>* momentum, const DOFmapInterface* row_DOFmap,
    const DOFmapInterface* col_DOFmap)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::delete_propagator_matrices ");
  NemoUtils::tic(tic_toc_prefix);
  msg.set_level(MsgLevel(4));
  if(input_Propagator!=NULL)
  {
    if(momentum!=NULL)
    {
      std::map<const std::vector<NemoMeshPoint>,bool>::iterator it=input_Propagator->allocated_momentum_Propagator_map.find(
            *momentum);
      NEMO_ASSERT(it!=input_Propagator->allocated_momentum_Propagator_map.end(),
                  "Propagation(\""+get_name()+"\")::delete_propagator_matrices have not found momentum in \""+input_Propagator->get_name()+"\"\n");
      if(it->second)
      {
        Propagator::PropagatorMap::iterator it2=input_Propagator->propagator_map.find(*momentum);
        NEMO_ASSERT(it2!=input_Propagator->propagator_map.end(),
                    "Propagation(\""+get_name()+"\")::delete_propagator_matrices have not found momentum in propagator_map of \""
                    +input_Propagator->get_name()+"\"\n");
        if(it2->second!=NULL)
        {
          msg<<"Propagation(\"" << get_name() << "\")::delete_propagator_matrices: "<<it2->second<<"\n";
          if(/*input_Propagator->get_name().find("combine")!=std::string::npos &&*/ row_DOFmap!=NULL)
          {
            ////Yu: this is used for free_memory_of combine solver
            //std::map<unsigned int, unsigned int> rows_superindex_map;
            //std::map<unsigned int, unsigned int>::const_iterator c_it;
            //std::map<unsigned int, unsigned int> cols_superindex_map;

            const DOFmapInterface& large_DOFmap=Hamilton_Constructor->get_dof_map(get_const_simulation_domain());
            //large_DOFmap.get_sub_DOF_index_map(row_DOFmap,rows_superindex_map,true);
            //NEMO_ASSERT(rows_superindex_map.size()>0,"Propagation(\""+get_name()+"\")::delete_propagator_matrices: "+"row super index map is empty. No place to store\n");

            std::vector<int> rows_to_delete;/*(rows_superindex_map.size(),0);
            c_it = rows_superindex_map.begin();
            for(; c_it!=rows_superindex_map.end(); c_it++)
              rows_to_delete[c_it->first]=c_it->second;*/

            Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,*row_DOFmap,rows_to_delete,get_const_simulation_domain());

            std::vector<int> cols_to_delete;
            if(col_DOFmap!=NULL)
            {
              //large_DOFmap.get_sub_DOF_index_map(col_DOFmap,cols_superindex_map,true);
              //NEMO_ASSERT(cols_superindex_map.size()>0,"Propagation(\""+get_name()+"\")::delete_propagator_matrices: "+"col super index map is empty. No place to store\n");

              //cols_to_delete.resize(cols_superindex_map.size());
              //c_it = cols_superindex_map.begin();
              //for(; c_it!=cols_superindex_map.end(); c_it++)
              //  cols_to_delete[c_it->first]=c_it->second;

              Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,*col_DOFmap,cols_to_delete,get_const_simulation_domain());
            }
            else
              cols_to_delete=rows_to_delete;



            //std::cerr<<"Propagation(\""+get_name()+"\")::delete_propagator_matrices: "<<"going to delete rows[0]: "<<rows_to_delete[0]<<" and cols[0]: "<<cols_to_delete[0]<<"\n";
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
            NemoUtils::toc(tic_toc_prefix);
            return void();
          }
          else
          {
            delete it2->second;
            it2->second=NULL;
          }
          input_Propagator->propagator_map[it2->first]=NULL;
        }
        it->second=false;

      }
      else {}
      set_job_done_momentum_map(&(input_Propagator->get_name()), momentum, false);
    }
    else
    {
      msg<<"Propagation(\"" << get_name() << "\")::delete_propagator_matrices: "<<"deleting Propagator \""<<input_Propagator->get_name()<<"\"\n";
      Propagator::PropagatorMap::iterator it2=input_Propagator->propagator_map.begin();
      for(; it2!=input_Propagator->propagator_map.end(); ++it2)
      {
        delete_propagator_matrices(input_Propagator, &(it2->first));
      }
    }
  }
  else
  {
    //1. loop over all writeable Propagators
    //std::map<std::string, Propagator*>::iterator prop_it=writeable_Propagators.begin();
    //for(; prop_it!=writeable_Propagators.end(); prop_it++)
    {
      //2. if this is the constructor of this propagator
      //std::map<std::string,Simulation*>::iterator constructor_it=pointer_to_Propagator_Constructors->find(prop_it->first);
      //NEMO_ASSERT(constructor_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found the constructor of \""+prop_it->first+"\"\n");
      //2.1 call delete_propagator_matrices(this_propagator, NULL);
      //if(constructor_it->second->get_name()==this->get_name())
      if(writeable_Propagator!=NULL)
        delete_propagator_matrices(writeable_Propagator,NULL);
      //else
      //  throw std::runtime_error(prefix+"constructor: "+constructor_it->second->get_name() +" this: "+this->get_name()+"\n");

    }
  }
  //delete_offloadable_matrices();
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::delete_offloadable_matrices()
{
  std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;
  if (offload_solver_initialized && offload_solver->offload)
  {
    std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::const_iterator c_it_momentum_offload_info;
    c_it_momentum_offload_info = offload_solver->offloading_momentum_map.begin();
    if (c_it_momentum_offload_info->second.offload_to == ResourceUtils::MIC)
    {
      if (!NemoMICUtils::first_offload_inverse)
      {
        NemoMICUtils::deallocate_zgesv(c_it_momentum_offload_info->second.coprocessor_num);
        NemoMICUtils::first_offload_inverse = true;
      }
      if (!NemoMICUtils::first_offload_mult)
      {
        NemoMICUtils::deallocate_zgemm(c_it_momentum_offload_info->second.coprocessor_num);
        NemoMICUtils::first_offload_mult = true;
      }
      if (offload_solver->coprocessor_booted)
      {
#if defined(NEMO_MIC) && defined(_OPENMP)
        pcl_lapack::pcl_lapack_shutdown_xeon_phi();
        offload_solver->coprocessor_booted = false;
#else
        throw std::runtime_error("Attempting to call pcl_lapack_shutdown_xeon_phi() without Xeon Phi or OpenMP build");
#endif
      }
    }
  }
}

void Propagation::initialize_Propagation(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::initialize_Propagation ");
  NemoUtils::tic(tic_toc_prefix);
  if(!Propagation_is_initialized)
  {
    Propagation_is_initialized=true;
    msg<<"\n\ninitializing: "<<get_name()<<"\n\n";
    //check if base_init was done (using the variable number_of_momenta)
    if(number_of_momenta<1)
      base_init();
    NEMO_ASSERT(number_of_momenta>0,tic_toc_prefix+"have not found any momenta\n");
    fill_momentum_meshes();
    if(!one_energy_only)
    {
      PropagationUtilities::set_parallel_environment(this, Mesh_tree_names, 
        Mesh_tree_downtop,Mesh_tree_topdown,
        Momentum_meshes,momentum_mesh_types,
        Mesh_Constructors,Parallelizer,get_simulation_domain()->get_one_partition_total_communicator());
      fill_all_momenta();
    }
    initialize_Propagators();
    if(debug_output_job_list)
      update_global_job_list();
  }

  nonblocking_rec_conter=0;
  nonblocking_send_conter=0;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::initialize_Propagation(const std::string propagation_name)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::initialize_Propagation2 ");
  NemoUtils::tic(tic_toc_prefix);
  if(!Propagation_is_initialized&&propagation_name==get_name())
  {
    Propagation_is_initialized=true;
    msg<<"\n\ninitializing: "<<get_name()<<"\n\n";
    //check if base_init was done (using the variable number_of_momenta)
    if(number_of_momenta<1)
      base_init();
    NEMO_ASSERT(number_of_momenta>0,tic_toc_prefix+"have not found any momenta\n");
    //Propagation_is_initialized=true; //set to true here, to avoid calling it recursively
    fill_momentum_meshes();
    if(!one_energy_only)
    {
      PropagationUtilities::set_parallel_environment(this, Mesh_tree_names, 
        Mesh_tree_downtop,Mesh_tree_topdown,
        Momentum_meshes,momentum_mesh_types,
        Mesh_Constructors,Parallelizer,get_simulation_domain()->get_one_partition_total_communicator());
      fill_all_momenta();
    }
    initialize_Propagators();

    if(debug_output_job_list)
      update_global_job_list();
  }
  nonblocking_rec_conter=0;
  nonblocking_send_conter=0;
  NemoUtils::toc(tic_toc_prefix);
}

//void Propagation::read_propagator(const std::string propagator_name,const std::vector<NemoMeshPoint>& momentum,
//    const PetscMatrixParallelComplex*& result)
//{
//  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::read_propagator ");
//  NemoUtils::tic(tic_toc_prefix);
//  std::string prefix = "Propagation(\""+get_name()+"\")::read_propagator ";
//  std::string temp_text;
//  const std::vector<NemoMeshPoint>* temp_momentum=&momentum;
//  translate_momentum_vector(temp_momentum,temp_text);
//  msg<<prefix+temp_text<<"\n";
//  //1. check that the input_propagator* of this is not NULL
//  std::map<std::string, const Propagator*>::iterator temp_it=Propagators.find(propagator_name);
//  NEMO_ASSERT(temp_it!=Propagators.end(),prefix+"Propagator \""+propagator_name+"\" is not known\n");
//  const Propagator* temp_propagator = temp_it->second;
//  if(temp_propagator==NULL)
//  {
//    //1.1 if it is NULL, get the pointer from the constructor using get_data and store it in Propagators
//    std::map<std::string,Simulation*>::iterator it=pointer_to_Propagator_Constructors->find(propagator_name);
//    NEMO_ASSERT(it!=pointer_to_Propagator_Constructors->end(),prefix+"constructor of propagator \""+propagator_name+"\" not found\n");
//    it->second->get_data(propagator_name,temp_propagator);
//    NEMO_ASSERT(temp_propagator!=NULL,prefix+"Pointer to propagator_name is still NULL - after the constructor \""+it->second->get_name()
//        +"\" is called\n");
//    temp_it->second=temp_propagator;
//  }
//  //2. loop over the solvers of this Propagator
//  std::map<std::string, std::set<Simulation*> >* pointer_to_solver_list=NULL;
//  get_Solver_list(propagator_name,pointer_to_solver_list);
//  std::map<std::string, std::set<Simulation*> >::iterator it=pointer_to_solver_list->find(propagator_name);
//  NEMO_ASSERT(it!=pointer_to_solver_list->end(),prefix+"solvers of propagator \""+propagator_name+"\" not found\n");
//  std::set<Simulation*>::iterator solver_it=it->second.begin();
//  for(; solver_it!=it->second.end(); solver_it++)
//  {
//    Propagation* temp_propagation= dynamic_cast<Propagation*> (*solver_it);
//    NEMO_ASSERT(temp_propagation!=NULL,prefix+"dynamic cast failed\n");
//    //2.1 if !job_done_momentum_map of this momentum trigger a do_solve_momentum via get_data
//    if(!temp_propagation->is_ready(propagator_name,momentum))
//    {
//      temp_propagation->get_data(propagator_name, &momentum, result,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
//    }
//    NEMO_ASSERT(temp_propagation->is_ready(propagator_name,momentum),
//        prefix+"propagation \""+temp_propagation->get_name()+"\" still reports not ready, after do_solve has been done\n");
//  }
//  //3. once everything is "ready", copy the matrix to result
//  result=temp_it->second->propagator_map.find(momentum)->second;
//  NemoUtils::toc(tic_toc_prefix);
//}

void Propagation::write_propagator(const std::string propagator_name,const std::vector<NemoMeshPoint>& momentum,
                                   PetscMatrixParallelComplex*& input_matrix)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::write_propagator ");
  NemoUtils::tic(tic_toc_prefix);
  std::string error_prefix = "Propagation(\""+get_name()+"\")::write_propagator ";
  std::string temp_text;
  const std::vector<NemoMeshPoint>* temp_momentum=&momentum;
  translate_momentum_vector(temp_momentum,temp_text);
  msg<<error_prefix+temp_text<<"\n";

  if(symmetrize_results)
  {
    NemoMath::symmetry_type type = NemoMath::antihermitian;
    symmetrize(input_matrix,type);
  }

  //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.find(propagator_name);
  //NEMO_ASSERT(it!=writeable_Propagators.end(),error_prefix+"propagator \""+propagator_name+"\" not found\n");
  NEMO_ASSERT(propagator_name==name_of_writeable_Propagator,error_prefix+"propagator \""+propagator_name+"\" not writeable\n");

  if(use_matrix_0_threshold)
  {
    NemoPhys::Propagator_type temp_type=get_Propagator_type(propagator_name);
    if(temp_type!=NemoPhys::Inverse_Green)
    {
      double threshold=options.get_option("matrix_0_threshold",NemoMath::d_zero_tolerance);
      input_matrix->zero_small_values(threshold);
    }
  }

  //1. check that the input_propagator* is not NULL
  if(writeable_Propagator==NULL)
  {
    //1.1 if it is NULL, get the pointer from the constructor using get_data
    std::map<std::string,Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(propagator_name);
    NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),error_prefix+"no constructor of \""+propagator_name+"\" found\n");
    //PetscMatrixParallelComplex* temp_matrix_pointer=NULL;
    //temp_it->second->get_data(propagator_name,momentum,temp_matrix_pointer);//change to get_data of Propagator*
    temp_it->second->get_data(propagator_name,writeable_Propagator);
    NEMO_ASSERT(writeable_Propagator!=NULL,error_prefix+"propagation \""+temp_it->second->get_name()+"\" is not constructing \""+propagator_name
                +"\" correctly (NULL pointer received)\n");
  }
  //2. overwrite the matrix of propagator(propagator_name) due to momentum with input_matrix
  Propagator::PropagatorMap::iterator prop_map_it=writeable_Propagator->propagator_map.find(momentum);
  if(prop_map_it==writeable_Propagator->propagator_map.end())
    writeable_Propagator->propagator_map[momentum]=NULL;
  prop_map_it=writeable_Propagator->propagator_map.find(momentum);
  //NEMO_ASSERT(prop_map_it!=it->second->propagator_map.end(),error_prefix+"momentum was not found in propagator \""+propagator_name+"\"\n");
  if(name_of_writeable_Propagator.find("constant_eta")==
      std::string::npos)//Do not put to true for constant_eta_scattering_self, since there, we have only one matrix...
    writeable_Propagator->allocated_momentum_Propagator_map[momentum]=true;
  prop_map_it->second=input_matrix;

  std::string file_name;
  const std::vector<NemoMeshPoint>* momentum_point=&momentum;
  translate_momentum_vector(momentum_point,file_name);

  int tr = 0;
  MPI_Comm_rank(holder.total_communicator, &tr);

  //if not combine solver
  //if(get_name().find("combine")==std::string::npos)
  if(!this_is_combine_Propagation)
  {
    if( do_outputL && (tr == 0))
    {
      //std::string temp_suffix = options.get_option("output_suffix",std::string(""));
      std::string temp_suffix = get_output_suffix();
      prop_map_it->second->assemble();
      prop_map_it->second->save_to_matlab_file(propagator_name+file_name+temp_suffix+std::string(".m"));
    }
  }

  ////3. update the job_done_momentum_map for all callers of this Propagator
  //std::map<std::string, std::set<Simulation*> >::iterator call_it=Propagator_Callers.find(propagator_name);
  //if(call_it!=Propagator_Callers.end()&& !ignore_callers_ready_map)
  //{
  //  //msg<<error_prefix<<"calling: "<<call_it->first<<"\n";
  //  //someone is calling this Propagator...
  //  std::set<Simulation*>::iterator set_it=call_it->second.begin();
  //  for(; set_it!=call_it->second.end(); set_it++)
  //  {
  //    //msg<<error_prefix<<"caller: "<<(*set_it)->get_name()<<"\n";
  //    if(*set_it!=this) //add if *set_it!=solver, otherwise this gives infinit loop
  //    {
  //      //msg<<error_prefix<<"getting into the set_job_done_momentum_map of "<<(*set_it)->get_name()<<"\n";
  //      //if(get_name().find("combine")==std::string::npos)
  //      if(!this_is_combine_Propagation)
  //        (*set_it)->set_job_done_momentum_map(NULL, &momentum, false);
  //    }
  //  }
  //}

  //4. check whether the result should be stored in an external ("combined") solver (only if result is not-combined...
  //if(options.check_option("store_result_in"))
  if(store_result_elsewhere)
  {
    //4.a get the name of the Propagator the result should be stored in
    //const std::string combine_propagator_name = options.get_option("store_result_in",std::string(""));
    //4.b get the source of this combine-Propagator
    //Propagation* combine_solver = dynamic_cast<Propagation*> (find_source_of_data(combine_propagator_name));
    NEMO_ASSERT(combine_solver_for_storage!=NULL,error_prefix+"no combine solver set\n");
    Propagation* combine_solver = dynamic_cast<Propagation*> (combine_solver_for_storage);
    NEMO_ASSERT(combine_solver!=NULL,error_prefix+"dynamic cast of the solver that solves \""+combine_solver_for_storage->get_name()+"\"into a Propagation failed\n");
    combine_solver->set_source_simulation_for_storage(this);
    //4.c make sure that the combine_solver knows about the result-propagator
    const InputOptions& combine_solver_option=combine_solver->get_options();
    const InputOptions save_combine_solver_option=combine_solver_option;

    //4.c.1 set the options of the combine solver
    //4.c.2 set the source_matrix_name
    InputOptions& writeable_combine_solver_options=combine_solver->get_reference_to_options();
    writeable_combine_solver_options.set_option("source_matrix_name",propagator_name);
    writeable_combine_solver_options.set_option(propagator_name+"_constructor",this->get_name());
    if(options.check_option("row_domain"))
      writeable_combine_solver_options.set_option("row_domain",options.get_option("row_domain",std::string("")));
    if(options.check_option("column_domain"))
      writeable_combine_solver_options.set_option("column_domain",options.get_option("column_domain",std::string("")));
    //4.c.3 set the source matrix DOFmaps (row and column)
    //default, assuming it is a diagonal matrix block:
    if(!options.check_option("row_DOFmap_solver"))
    {
      writeable_combine_solver_options.set_option("row_DOFmap_solver",Hamilton_Constructor->get_name());
      writeable_combine_solver_options.set_option("column_DOFmap_solver",Hamilton_Constructor->get_name());
    }
    else
    {
      writeable_combine_solver_options.set_option("row_DOFmap_solver",options.get_option("row_DOFmap_solver",std::string("")));
      writeable_combine_solver_options.set_option("column_DOFmap_solver",options.get_option("row_DOFmap_solver",std::string("")));
    }
    NemoPhys::Propagator_type p_type=get_Propagator_type(propagator_name);
    if(p_type==NemoPhys::Fermion_retarded_Green || p_type==NemoPhys::Boson_retarded_Green)
    {
      if(inversion_method == NemoMath::recursive_off_diagonal_inversion)
      {
        //the following is taken from Greensolver.cpp, solve_interdomain_Green_RGF to get the source simulations of the DOFmaps of the storable matrix
        NEMO_ASSERT(options.check_option("target_domain_Hamiltonian_constructor"),error_prefix+"please define \"target_domain_Hamiltonian_constructor\"\n");
        std::string target_H_constructor_name = options.get_option("target_domain_Hamiltonian_constructor",std::string(""));
        writeable_combine_solver_options.set_option("row_DOFmap_solver",target_H_constructor_name);
      }
      combine_solver->set_options(writeable_combine_solver_options);
    }

    //std::string temp_string=combine_solver_option.get_option("source_matrix_name",std::string(""));
    //NEMO_ASSERT(temp_string==propagator_name,error_prefix+"solver \""+combine_propagator_name+"\" does not have \""+propagator_name
    //            +"\" as source matrix\n");

    //4.d Make the combine solver to store the result, i.e. do_solve...
    std::set<Propagator*> temp_set;
    combine_solver->list_writeable_Propagators(temp_set);
    Propagator* temp_propagator=*(temp_set.begin());
    combine_solver->get_data(temp_propagator->get_name(),temp_propagator);
    combine_solver->do_solve(temp_propagator,*momentum_point);

    //4.e restore the input options of the combine solver (NOTE: to be replaced later)
    writeable_combine_solver_options.set_option("row_DOFmap_solver",save_combine_solver_option.get_option("row_DOFmap_solver",std::string("")));
    writeable_combine_solver_options.set_option("column_DOFmap_solver",save_combine_solver_option.get_option("column_DOFmap_solver",std::string("")));
    writeable_combine_solver_options.set_option("source_matrix_name",save_combine_solver_option.get_option("source_matrix_name",std::string("")));
    combine_solver->set_options(writeable_combine_solver_options);
  }
  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::update_otherPropagations_ready_maps(const Propagator* , const std::vector<NemoMeshPoint>& )
{
  throw std::runtime_error("stop - this is obsolete code");
  //std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::update_otherPropagations_ready_maps ");
  //NemoUtils::tic(tic_toc_prefix);
  //std::string prefix = "Propagation(\""+this->get_name()+"\")::update_otherPropagations_ready_maps: ";
  ////set_readymap_false and set_job_done_momentum_map(false) for all writeable_Propagators of those Propagations that ask for this Propagator at this momentum
  ////1. find Propagations that call this Propagator
  //std::map<std::string, std::set<Simulation*> >::iterator caller_it=Propagator_Callers.find(new_resulting_Propagator->get_name());
  //NEMO_ASSERT(caller_it!=Propagator_Callers.end(),
  //            std::string("Propagation(\""+this->get_name()+"\")::update_otherPropagations_ready_maps \""
  //                        +new_resulting_Propagator->get_name()+"\" not found in Propagator caller map\n"));
  //std::set<Simulation*>::iterator set_it=caller_it->second.begin();
  ////2. loop over all writeable Propagators of the Propagations found in 1.
  //for(; set_it!=caller_it->second.end(); ++set_it)
  //{
  //  Propagation* temp_propagation= dynamic_cast<Propagation*> (*set_it);
  //  NEMO_ASSERT(temp_propagation!=NULL,prefix+"dynamic cast failed\n");
  //  if(temp_propagation!=this && temp_propagation->is_Propagation_initialized())
  //  {
  //    //all the Propagators of this simulation are no longer ready
  //    temp_propagation->set_all_ready_false();

  //    //3. set the ready_momentum_Propagator_map for all writeable Propagators of this specific Propagation for this momentum to false
  //    //hereby we assume that all writeable Propagators of this specific Propagation depend on new_resulting_Propagator
  //    std::set<Propagator*> Propagator_list;
  //    temp_propagation->list_writeable_Propagators(Propagator_list);
  //    std::set<Propagator*>::iterator list_it=Propagator_list.begin();
  //    for(; list_it!=Propagator_list.end(); ++list_it)
  //    {
  //      if((*list_it)!=new_resulting_Propagator)
  //      {
  //        temp_propagation->set_job_done_momentum_map(&((*list_it)->get_name()),&momentum,false);
  //      }
  //    }
  //  }
  //}
  //NemoUtils::toc(tic_toc_prefix);
}


void Propagation::list_readable_Propagators(std::set<const Propagator*>& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::list_readable_Propagators ");
  NemoUtils::tic(tic_toc_prefix);
  result.clear();
  std::map<std::string, const Propagator*>::iterator c_it=Propagators.begin();
  for(; c_it!=Propagators.end(); ++c_it)
  {
    //std::map<std::string, Propagator*>::iterator it2=writeable_Propagators.find(c_it->first);
    //if(it2==writeable_Propagators.end())
    if(c_it->first!=name_of_writeable_Propagator)
    {
      if(c_it->second==NULL)
      {
        std::map<std::string,Simulation*>::const_iterator temp_cit=pointer_to_Propagator_Constructors->find(c_it->first);
        NEMO_ASSERT(temp_cit!=pointer_to_Propagator_Constructors->end(),tic_toc_prefix+"have not found constructor of \""+c_it->first+"\"\n");
        temp_cit->second->get_data(c_it->first,c_it->second);
        //c_it->second=it2->second;
      }
      NEMO_ASSERT(c_it->second!=NULL,tic_toc_prefix+"Pointer to propagator \""+c_it->first+"\" is NULL\n");
      result.insert(c_it->second);
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::set_readymap_false(Propagator* input_Propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::set_readymap_false ");
  NemoUtils::tic(tic_toc_prefix);
  set_job_done_momentum_map(&(input_Propagator->get_name()),NULL,false);
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::set_all_ready_false(void)
{
  std::string tic_toc_prefix = "Propagation(\""+tic_toc_name+"\")::set_all_ready_false ";
  NemoUtils::tic(tic_toc_prefix);
  std::map<std::string, bool>::iterator Prop_it=ready_Propagator_map.begin();
  for(; Prop_it!=ready_Propagator_map.end(); ++Prop_it)
    Prop_it->second=false;

  //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.begin();
  //for(; it!=writeable_Propagators.end(); ++it)
  set_readymap_false(writeable_Propagator);
  NemoUtils::toc(tic_toc_prefix);
}

NemoPhys::Propagator_type Propagation::get_Propagator_type(const std::string& Propagator_name)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_Propagator_type ");
  NemoUtils::tic(tic_toc_prefix);
  std::map<std::string, NemoPhys::Propagator_type>::const_iterator c_it=Propagator_types.find(Propagator_name);
  NEMO_ASSERT(c_it!=Propagator_types.end(),"Propagation::get_Propagator_type called with unknown Propagation name: \""+Propagator_name+"\"\n");
  NemoUtils::toc(tic_toc_prefix);
  return c_it->second;

}

void Propagation::print_Propagator(const Propagator* out_propagator) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_Propagator ");
  NemoUtils::tic(tic_toc_prefix);
  Propagator::PropagatorMap::const_iterator c_it=out_propagator->propagator_map.begin();
  for(; c_it!=out_propagator->propagator_map.end(); ++c_it)
  {
    const std::vector<NemoMeshPoint>* momentum_point=&(c_it->first);
    print_Propagator(out_propagator,momentum_point);
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::print_Propagator(const Propagator* out_propagator,const std::vector<NemoMeshPoint>* momentum) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_Propagator2 ");
  NemoUtils::tic(tic_toc_prefix);
  Propagator::PropagatorMap::const_iterator c_it=out_propagator->propagator_map.find(*momentum);
  NEMO_ASSERT(c_it!=out_propagator->propagator_map.end(),"Propagation(\""+this->get_name()+"\")::print_Propagator: momentum not found\n");
  std::string filename;
  translate_momentum_vector(momentum, filename);
  if (get_const_simulation_domain()->get_geometry_replica() /*Nemo::geometry_replica*/ == 0)
  {
    std::string suffix;
    if(options.check_option("output_suffix"))
    {
      suffix = "_";
      //suffix += options.get_option("output_suffix",std::string("failed"));
      suffix += get_output_suffix("failed");
    }
    NEMO_ASSERT(c_it->second!=NULL,tic_toc_prefix+"received NULL for the output matrix\n");
    c_it->second->save_to_matlab_file(out_propagator->get_name()+filename+suffix+std::string(".m"));
    print_diagonal_Propagator(out_propagator,*momentum,"diagonal"+out_propagator->get_name());
  }
  NemoUtils::toc(tic_toc_prefix);
}



void Propagation::print_diagonal_Propagator(const Propagator* out_propagator, const std::vector<NemoMeshPoint>& momentum,
    const std::string& file_name) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_diagonal_Propagator ");
  NemoUtils::tic(tic_toc_prefix);
  if(!no_file_output)
  {
    std::string prefix="Propagation(\""+this->get_name()+"\")::print_diagonal_Propagator ";
    NEMO_ASSERT(Hamilton_Constructor!=NULL, prefix+"Hamilton_Constructor is NULL\n");
    const DOFmapInterface&           dof_map = Hamilton_Constructor->get_dof_map(get_const_simulation_domain());

    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_const_simulation_domain());
    const AtomicStructure& all_atoms   = domain->get_atoms();

    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator c_it=out_propagator->propagator_map.find(
          momentum);
    NEMO_ASSERT(c_it!=out_propagator->propagator_map.end(),prefix+"momentum not found\n");
    {
      const PetscMatrixParallelComplex* matrix_pointer=c_it->second;


      const std::vector<NemoMeshPoint>* momentum_point=&(c_it->first);

      std::string final_filename;
      translate_momentum_vector(momentum_point,final_filename);
      final_filename=file_name+final_filename+".dat";

      std::ofstream out_file;
      out_file.open(final_filename.c_str());
      ConstActiveAtomIterator it  = all_atoms.active_atoms_begin();
      ConstActiveAtomIterator end = all_atoms.active_atoms_end();
      //loop over the atoms
      for ( ; it != end; ++it)
      {
        const AtomStructNode& nd        = it.node();
        std::vector<double> position(3);
        position[0]=nd.position[0];
        position[1]=nd.position[1];
        position[2]=nd.position[2];
        const std::map<short, unsigned int>* temp_atom_dof_map=dof_map.get_atom_dof_map(nd.id);
        std::map<short, unsigned int>::const_iterator it_dofs;
        //loop over the atomic orbitals
        for(unsigned int i=0; i<temp_atom_dof_map->size(); i++)
        {
          it_dofs = temp_atom_dof_map->find(i);
          assert( it_dofs != temp_atom_dof_map->end() );
          out_file<<"orbital: "<<i<<"\t\t"<<position[0]<<"\t\t"<<position[1]<<"\t\t"<<position[2]<<"\t\t";
          std::complex<double> temp_complex=matrix_pointer->get(it_dofs->second,it_dofs->second);
          out_file<<temp_complex.real()<<"\t"<<temp_complex.imag()<<"\n";
        }
      }
      out_file.close();
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}
void Propagation::translate_momentum_vector(const std::vector<NemoMeshPoint>*& momentum_point, std::string& result, const bool for_filename) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::translate_momentum_vector ");
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
void Propagation::translate_momentum_vector(const std::vector<double>* momentum_point, std::string& result) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::translate_momentum_vector2 ");
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

void Propagation::do_solve_lesser_equilibrium(Propagator*& result, const Propagator* retarded_input_Propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve_lesser_equilibrium ");
  NemoUtils::tic(tic_toc_prefix);
  //we assume that the retarded_input_Propagator is indeed a meaningful retarded Propagator and we are in stationary case (otherwise an advanced Propagator is given in addition
  //the result will be calculated as a product of a Fermi or Bose distribution and the imaginary part of the retarded Propagator
  delete_propagator_matrices(result);
  Propagator::PropagatorMap::const_iterator it=retarded_input_Propagator->propagator_map.begin();
  for(; it!=retarded_input_Propagator->propagator_map.end(); it++)
    do_solve_lesser_equilibrium(result,it->first,retarded_input_Propagator);
  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::get_energy_from_momentum(const std::vector<NemoMeshPoint>& momentum_point, double energy)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_energy_from_momentum ");
  NemoUtils::tic(tic_toc_prefix);
  //NEMO_ASSERT(writeable_Propagators.size()>0,tic_toc_prefix+"called with empty writeable_Propagators\n");
  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with empty writeable_Propagator\n");
  //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.begin();
  energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point,writeable_Propagator);
  NemoUtils::toc(tic_toc_prefix);
}

std::vector<double> Propagation::read_kvector_from_momentum(const std::vector<NemoMeshPoint>& momentum_point, const Propagator* input_Propagator,
    const NemoPhys::Momentum_type* input_momentum_type) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::read_kvector_from_momentum ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::read_kvector_from_momentum ";

  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
  std::string k_vector_name=std::string("");
  for (; momentum_name_it!=momentum_mesh_types.end()&&k_vector_name==std::string(""); ++momentum_name_it)
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

void Propagation::do_solve_lesser_equilibrium(Propagator*& result, const std::vector<NemoMeshPoint>& momentum_point,
    const Propagator* retarded_input_Propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve_lesser_equilibrium ");
  NemoUtils::tic(tic_toc_prefix);

  bool complex_energy=complex_energy_used();

  //1. find the Matrix corresponding to the momentum_point
  Propagator::PropagatorMap::iterator lesser_it;
  Propagator::PropagatorMap::const_iterator retarded_it;
  lesser_it=result->propagator_map.find(momentum_point);
  retarded_it=retarded_input_Propagator->propagator_map.find(momentum_point);
  NEMO_ASSERT(lesser_it!=result->propagator_map.end(),
              "Propagation(\""+this->get_name()+"\")::do_solve_lesser_equilibrium have not found momentum in \""+result->get_name()+"\"\n");
  NEMO_ASSERT(retarded_it!=retarded_input_Propagator->propagator_map.end(),
              "Propagation(\""+this->get_name()+"\")::do_solve_lesser_equilibrium have not found momentum in \""+retarded_input_Propagator->get_name()+"\"\n");

  //2. find the energy value
  //Propagator_type lesser_type=Propagator_types.find(result->get_name())->second;
  //3. read in the temperature and the chemical potential
  //double temperature=options.get_option("temperature", NemoPhys::temperature);
  double energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, result);
  bool lead_electron_like = true;
  if(energy<threshold_energy_for_lead)
    lead_electron_like=false;

  //4. fill the Matrix with -fermi(or bose)*(imaginary_part of retarded_input_Propagator)
  lesser_it->second = new PetscMatrixParallelComplex (*(retarded_it->second));
  result->allocated_momentum_Propagator_map.find(lesser_it->first)->second=true;
  //only imaginary part:
  lesser_it->second->imaginary_part();

  bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);

  //lesser_it->second->save_to_matlab_file("test_imaginarypart_matrix.m");
  //if(lesser_type==Fermion_lesser_Green||lesser_type==Fermion_lesser_self)
  if(particle_type_is_Fermion)
  {
    if(!complex_energy)
    {
      double energy=PropagationUtilities::read_energy_from_momentum(this,momentum_point,result);
      if(!differentiate_over_energy)
      {
        if(lead_electron_like)
          *(lesser_it->second) *= std::complex<double>(-NemoMath::fermi_distribution(chemical_potential,temperature_in_eV,energy),0.0);
        else // -(-(1-f))
          *(lesser_it->second) *= std::complex<double>(1.0-NemoMath::fermi_distribution(chemical_potential,temperature_in_eV,energy),0.0);
      }
      else
        *(lesser_it->second) *= std::complex<double>(-NemoMath::dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,
            energy),0.0);  //weight with derivative of the Fermi distribution

    }
    else
    {
      std::complex<double> energy=PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point,result);
      if (!differentiate_over_energy)
      {
        if(lead_electron_like)
          *(lesser_it->second) *= -NemoMath::complex_fermi_distribution(chemical_potential,temperature_in_eV,energy);
        else
          *(lesser_it->second) *= std::complex<double>(1.0,0.0)-NemoMath::complex_fermi_distribution(chemical_potential,temperature_in_eV,energy);
      }
      else
        *(lesser_it->second) *= -NemoMath::complex_dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,energy);
    }
  }
  else // if(lesser_type==Boson_lesser_Green||lesser_type==Boson_lesser_self)
  {
    NEMO_ASSERT(!complex_energy,tic_toc_prefix+"complex energies are not implemented for Bosons yet\n");
    double energy=PropagationUtilities::read_energy_from_momentum(this,momentum_point,result);
    if (!differentiate_over_energy)
      *(lesser_it->second) *= std::complex<double>(-NemoMath::bose_distribution(chemical_potential,temperature_in_eV,energy),0.0);   //weight with Bose distribution
    else
      *(lesser_it->second) *= std::complex<double>(-NemoMath::dbose_distribution_over_dE(chemical_potential,temperature_in_eV,
                              energy),0.0);   //weight with derivative of the Bose distribution
  }
  //else
  //  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_lesser_equilibrium unknown Propagator type\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::do_solve_lesser_equilibrium(Propagator*& output_Propagator,const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{

  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve_lesser_equilibrium ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::do_solver_lesser_equilibrium(): ";
  bool complex_energy=complex_energy_used();
  //1. find the retarded and/or the advanced Green's functions or self-energies in the list of Propagators
  const Propagator* retarded_Propagator=NULL;
  const Propagator* advanced_Propagator=NULL;
  PetscMatrixParallelComplex* temp_advanced=NULL;
  PetscMatrixParallelComplex* temp_retarded=NULL;
  /*if(exact_GR_solver!=NULL)
  {
    exact_GR_solver->get_Propagator(retarded
  }*/
  std::map<std::string, const Propagator*>::iterator it=Propagators.begin();
  for(; it!=Propagators.end(); it++)
  {
    NemoPhys::Propagator_type p_type=Propagator_types.find(it->first)->second;
    //2. find the retarded and/or advanced Green's function
    if(p_type == NemoPhys::Fermion_retarded_Green ||p_type == NemoPhys::Boson_retarded_Green||p_type == NemoPhys::Fermion_retarded_self ||
        p_type == NemoPhys::Boson_retarded_self)
    {
      if(it->second==NULL)
      {
        pointer_to_Propagator_Constructors->find(it->first)->second->get_data(it->first,retarded_Propagator);
        it->second=retarded_Propagator;
      }
      else
        retarded_Propagator=it->second;
    }
    else if(p_type == NemoPhys::Fermion_advanced_Green ||p_type == NemoPhys::Boson_advanced_Green||p_type == NemoPhys::Fermion_advanced_self ||
            p_type == NemoPhys::Boson_advanced_self)
    {
      if(it->second==NULL)
      {
        pointer_to_Propagator_Constructors->find(it->first)->second->get_data(it->first,advanced_Propagator);
        it->second=advanced_Propagator;
      }
      else
        advanced_Propagator=it->second;
    }
  }
  //get the retarded and advanced matrices
  if(retarded_Propagator!=NULL)
  {
    //Simulation* retarded_solver = pointer_to_Propagator_Constructors->find(retarded_Propagator->get_name())->second;
    Simulation* retarded_solver = find_source_of_data(retarded_Propagator);
    retarded_solver->get_data(retarded_Propagator->get_name(),&momentum_point,temp_retarded,
                              &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
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
      Simulation* advanced_solver = find_source_of_data(advanced_Propagator);
      advanced_solver->get_data(advanced_Propagator->get_name(),&momentum_point,temp_advanced,
                                &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    }
  }
  else if(advanced_Propagator!=NULL)
  {
    //Simulation* advanced_solver = pointer_to_Propagator_Constructors->find(advanced_Propagator->get_name())->second;
    Simulation* advanced_solver = find_source_of_data(advanced_Propagator);
    advanced_solver->get_data(advanced_Propagator->get_name(),&momentum_point,temp_advanced,
                              &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
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
      Simulation* retarded_solver = find_source_of_data(retarded_Propagator);
      retarded_solver->get_data(retarded_Propagator->get_name(),&momentum_point,temp_retarded,
                                &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    }
  }
  else
    throw std::invalid_argument(prefix+"at least one, either a retarded or an advanced Propagator must be given as input\n");

  //2. find the energy value - moved to the multiplication block below
  //Propagator_type lesser_type=Propagator_types.find(output_Propagator->get_name())->second;
  //3. read in the temperature and the chemical potential
  //double temperature=options.get_option("temperature", NemoPhys::temperature);
  //double temperature_in_eV=temperature*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
  //double chemical_potential=options.get_option("chemical_potential", 0.0); //default is 0 = value for mass less Bosons;

  //4. fill the Matrix with -fermi(or bose)*(retarded_Propagator-advanced_Propagator) or with the 1-fermi for holes if user specified so
  delete result;
  result = new PetscMatrixParallelComplex (*temp_retarded); //result = PR
  result->add_matrix(*temp_advanced,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));   //result = PR-PA

  //a hack to calculate jacobian
  bool differentiate_over_energy = options.get_option("differentiate_over_energy", false);

  double energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, output_Propagator);
  bool lead_electron_like = true;
  if(energy<threshold_energy_for_lead)
    lead_electron_like = false;

  //if(lesser_type==Fermion_lesser_Green||lesser_type==Fermion_lesser_self)
  if(particle_type_is_Fermion)
  {
    if(!complex_energy)
    {
      double energy=PropagationUtilities::read_energy_from_momentum(this,momentum_point,output_Propagator);
      if(!options.get_option("electron_hole_model",false))
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
          std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
          NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n");
          if(lead_electron_like)
          {
            //electron
            if(options.check_option("analytical_momenta_me"))
            {
              double analytical_momenta_me = options.get_option("analytical_momenta_me",1.0);
              *result *= std::complex<double>(-get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, true,
                                              differentiate_over_energy, false, analytical_momenta_me*NemoPhys::electron_mass),0.0);
            }
            else
              *result *= std::complex<double>(-get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, true,
                                              differentiate_over_energy, false),0.0);
          }
          else //hole
          {
            if(options.check_option("analytical_momenta_mh"))
            {
              double analytical_momenta_mh = options.get_option("analytical_momenta_mh",1.0);
              *result *= std::complex<double>(get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, false,
                                              differentiate_over_energy, false, analytical_momenta_mh*NemoPhys::electron_mass),0.0);
            }
            else
              *result *= std::complex<double>(get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, false,
                                              differentiate_over_energy, false),0.0);
          }
        }
      }
      else //distinguishing electrons and holes via hole factor
      {
        Simulation* hole_factor_constructor = NULL;//Hamilton_Constructor;
        if(options.check_option("hole_factor_dof_constructor"))
        {
          std::string hole_factor_constructor_name = options.get_option("hole_factor_dof_constructor",std::string(""));
          hole_factor_constructor = find_simulation(hole_factor_constructor_name);
          NEMO_ASSERT(hole_factor_constructor!=NULL,get_name() + " set hole factor_constructor");
        }
        else
          hole_factor_constructor = Hamilton_Constructor;
        std::vector<double> temp_data;
        hole_factor_constructor->get_data(std::string("averaged_hole_factor"),energy,temp_data);
        NEMO_ASSERT(temp_data[0]==temp_data[1],prefix+"unequal hole factors received\n");
        double hole_factor=temp_data[0];
        if(!use_analytical_momenta)
        {
          if(!differentiate_over_energy)
            *result*=std::complex<double>((hole_factor-NemoMath::fermi_distribution(chemical_potential,temperature_in_eV,energy)),0.0);
          else
            *result*=std::complex<double>(-NemoMath::dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,energy),0.0);
        }
        else //analytical integration + hole_factor
        {
          std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
          NEMO_ASSERT(momentum_type=="1D"||momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n");
          PetscMatrixParallelComplex temp_matrix(*result);
          if(options.check_option("analytical_momenta_me") && options.check_option("analytical_momenta_mh"))
          {
            double analytical_momenta_me = options.get_option("analytical_momenta_me",1.0);
            double analytical_momenta_mh = options.get_option("analytical_momenta_mh",1.0);
            temp_matrix*=std::complex<double>(hole_factor*get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, false,
                                              differentiate_over_energy, false, analytical_momenta_mh*NemoPhys::electron_mass),0.0);
            *result*=std::complex<double>(-(1.0-hole_factor)*get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, true,
                                          differentiate_over_energy, false, analytical_momenta_me*NemoPhys::electron_mass),0.0);
          }
          else
          {
            temp_matrix*=std::complex<double>(hole_factor*get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, false,
                                              differentiate_over_energy, false),0.0);
            *result*=std::complex<double>(-(1.0-hole_factor)*get_analytically_integrated_distribution(energy, temperature_in_eV, chemical_potential, momentum_type, true,
                                          differentiate_over_energy, false),0.0);
          }
          result->add_matrix(temp_matrix,SAME_NONZERO_PATTERN);
        }
      }
    }
    else //start of complex_energy
    {
      if(options.get_option("analytical_kspace_1D",false) || options.get_option("analytical_kspace_2D",false))
        throw std::runtime_error("Propagation(\""+this->get_name()+"\")::analytical_kspace for complex energy is not implemented\n");

      std::complex<double> energy=PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point,output_Propagator);
      if(!options.get_option("electron_hole_model",false))
      {
        if(!options.get_option("hole_distribution",false))
        {
          if (!differentiate_over_energy)
            *result *= -NemoMath::complex_fermi_distribution(chemical_potential,temperature_in_eV,energy);  //weight with Fermi distribution, result = -Fermi*(PR-PA)
          else
            *result *= -NemoMath::complex_dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,energy);
        }
        else
        {
          if (!differentiate_over_energy)
            *result *= (1.0-NemoMath::complex_fermi_distribution(chemical_potential,temperature_in_eV,
                        energy));  //weight with 1-Fermi distribution, result = 1-Fermi*(PR-PA)
          else
            *result *= -NemoMath::complex_dfermi_distribution_over_dE(chemical_potential,temperature_in_eV,
                       energy);  //weight with 1-Fermi distribution, result = 1-Fermi*(PR-PA)
        }
      }
      else
      {
        std::map<unsigned int, double> threshold_map;
        std::map<unsigned int, double> threshold_map_right;
        Simulation* hole_factor_constructor = NULL;//Hamilton_Constructor;
        if(options.check_option("hole_factor_dof_constructor"))
        {
          std::string hole_factor_constructor_name = options.get_option("hole_factor_dof_constructor",std::string(""));
          hole_factor_constructor = find_simulation(hole_factor_constructor_name);
          NEMO_ASSERT(hole_factor_constructor!=NULL,get_name() + " set hole factor_constructor");
        }
        else
          hole_factor_constructor = Hamilton_Constructor;

        hole_factor_constructor->get_data(std::string("hole_factor_dof"),energy.real(), threshold_map, threshold_map_right);
        std::vector<complex<double> > LDOS;
        std::vector<complex<double> > G_lesser;
        std::vector<int> indices;
        result->get_diagonal(&LDOS);
        //loop over all degrees of freedom
        for(unsigned int j=0; j<LDOS.size(); j++)
        {
          std::complex<double> temp(std::complex<double>(0.0,0.0));
          std::map<unsigned int, double>::const_iterator temp_threshold_it = threshold_map.find(j);
          NEMO_ASSERT(temp_threshold_it!=threshold_map.end(),prefix+"have not found a DOFindex in threshold map\n");
          double hole_factor = temp_threshold_it->second;
          std::complex<double> fermi_factor = NemoMath::complex_fermi_distribution(chemical_potential,temperature_in_eV,energy);
          temp = -LDOS[j]*(1.0-hole_factor)*fermi_factor + LDOS[j]*hole_factor*(1.0-fermi_factor); // result = -A*fermi*(1-C_h) + A*(1-fermi)*C_h
          G_lesser.push_back(temp);
          indices.push_back(j);
        }
        PetscVectorNemo<std::complex<double> > temp_vector(result->get_num_rows(),result->get_num_rows(),result->get_communicator());
        temp_vector.set_values(indices,G_lesser);
        result->matrix_diagonal_shift(temp_vector,INSERT_VALUES);
      }
    }
  }
  else //if(lesser_type==Boson_lesser_Green||lesser_type==Boson_lesser_self)
  {
    NEMO_ASSERT(!complex_energy,prefix+"complex energies for Bosons are not implemented yet\n");
    if(options.get_option("analytical_kspace_1D",false) || options.get_option("analytical_kspace_2D",false))
      throw std::runtime_error("Propagation(\""+this->get_name()+"\")::analytical_kspace for Bosons is not implemented\n");
    double energy=PropagationUtilities::read_energy_from_momentum(this,momentum_point,output_Propagator);
    if (!differentiate_over_energy)
      *result *= std::complex<double>(-NemoMath::bose_distribution(chemical_potential,temperature_in_eV,energy),
                                      0.0);   //weight with Bose distribution, result = -Bose*(PR-PA)
    else
      *result *= std::complex<double>(-NemoMath::dbose_distribution_over_dE(chemical_potential,temperature_in_eV,energy),
                                      0.0);   //weight with Bose distribution, result = -Bose*(PR-PA)
  }
  //else
  //  throw std::runtime_error("Propagation(\""+this->get_name()+"\")::do_solve_lesser_equilibrium unknown Propagator type\n");

  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum_point, true);
  if(retarded_Propagator==NULL)
    delete temp_retarded;
  if(advanced_Propagator==NULL)
    delete temp_advanced;
  PropagationUtilities::supress_noise(this,result);
  NemoUtils::toc(tic_toc_prefix);

}

void Propagation::integrate_diagonal(Simulation* source_of_data, NemoPhys::Propagator_type input_type, const std::string& data_name, 
                                     bool get_energy_resolved_data, bool get_k_resolved_data, bool& get_energy_resolved_nonrectangular_data, bool density_by_hole_factor,
                                     NemoPhys::Spin_element_type spin_element)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::integrate_diagonal ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation::(\""+this->get_name()+"\")::integrate_diagonal: ";

  bool complex_energy = complex_energy_used();
  std::string name_of_propagator;
  if(writeable_Propagator!=NULL)
    name_of_propagator=name_of_writeable_Propagator;
  else if(options.check_option("density_of"))
    name_of_propagator=options.get_option("density_of",std::string(""));
  //NemoPhys::Propagator_type input_type = get_Propagator_type(name_of_propagator);
  //1. we assume that data_name is the name of a propagator; use get_data to get a pointer to the Propagator
  NemoUtils::tic(tic_toc_prefix+"1.");
  Propagator* Propagator_pointer;
  std::string prefix_get_data1="Propagation::(\""+this->get_name()+"\")::integrate_diagonal: get_data 1 ";
  NemoUtils::tic(prefix_get_data1);
  source_of_data->get_data(data_name,Propagator_pointer);
  NemoUtils::toc(prefix_get_data1);
  Propagator::PropagatorMap::const_iterator momentum_c_it=Propagator_pointer->propagator_map.begin();

  PetscMatrixParallelComplex* temp_matrix=NULL;
  std::string prefix_get_data2="Propagation::(\""+this->get_name()+"\")::integrate_diagonal: get_data 2 ";
  NemoUtils::tic(prefix_get_data2);
  source_of_data->get_data(data_name,&(momentum_c_it->first),temp_matrix,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
  //GreensfunctionInterface* green_source=dynamic_cast<GreensfunctionInterface*>(source_of_data);
  //NEMO_ASSERT(green_source!=NULL, prefix + source_of_data->get_name() + "is not a GreensfunctionInterface\n");
  //green_source->get_Greensfunction(momentum_c_it->first, temp_matrix, &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())),
  //    &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())), input_type);


  NemoUtils::toc(prefix_get_data2);
  NEMO_ASSERT(temp_matrix!=NULL,prefix+"have received NULL for the matrix pointer\n");
  //if(temp_matrix->if_container())
  //  temp_matrix->assemble();

  std::vector<std::complex<double> > temp_result(temp_matrix->get_num_cols(),std::complex<double>(0.0,0.0));
  std::string momentum_loop_tic = "Propagation(\""+tic_toc_name+"\")::integrate_diagonal momentum loop ";
  NemoUtils::toc(tic_toc_prefix+"1.");
  NemoUtils::tic(momentum_loop_tic);

  //bool get_energy_resolved_data=options.get_option(data_name+"_energy_resolved_output",false) ||
  //                              options.get_option(data_name + "_energy_resolved",bool(false)) || options.get_option("energy_resolved_density",false);
  //bool get_energy_resolved_nonrectangular_data = false;
  /*if(get_energy_resolved_data)
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
  }*/
  //1.1 if energy resolved data is wanted, prepare the required container and the all_energies
  std::vector<double> all_energies;
  std::vector<std::complex<double> > all_complex_energies;
  std::map<double, int> translation_map_energy_index;
  std::map<std::complex<double>, int, Compare_complex_numbers> translation_map_complex_energy_index;
  std::vector<vector<double> > all_kvectors;
  std::map<vector<double>, set<double> > all_energies_per_kvector;
  std::map<vector<double>, int > translation_map_kvector_index;
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
        temp_energy = PropagationUtilities::read_energy_from_momentum(this,*c_it,temp_propagator);
        temp_all_energies.insert(temp_energy);
      }
      else
      {
        std::complex<double> temp_energy(0,0);
        Propagator * temp_propagator=NULL;
        source_of_data->get_data(data_name,temp_propagator);
        temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this,*c_it, temp_propagator);
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
      temp_energy = PropagationUtilities::read_energy_from_momentum(this, *c_it, temp_propagator);
      std::vector<double> momentum_point = PropagationUtilities::read_kvector_from_momentum(this, *c_it, temp_propagator);
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
       std::vector<double> momentum_point=PropagationUtilities::read_kvector_from_momentum(this,*c_it, temp_propagator);
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
      std::string prefix_get_data_loop="Propagation::(\""+this->get_name()+"\")::integrate_diagonal: get_data in loop ";
      NemoUtils::tic(prefix_get_data_loop);

      if(!mesh_options.get_option(std::string("non_rectangular"),false))
      {
        if(complex_energy && momentum_mesh_name.find("energy") != std::string::npos)
        {
          NEMO_ASSERT(input_type == NemoPhys::Fermion_lesser_Green, prefix + "input type is not Fermion lesser Green for complex energy\n");
          std::string energy_mesh_type = mesh_options.get_option("energy_mesh_type", std::string(""));
          if(energy_mesh_type == "neq")
            mesh_constructor->get_data("integration_weight", (momentum_c_it->first)[i], temp_complex);
          //if(input_type==NemoPhys::Fermion_retarded_Green)
          //{
          //  bool isPole=0;
          //  std::complex<double> temp_complex_energy((momentum_c_it->first)[i].get_x(),(momentum_c_it->first)[i].get_y());
          //  mesh_constructor->get_data(temp_complex_energy, isPole);
          //  if(isPole)
          //  {
          //    temp_complex = 0.0;
          //  }
          //  else
          //  {
          //    mesh_constructor->get_data("integration_weight",(momentum_c_it->first)[i],temp_complex);
          //  }
          //}
          //else
          //{
          //  mesh_constructor->get_data("integration_weight",(momentum_c_it->first)[i],temp_complex);
          //}
        }
        else
        {
          mesh_constructor->get_data("integration_weight",(momentum_c_it->first)[i],temp_double);
          if (complex_energy)
            temp_complex = std::complex<double>(temp_double,temp_complex.imag());
        }
      }
      else
      {
        //3.1 if integration weight for energy
        if(Propagator_pointer->momentum_mesh_names[i].find("energy")!=std::string::npos)
        {
          //get the k-point from the momentum
          std::vector<double> temp_vector=PropagationUtilities::read_kvector_from_momentum(this, momentum_c_it->first, Propagator_pointer);
          NemoMeshPoint temp_momentum(0,temp_vector);
          std::vector<NemoMeshPoint> temp_vector_momentum(1,temp_momentum);
          //mesh_constructor->get_data("integration_weight",only-k (1dvector<NemoMeshPoint>,only energy NemoMeshPoint,temp_double);
          if(complex_energy)
          {
            NEMO_ASSERT(input_type == NemoPhys::Fermion_lesser_Green, prefix + "input type is not Fermion lesser Green for complex energy\n");
            std::string energy_mesh_type = mesh_options.get_option("energy_mesh_type", std::string(""));
            if (energy_mesh_type == "neq")
              mesh_constructor->get_data("integration_weight", (momentum_c_it->first)[i], temp_complex);
            //if (input_type == NemoPhys::Fermion_retarded_Green)
            //{
            //  bool isPole = 0;
            //  std::complex<double> temp_complex_energy((momentum_c_it->first)[i].get_x(), (momentum_c_it->first)[i].get_y());
            //  mesh_constructor->get_data(temp_complex_energy, isPole);
            //  if (isPole)
            //  {
            //    temp_complex = 0.0;
            //  }
            //  else
            //  {
            //    mesh_constructor->get_data("integration_weight", temp_vector_momentum, (momentum_c_it->first)[i], temp_complex);
            //  }
            //}
            //else
            //  mesh_constructor->get_data("integration_weight", temp_vector_momentum, (momentum_c_it->first)[i], temp_complex);
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
            temp_complex = std::complex<double>(temp_double,temp_complex.real());
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
    source_of_data->get_data(data_name,&(momentum_c_it->first),temp_matrix,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    NemoUtils::toc(tic_toc_prefix+"4.");
    NemoUtils::tic(tic_toc_prefix+"5.");
    //5. get the diagonal of the matrix of this respective momentum
    //if(temp_matrix->if_container())
    //  temp_matrix->assemble();
    
    //take diagonal in the default (e.g. density) calculation
    if (spin_element == NemoPhys::Nospin)
    {
      temp_matrix->get_diagonal(&diagonal);
      NEMO_ASSERT(temp_result.size() == diagonal.size(), prefix + "mismatch of diagonal size\n");
    }
    else
    {
      //get some spin-offdiagonals
      extract_spin_diagonal(temp_matrix, diagonal, spin_element);
    }

    
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
        double energy = PropagationUtilities::read_energy_from_momentum(this,momentum_c_it->first,Propagator_pointer);
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
      unsigned int number_of_DOFs = Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()).get_number_of_dofs();
      if (number_of_DOFs == temp_result.size() && options.get_option("transform_density",false))
      {
        temp_result_transformed.clear();
        cplx multiplication_factor(1.0, 0.0);
        //don't multiply by atom volume here. The transformation to density is done in Transformation
        TransformationUtilities::transform_vector_orbital_to_atom_resolved(Hamilton_Constructor, multiplication_factor, temp_result,
            temp_result_transformed, false /*multiply_by_atom_volume*/);
      }

      if(get_energy_resolved_data)
      {
        if(complex_energy)
        {
          std::complex<double> energy = PropagationUtilities::read_complex_energy_from_momentum(this,momentum_c_it->first,Propagator_pointer);
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
          double energy = PropagationUtilities::read_energy_from_momentum(this,momentum_c_it->first,Propagator_pointer);
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
        double energy = PropagationUtilities::read_energy_from_momentum(this,momentum_c_it->first,Propagator_pointer);
        vector<double> k_vector = PropagationUtilities::read_kvector_from_momentum(this,momentum_c_it->first,Propagator_pointer);
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
          vector<double> k_vector = read_kvector_from_momentum(momentum_c_it->first,Propagator_pointer);
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
  
  std::string mpi_tic_tocs = "Propagation(\""+tic_toc_name+"\")::integrate_diagonal2 MPI Barrier/Reduce ";
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
// this function work as MPI_Reduce but without need for creat communicator
int Propagation::MPI_NEMO_Reduce(const std::vector<std::complex<double> >* sendbuf_c, std::vector<std::complex<double> >*&  recvbuf, int count,
                                 MPI_Datatype datatype,
                                 const MPI_Comm& integration_comm, const int root_rank, std::vector <int>& group)
{


  std::string prefix="Propagation::(\""+this->get_name()+"\")::MPI_NEMO_Reduce: ";

  int my_rank;
  MPI_Comm_rank(integration_comm,&my_rank);
  int rank_count = group.size();
  //handle the case of calculating integral of loacal E-K points that will not need communication
  //handle the communication pair that contain oen rank
  //copy the data from the send buffer to the receive buffer and return.
  if(((rank_count==1)&&(my_rank==group[0]))||(rank_count==0))
  {
    recvbuf->resize(count );
    for(int i=0; i<count; i++)
    {
      (*recvbuf)[i]=(*sendbuf_c)[i];
    }
    return 1;
  }
  NEMO_ASSERT(root_rank==group[0],prefix+"The first element of group of ranks is not the root rank\n");
  //handle normal communication pairs
  std::vector<std::complex<double> > sendbuf_Obj;
  std::vector<std::complex<double> >* sendbuf=&sendbuf_Obj;
  sendbuf->resize(count);
  //copy the dat to a new buffer where the summation operation will be done
  //without chaning the input send buffer.
  for(int i=0; i<count; i++)
  {
    (*sendbuf)[i]=(*sendbuf_c)[i];
  }
  //iterate over the senders to send data.
  std::vector< int >::iterator rank_sender_it=group.begin();
  rank_sender_it++;
  for(; rank_sender_it != group.end(); rank_sender_it++)
  {
    if(*rank_sender_it==my_rank)
    {
      MPI_Send((&(*sendbuf)[0]) ,  count, datatype, group[0],0, integration_comm);
    }
  }
  //iterate over the sender again to receive the data at the receiver
  rank_sender_it=group.begin();
  rank_sender_it++;
  if(my_rank==group[0])
  {
    recvbuf->resize(count);
    for(; rank_sender_it != group.end(); rank_sender_it++)
    {
      MPI_Recv((&(*recvbuf)[0]),  count, datatype,*rank_sender_it,0, integration_comm, MPI_STATUS_IGNORE);
      //do the integration operation
      for(int i=0; i<count; i++)
      {
        (*sendbuf)[i]+=(*recvbuf)[i];
      }
    }
    //at the end put the results on the receieve buffer again
    for(int i=0; i<count; i++)
    {
      (*recvbuf)[i]=(*sendbuf)[i];
    }
  }

  return 1;
}



// this function work as MPI_Reduce but without need for creat communicator
int Propagation::MPI_NEMO_Reduce_nonblocking(const std::vector<std::complex<double> >* sendbuf_c, std::vector<std::complex<double> >*&  recvbuf, int count,
                                 MPI_Datatype datatype,
                                 const MPI_Comm& integration_comm, const int root_rank, std::vector <int>& group)
{


  std::string prefix="Propagation::(\""+this->get_name()+"\")::MPI_NEMO_Reduce_nonblocking: ";

  int my_rank;
  MPI_Comm_rank(integration_comm,&my_rank);
  int rank_count = group.size();
  //handle the case of calculating integral of loacal E-K points that will not need communication
  //handle the communication pair that contain oen rank
  //copy the data from the send buffer to the receive buffer and return.
  if(((rank_count==1)&&(my_rank==group[0]))||(rank_count==0))
  {
    recvbuf->resize(count );
    for(int i=0; i<count; i++)
    {
      (*recvbuf)[i]=(*sendbuf_c)[i];
    }
    return 1;
  }
  NEMO_ASSERT(root_rank==group[0],prefix+"The first element of group of ranks is not the root rank\n");



//iterate to calculate teh tag of the message = sum of the ranks
 int tag=0;
 for(unsigned int i=0;i< group.size();i++)
 {
   tag+=group[i];
 }
  //iterate over the senders to send data.
  std::vector< int >::iterator rank_sender_it=group.begin();
  for(; rank_sender_it != group.end(); rank_sender_it++)
  {
    if(*rank_sender_it==my_rank)
    {
      MPI_Request send_req;
      map_nonblocking_send_buf.insert(std::pair<int, std::vector<std::complex<double> > > (nonblocking_send_conter,*sendbuf_c));
      MPI_Isend(&(map_nonblocking_send_buf[nonblocking_send_conter][0]),  count, datatype, group[0],tag, integration_comm,&send_req);
      map_nonblocking_send_req.insert(std::pair<int,MPI_Request> (nonblocking_send_conter,send_req));
      nonblocking_send_conter++;
    }
  }
  //iterate over the sender again to allocate teh receiver buffer
  rank_sender_it=group.begin();
  rank_sender_it++;
  if(my_rank==group[0])
  {
    recvbuf->resize(count,std::complex<double>(0,0));
    if(map_nonblocking_rec_buf.find(nonblocking_rec_conter)!=map_nonblocking_rec_buf.end())
    {
      map_nonblocking_rec_group[nonblocking_rec_conter]=group;
      map_nonblocking_rec_buf[nonblocking_rec_conter]=recvbuf;
    }
    else
    {
      map_nonblocking_rec_group.insert(std::pair<int, std::vector<int> > (nonblocking_rec_conter,group));
      map_nonblocking_rec_buf.insert(std::pair<int, std::vector<std::complex<double> >* > (nonblocking_rec_conter,recvbuf));
    }
    nonblocking_rec_conter++;
  }

  return 1;
}

  // this function complete the send and recieve operations
int Propagation::MPI_NEMO_Reduce_nonblocking_finalize( MPI_Datatype datatype,const MPI_Comm& integration_comm)
{

  std::map<int, std::vector<int >  >::iterator  map_nonblocking_rec_group_it=map_nonblocking_rec_group.begin();
  std::map<int, std::vector<std::complex<double> > * >::iterator  map_nonblocking_rec_buf_it=map_nonblocking_rec_buf.begin();

  std::vector<std::complex<double> > temp;
  // do the recieve operation
  for(; map_nonblocking_rec_group_it != map_nonblocking_rec_group.end(); map_nonblocking_rec_group_it++,map_nonblocking_rec_buf_it++)
  {
    temp.resize( map_nonblocking_rec_buf_it->second->size(),std::complex<double>(0,0));
    std::vector <int >::iterator rank_sender_it=map_nonblocking_rec_group_it->second.begin();
    for(; rank_sender_it != map_nonblocking_rec_group_it->second.end(); rank_sender_it++)
    {
      MPI_Request send_req;
      int tag=0;
        int index = map_nonblocking_rec_buf_it->first;
        for(unsigned int i=0;i< map_nonblocking_rec_group[index].size();i++)
        {
          tag+=map_nonblocking_rec_group[index][i];
        }
        MPI_Irecv(&(temp[0]),  map_nonblocking_rec_buf_it->second->size(), datatype,*rank_sender_it,tag, integration_comm, &send_req);
        MPI_Status status;
        MPI_Wait(&send_req, &status);
        for(unsigned int i =0; i<map_nonblocking_rec_buf_it->second->size();i++)
        {
          (*map_nonblocking_rec_buf_it->second)[i]+=temp[i];
        }
    }
  }
  std::map<int, MPI_Request >::iterator  map_nonblocking_send_req_it= map_nonblocking_send_req.begin();
  //make sure that the nonblocking send finished.
  for(;map_nonblocking_send_req_it!=map_nonblocking_send_req.end();map_nonblocking_send_req_it++)
  {
    MPI_Status status;
    MPI_Wait(&(map_nonblocking_send_req_it->second), &status);
  }
  //clear all bufferes
  map_nonblocking_rec_buf.clear();
  map_nonblocking_send_req.clear();
  map_nonblocking_send_buf.clear();
  nonblocking_send_conter = 0;
  nonblocking_rec_conter = 0;
  return 1;
}

int Propagation::MPI_NEMO_Reduce_Tree(const std::vector<std::complex<double> >* sendbuf_c, std::vector<std::complex<double> >*&  recvbuf, int count, MPI_Datatype datatype,
      const MPI_Comm& integration_comm, const int /*root_rank*/, std::vector <int> & group)
{
  int my_rank;
  MPI_Comm_rank(integration_comm,&my_rank);
  int rank_count = group.size();
  //handel the case of calculating integral of loacal E-K points that will not need communication
  //handle the communication pair that contain oen rank
  //copy the data from the send buffer to the receive buffer and return.
  if(((rank_count==1)&&(my_rank==group[0]))||(rank_count==0))
  {
    recvbuf->resize(count );
    for(int i=0;i<count;i++)
    {
      (*recvbuf)[i]=(*sendbuf_c)[i];
    }
    return 1;
  }
  std::string type_of_MPI_reduce=options.get_option("type_of_MPI_reduce",std::string("synchronous_split"));
  MPI_Request request;
  //handle normal communication pairs
  std::vector<std::complex<double> > sendbuf_Obj;
  std::vector<std::complex<double> > *sendbuf=&sendbuf_Obj;
  sendbuf->resize(count);
  std::set<int> rowIndex;
  for (int j = 0; j < rank_count; j++)
  {
    rowIndex.insert(j);
  }
  for(int i=0;i<count;i++)
  {
    (*sendbuf)[i]=(*sendbuf_c)[i];
  }
  std::set< int >::iterator rank_sender_it=rowIndex.begin();
  rank_sender_it++;
  std::set< int >::iterator rank_receiver_it=rowIndex.begin();
  int tag=0;
  for(; rank_count > 1;)
  {
    for(;((rank_receiver_it != rowIndex.end())&&(rank_sender_it != rowIndex.end()));)
    {
      if(my_rank==group[*rank_sender_it])
      {
        if(type_of_MPI_reduce=="asynchronous_split_less_tree")
        {
          MPI_Send((&(*sendbuf)[0]) ,  count, datatype, group[*rank_receiver_it],tag, MPI_COMM_WORLD);
        }
        else
        {
          MPI_Isend((&(*sendbuf)[0]) ,  count, datatype, group[*rank_receiver_it],tag, MPI_COMM_WORLD,&request);
        }
      }
      else if (my_rank==group[*rank_receiver_it])
      {
        if(tag==0)
        {
          recvbuf->resize(count);
        }
        MPI_Recv((&(*recvbuf)[0]),  count, datatype, group[*rank_sender_it],tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i=0;i<count;i++)
        {
          (*sendbuf)[i]+=(*recvbuf)[i];
        }
      }
      rank_receiver_it++;
      rank_receiver_it++;
      rowIndex.erase(rank_sender_it);
      rank_sender_it=rank_receiver_it;
      rank_sender_it++;
      rank_count--;
    }
    rank_sender_it=rowIndex.begin();
    rank_sender_it++;
    rank_receiver_it=rowIndex.begin();
    tag++;
  }
  if(my_rank!=group[0] )
  {
    recvbuf->clear();
  }
  else
  {
    for(int i=0;i<count;i++)
    {
      (*recvbuf )[i]=(*sendbuf)[i];
    }
  }

  if (options.get_option("debug_output_job_list", false)&&!no_file_output)
  {
    static int group_counter=0;
    std::stringstream convert_to_string;
    convert_to_string << my_rank;
    ofstream outfile;
    if(group.size()>0)//if (options.get_option("debug_output_job_list", false))
    {

      std::string filename;

      filename = get_name() + "_MPI_Reduce_" + convert_to_string.str()
                                    + ".dat";
      outfile.open(filename.c_str(),std::fstream::in | std::fstream::out | std::fstream::app);
      outfile << "type_of_MPI_reduce "<< type_of_MPI_reduce<< "\n";
      outfile << "group number  "<< group_counter<< "\n";
      outfile << "array size "<< count<< "\n";

      for (unsigned int i = 0; i < group.size();i++)
      {
        outfile << "R"<< group[i]<< "  \t";
      }
      outfile <<"\n";
    }
    group_counter++;
    outfile.close();
  }
  return 1;
}
double Propagation::polar_optical_prefactor(const std::vector<NemoMeshPoint>& momentum1,
           const std::vector<NemoMeshPoint>& momentum2,
           unsigned int x1, unsigned int x2)
{
  NEMO_ASSERT(false,"Propagation::polar_optical_prefactor is not implemmented\n");
  //this code just to turn off GCC compiler warning.
  x1=1;
  x2=2;
  std::vector<NemoMeshPoint> momentum3=momentum1;
  momentum3=momentum2;
  return -1;
}

Simulation* Propagation::find_source_of_data(const Propagator* input_propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::find_source_of_data ");
  NemoUtils::tic(tic_toc_prefix);
  Simulation* result_simulation = find_source_of_data(input_propagator->get_name());
  NemoUtils::toc(tic_toc_prefix);
  return result_simulation;
}

Simulation* Propagation::find_source_of_data(const std::string& inputname)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::find_source_of_data2 ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+get_name()+"\")::find_source_of_data: ";
  if(inputname.size()==0)
    NEMO_ASSERT(inputname.size()>0,prefix+"empty string received\n");
  Simulation* result;
  //1. if solver of inputname is given -> return this solver
  std::string variable = inputname+"_solver";
  if(options.check_option(variable))
  {
    std::string solver_name=options.get_option(variable,std::string(""));
    result=this->find_simulation(solver_name);
    NEMO_ASSERT(result!=NULL,prefix+"simulation \""+solver_name+"\" not found\n");
  }
  else
  {
    //2. otherwise, return the constructor
    std::map<std::string,Simulation*>::const_iterator c_it=pointer_to_Propagator_Constructors->find(inputname);
    if(c_it==pointer_to_Propagator_Constructors->end())
    {
      std::map<std::string,Simulation*>::const_iterator c_it2=pointer_to_Propagator_Constructors->begin();
      std::string temp_name;
      NEMO_ASSERT(options.check_option(inputname+"_constructor"),prefix+"please define \""+inputname+"_constructor\"\n");
      temp_name=options.get_option(inputname+"_constructor",std::string(""));
      result=find_simulation(temp_name);
      NEMO_ASSERT(result!=NULL,prefix+"have not found simulation \""+temp_name
                  +"\" (constructor of \""+inputname+"\")\n");
    }
    else
      result=c_it->second;
  }
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

GreensfunctionInterface* Propagation::find_source_of_Greensfunction(const std::string& inputname)
{
  GreensfunctionInterface* temp = dynamic_cast<GreensfunctionInterface*>(find_source_of_data(inputname));
  NEMO_ASSERT(temp!=NULL,"Propagation(\""+get_name()+"\")::find_source_of_Greensfunction "+inputname+" is not Greensfunction compatible "+
              "(type:\""+find_source_of_data(inputname)->get_type()+"\"\n");
  return temp;
}

void Propagation::find_solver_and_readable_propagator_of_type(const NemoPhys::Propagator_type input_type,const Propagator*& result_propagator,
    Simulation*& solver)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::find_solver_and_readable_propagator_of_type ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+get_name()+"\")::find_solver_and_readable_propagator_of_type: ";

  if(input_type==NemoPhys::Fermion_retarded_Green||input_type==NemoPhys::Boson_retarded_Green)
    solver=exact_GR_solver;
  else if(input_type==NemoPhys::Fermion_lesser_Green||input_type==NemoPhys::Boson_lesser_Green)
    solver=exact_GL_solver;
  else if(input_type==NemoPhys::Inverse_Green)
    solver=inverse_GR_solver;
  else if(input_type==NemoPhys::Fermion_retarded_self||input_type==NemoPhys::Boson_retarded_self)
    solver=scattering_sigmaR_solver;
  else if(input_type==NemoPhys::Fermion_lesser_self||input_type==NemoPhys::Boson_lesser_self)
    solver=scattering_sigmaL_solver;
  else
  {
    std::map<NemoPhys::Propagator_type, std::string>::const_iterator cit=Propagator_type_map.find(input_type);
    if(cit!=Propagator_type_map.end())
      throw std::invalid_argument(prefix+"called with input_type: "+cit->second+"\n");
    else
      throw std::invalid_argument(prefix+"called with unknown input_type\n");
  }

  NEMO_ASSERT(solver!=NULL,prefix+"required solver is NULL\n");
  PropagatorInterface* temp_interface=dynamic_cast<PropagatorInterface*>(solver);
  NEMO_ASSERT(temp_interface!=NULL,prefix+solver->get_name()+" is not a PropagatorInterface\n");
  temp_interface->get_Propagator(result_propagator, &input_type);

  NemoUtils::toc(tic_toc_prefix);
}
void Propagation::calculate_transmission(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_transmission ");
  NemoUtils::tic(tic_toc_prefix);
  std::vector<double>* temp = new std::vector<double>;
  calculate_transmission_energy(temp);
  delete temp;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::calculate_energy_interpolated_transmission(std::map<std::vector<NemoMeshPoint>, double>& local_momentum_transmission )
{

  if(!options.get_option("energy_interpolated_output",false))
  {
    return;
  }
  NemoMesh* energy_mesh=Mesh_tree_topdown.begin()->first;
  MPI_Barrier(energy_mesh->get_global_comm());
  int myrank;
  myrank=get_simulation_domain()->get_geometry_replica();
  int energy_rank;
  MPI_Comm_rank(energy_mesh->get_global_comm(),&energy_rank);
  if((myrank==0)&&(energy_rank==0)&&!no_file_output)
  {
    energy_resolved_transmission_interpolated.clear();
    double number_of_interpolation_points=options.get_option("number_of_interpolation_points",1400);
    interpolate_nonrectangular_energy(local_momentum_transmission, energy_resolved_transmission_interpolated ,number_of_interpolation_points);
    //3. store the result on file by one MPI process, with the appropriate voltage in the filename
    std::map<double,double>::iterator energy_resolved_transmission_it =energy_resolved_transmission_interpolated.begin();
    std::stringstream stm;
    std::string potential_diff = stm.str();
    ofstream out_file;
    string filename=get_name()+"_"+potential_diff+"_energy_interpolated_transmission_" + get_output_suffix() + ".dat";
    out_file.open(filename.c_str());
    out_file.precision(10);
    for(; energy_resolved_transmission_it != energy_resolved_transmission_interpolated.end(); energy_resolved_transmission_it++)
    {
      out_file<<energy_resolved_transmission_it->first<<"\t"<<energy_resolved_transmission_it->second<<"\n";
    }
    out_file.close();
  }

}

void Propagation::calculate_momentum_resolved_transmission(std::map<std::vector<NemoMeshPoint>,double >& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_momentum_resolved_transmission ");
  NemoUtils::tic(tic_toc_prefix);

  if(options.get_option("Non_orthogonal",false) )
  {
    std::string backward_green_module_name =options.get_option("backward_green_module", string(""));
    Simulation* backward_green_module =find_simulation(backward_green_module_name);
    BackwardRGFSolver* temp_backwardsolver = dynamic_cast<BackwardRGFSolver*>(backward_green_module);
    result = temp_backwardsolver->get_local_tranmission();
  }
  else
  {
    std::map<std::string, const Propagator*>::const_iterator G_iterator, self_iterator1, self_iterator2;
    if(options.check_option("G_for_transmission"))
    {
      std::string name_of_G=options.get_option("G_for_transmission",std::string(""));
      G_iterator=Propagators.find(name_of_G);
      NEMO_ASSERT(G_iterator!=Propagators.end(),tic_toc_prefix+"have not found Propagator \""+name_of_G+"\n");
      NemoPhys::Propagator_type G_type=Propagator_types.find(name_of_G)->second;
      NEMO_ASSERT(G_type==NemoPhys::Fermion_retarded_Green ||
                G_type==NemoPhys::Boson_retarded_Green,tic_toc_prefix+"Type of \""+name_of_G
                +" is \""+Propagator_type_map.find(G_type)->second+"\"\n");
    }
    else
    {
      G_iterator=Propagators.end();
      std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
      for(; c_it!=Propagators.end(); ++c_it)
      {
      std::map<std::string, NemoPhys::Propagator_type>::const_iterator temp_cit=Propagator_types.find(c_it->first);
      NEMO_ASSERT(temp_cit!=Propagator_types.end(),tic_toc_prefix+"Type of \""+c_it->first+" is not found\n");
      if(temp_cit->second==NemoPhys::Fermion_retarded_Green || temp_cit->second==NemoPhys::Boson_retarded_Green)
      {
        NEMO_ASSERT(G_iterator==Propagators.end(),tic_toc_prefix+"found more than one retarded Green's function\n");
        G_iterator=c_it;
      }
    }
  }
  //get the retarded Green's function from its constructor...
  std::map<std::string,Simulation*>::const_iterator prop_cit=pointer_to_Propagator_Constructors->find(G_iterator->first);
  NEMO_ASSERT(prop_cit!=pointer_to_Propagator_Constructors->end(),tic_toc_prefix+"have not found constructor of \""+G_iterator->first+"\"\n");
  //const Propagator * ret_Green=G_iterator->second;
  const Propagator* ret_Green=NULL;
  prop_cit->second->get_data(G_iterator->first,ret_Green);
  NEMO_ASSERT(ret_Green!=NULL,tic_toc_prefix+"have not found the retarded Green's function\n");

  //get the first & second retarded contact self-energy
  NEMO_ASSERT(options.check_option("selfenergy1_for_transmission"),
              tic_toc_prefix+"define \"selfenergy1_for_transmission\"\n");
  NEMO_ASSERT(options.check_option("selfenergy2_for_transmission"),
              tic_toc_prefix+"define \"selfenergy2_for_transmission\"\n");
  std::string self_energy1_name=options.get_option("selfenergy1_for_transmission",std::string(""));
  std::string self_energy2_name=options.get_option("selfenergy2_for_transmission",std::string(""));
  std::set<Simulation*>::iterator contact_sigma_solvers_it = contact_sigmaR_solvers.begin();
  Simulation* source_of_self_energy1 = *contact_sigma_solvers_it;
  contact_sigma_solvers_it++;
  Simulation* source_of_self_energy2 = *contact_sigma_solvers_it;

  //loop over all momenta - store in transmission
  Propagator::PropagatorMap::const_iterator green_matrix_iterator, self1_matrix_iterator,self2_matrix_iterator;
  green_matrix_iterator=ret_Green->propagator_map.begin();
  for(; green_matrix_iterator!=ret_Green->propagator_map.end(); ++green_matrix_iterator)
  {
    //find the required matrices
    PetscMatrixParallelComplex* green_matrix=green_matrix_iterator->second;
    PetscMatrixParallelComplex* self_matrix1;  //=self1_matrix_iterator->second;
    source_of_self_energy1->get_data(self_energy1_name,&(green_matrix_iterator->first),self_matrix1,
                                     &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    PetscMatrixParallelComplex* self_matrix2; //=self2_matrix_iterator->second;
    source_of_self_energy2->get_data(self_energy2_name,&(green_matrix_iterator->first),self_matrix2,
                                     &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));

    
    double transmission;
    PropagationUtilities::core_calculate_transmission(this, self_matrix1, self_matrix2, green_matrix, transmission);

    if(result.find(green_matrix_iterator->first)!=result.end())
      result.find(green_matrix_iterator->first)->second += transmission;
    else
      result[green_matrix_iterator->first] = transmission;
   }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::calculate_transmission_energy(std::vector<double>*& transmission_energy)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_transmission_energy ");
  NemoUtils::tic(tic_toc_prefix);
  std::map<double,std::complex<double> > transmission;

  //std::map<std::vector<NemoMeshPoint>,double > momentum_transmission;
  if(momentum_transmission.empty())
    calculate_momentum_resolved_transmission(momentum_transmission);

  //get pointer_to_all_momenta if it is not initialized
  if (pointer_to_all_momenta->size() == 0)
    Parallelizer->get_data("all_momenta", pointer_to_all_momenta);
  NEMO_ASSERT(pointer_to_all_momenta->size() > 0, tic_toc_prefix + "received empty all_momenta\n");

  //output T(E) for all other momenta.
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
  //0.1 initialize a place holder vector of the size of all momenta
  std::vector<double> local_transmission_vector(pointer_to_all_momenta->size(),0.0);
  if(options.get_option("output_raw_transmission_data",false))
  {
    //0.2 copy the local transmission into the place holder
    int counter=0;
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      //0.2.1 copy tranmission if found
      std::map<std::vector<NemoMeshPoint>,double>::const_iterator transmission_cit=momentum_transmission.find(*c_it);
      if(transmission_cit!=momentum_transmission.end())
        local_transmission_vector[counter]=transmission_cit->second; //.real();
      counter++;
    }
    //0.3 MPIreduce the transmission into central rank
    //0.3.1 reduce data w.r.t. all-momentum rank 0
    NemoMesh* energy_mesh=Mesh_tree_topdown.begin()->first;
    MPI_Barrier(energy_mesh->get_global_comm());
    NemoUtils::tic(tic_toc_prefix+"communication of doubles for output");
    if(solve_on_single_replica)
    {
      std::vector<double> temp2(local_transmission_vector);
      MPI_Reduce(&(temp2[0]),&(local_transmission_vector[0]),local_transmission_vector.size(),MPI_DOUBLE, MPI_SUM, 0,
                 energy_mesh->get_global_comm());
    }
    else
    {
      MPI_Barrier(energy_mesh->get_global_comm());
      NemoUtils::tic(tic_toc_prefix+"communication of doubles for output MPIAllreduce");
      MPI_Allreduce(MPI_IN_PLACE,&(local_transmission_vector[0]),local_transmission_vector.size(),MPI_DOUBLE, MPI_SUM,
                    energy_mesh->get_global_comm());
      NemoUtils::toc(tic_toc_prefix+"communication of doubles for output MPIAllreduce");
    }
    //0.4 on all-momentum rank 0 && real space rank 0: write momentum tupel and transmission to file
    int myrank;
    myrank=get_simulation_domain()->get_geometry_replica();
    int energy_rank;
    MPI_Comm_rank(energy_mesh->get_global_comm(),&energy_rank);
    if(myrank==0&&energy_rank==0&&!no_file_output)
    {
      std::string filename=get_name()+"_raw_transmission"+get_output_suffix()+".dat";
      std::ofstream out_file;
      out_file.open(filename.c_str());
      //first write the names of the meshes
      {
        //NEMO_ASSERT(writeable_Propagators.size()>0,prefix+"found empty propagator list\n");
        NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"received NULL for writeable_Propagator\n");
        std::vector<std::string> meshnames=writeable_Propagator->momentum_mesh_names;
        for(unsigned j=0; j<meshnames.size(); j++)
        {
          //TODO @Fabio, need to support valley output
          out_file<<meshnames[j];
          if(meshnames[j].find("energy")!=std::string::npos)
            out_file<<" [eV]\t";
          else if(meshnames[j].find("valley")!=std::string::npos)
            out_file<<" [NA]\t";
          else
            out_file<<"[1/nm] [1/nm] [1/nm]\t";
        }
        out_file<<"transmission"<<"\n";
      }
      int counter=0;
      for(c_it=pointer_to_all_momenta->begin(); c_it!=pointer_to_all_momenta->end(); c_it++)
      {
        std::vector<NemoMeshPoint> temp_point = *c_it;
        for(unsigned int i=0; i<temp_point.size(); i++)
        {
          const vector<double>& temp_coords=temp_point[i].get_coords();
          for(unsigned int j=0; j<temp_coords.size(); j++)
          {
            out_file<<temp_coords[j]<<"\t";
          }
        }
        out_file<<local_transmission_vector[counter]<<"\n";
        counter++;
      }
      out_file.close();
    }
    NemoUtils::toc(tic_toc_prefix+"communication of doubles for output");
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  //      caculate energy resolved transmission using interpolation in case of K space   :
  /////////////////////////////////////////////////////////////////////////////////////////


  std::set < std::vector< NemoMeshPoint > >::iterator mom_it = pointer_to_all_momenta->begin();
  std::map<std::vector<NemoMeshPoint>, double>local_momentum_transmission;

  for( int counter=0; mom_it!=pointer_to_all_momenta->end(); mom_it++, counter++)
  {
    local_momentum_transmission[(*mom_it)]=local_transmission_vector[counter];
  }
  calculate_energy_interpolated_transmission(local_momentum_transmission );

  //////////////////////////////////////////////////////////////////////////////////////////
  //    end caculate energy resolved transmission using interpolation in case of K space  //
  //////////////////////////////////////////////////////////////////////////////////////////
  if(Mesh_tree_names.size()>1) //there are more meshes than just energy...
  {
    //integrate the transmission with respect to the (non-energy) momenta

    //loop over the momenta
    std::map<std::vector<NemoMeshPoint>,double >::const_iterator c_it=momentum_transmission.begin();
    for(; c_it!=momentum_transmission.end(); ++c_it)
    {
      std::complex<double> complex_energy = PropagationUtilities::read_complex_energy_from_momentum(this, c_it->first, writeable_Propagator);
      double energy = complex_energy.real();
      //get the integration weight for this momentum from the momentum constructor(s)
      double integration_weight=1.0;
      if(!options.get_option("no_integration_for_transmission",false))
      {
        std::string momentum_name=std::string("");
        for(unsigned int i=0; i<Mesh_tree_names.size(); i++)
        {
          //find the momentum name that does not contain "energy"
          if(Mesh_tree_names[i].find("energy")==std::string::npos)
          {
            momentum_name=Mesh_tree_names[i];
            //find the momentum mesh constructor and ask it for the weight
            std::map<std::string,Simulation*>::const_iterator c_it2 = Propagation::Mesh_Constructors.begin();
            c_it2 = Propagation::Mesh_Constructors.find(momentum_name);
            NEMO_ASSERT(c_it2!=Propagation::Mesh_Constructors.end(),
                        tic_toc_prefix+"have not found constructor for momentum mesh \""+momentum_name+"\"\n");
            double temp;
            InputOptions& mesh_options=c_it2->second->get_reference_to_options();
            if(!mesh_options.get_option(std::string("non_rectangular"),false))
              //if(!options.get_option("non_rectangular_"+momentum_name,false))
              c_it2->second->get_data("integration_weight",(c_it->first)[i],temp);
            else
              c_it2->second->get_data("integration_weight",c_it->first,(c_it->first)[i],temp);
            integration_weight*=temp;

            if(momentum_name.find("momentum_1D")!=std::string::npos)
              integration_weight/=2.0*NemoMath::pi;
            else if(momentum_name.find("momentum_2D")!=std::string::npos)
              integration_weight/=4.0*NemoMath::pi*NemoMath::pi;
            else if(momentum_name.find("momentum_3D")!=std::string::npos)
              integration_weight/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
          }
        }
      }
      //add the weighted transmission to the storing map
      std::map<double,std::complex<double> >::iterator it=transmission.find(energy);
      //std::cerr << prefix << integration_weight<<"\n";
      if(it!=transmission.end())
        it->second+=integration_weight*c_it->second;
      else
        transmission[energy]=integration_weight*c_it->second;
    }
  }
  else //there is only energy (1D wire case)
  {
    //copy momentum_transmission to transmission
    std::map<std::vector<NemoMeshPoint>,double >::const_iterator c_it=momentum_transmission.begin();
    for(; c_it!=momentum_transmission.end(); ++c_it)
    {
      NEMO_ASSERT(c_it->first.size()==1,tic_toc_prefix+"wrong dimensionality in the momentum mesh\n");
      std::complex<double> complex_energy = PropagationUtilities::read_complex_energy_from_momentum(this, c_it->first, writeable_Propagator);
      double energy = complex_energy.real();
      transmission[energy]=c_it->second;
    }
  }

  //------MPI-reduce the result------
  //find all existing energies
  NEMO_ASSERT(pointer_to_all_momenta->size()>0,tic_toc_prefix+"received empty all_momenta\n");
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=pointer_to_all_momenta->begin();
  std::set<double> temp_all_energies;
  for(; c_it2!=pointer_to_all_momenta->end(); c_it2++)
  {
    const Propagator* temp_propagator=*(known_Propagators.begin());
    NEMO_ASSERT(temp_propagator!=NULL,tic_toc_prefix+"have not found non-NULL propagator\n");
    std::complex<double> complex_energy = PropagationUtilities::read_complex_energy_from_momentum(this, *c_it2, temp_propagator);
    double temp_energy = complex_energy.real();
    temp_all_energies.insert(temp_energy);
  }
  std::map<std::string, NemoMesh*>::const_iterator temp_cit=Momentum_meshes.begin();
  NemoMesh* energy_mesh=NULL;
  for (; temp_cit!=Momentum_meshes.end(); temp_cit++)
    if(temp_cit->first.find("energy")!=std::string::npos)
    {
      energy_mesh=temp_cit->second;
      if(energy_mesh==NULL)
      {
        std::map<std::string, Simulation*>::const_iterator temp_c_it2=Mesh_Constructors.find(temp_cit->first);
        NEMO_ASSERT(temp_c_it2!=Mesh_Constructors.end(),tic_toc_prefix+"no mesh constructor found for \""+temp_cit->first+"\"\n");
        Simulation* mesh_source = temp_c_it2->second;
        mesh_source->get_data("e_space_"+temp_cit->first,energy_mesh);
      }
    }
  NEMO_ASSERT(energy_mesh!=NULL,tic_toc_prefix+"have not found the energy mesh\n");
  //create a vector with all energies that exist (not only within this MPI-process)
  std::vector<double> all_energies(temp_all_energies.size(),0.0); //(temp_energy_pointer.size(),0.0);
  //for(unsigned int i=0; i<temp_energy_pointer.size(); i++)
  //  all_energies[i]=temp_energy_pointer[i]->get_x();
  unsigned int counter=0;
  for(std::set<double>::iterator it=temp_all_energies.begin(); it!=temp_all_energies.end(); ++it)
  {
    all_energies[counter]=*it;
    counter++;
  }


  //fill the result vector with zeros or the values calculated within this MPI-process
  std::vector<double> result_transmission(all_energies.size(),0.0);
  std::map<double,std::complex<double> >::const_iterator transmission_cit;
  counter=0;
  for(unsigned int i=0; i<result_transmission.size(); i++)
  {
    transmission_cit=transmission.find(all_energies[i]);
    if(transmission_cit!=transmission.end())
    {
      result_transmission[i]+=transmission_cit->second.real();
      counter++;
    }
  }
  NemoMesh* energy_mesh_for_comm=Mesh_tree_topdown.begin()->first;
  //check that all transmission results are put into the vector
  NEMO_ASSERT(counter==transmission.size(),tic_toc_prefix+"incomplete storage of transmission results\n");
  //call MPI_allreduce for this vector

  int myrank;
  MPI_Comm_rank(energy_mesh_for_comm->get_global_comm(), &myrank);
  MPI_Barrier(energy_mesh_for_comm->get_global_comm());
  MPI_Allreduce(MPI_IN_PLACE,&(result_transmission[0]),result_transmission.size(),MPI_DOUBLE, MPI_SUM,energy_mesh_for_comm->get_global_comm());

  //output - for test purposes
  //Suppressed by default - ruins scaling - EMW
  //Option isn't currently read correctly from input deck
  //Think option isn't being read correctly from RGF module
  //
  bool no_transmission_output = options.get_option("no_transmission_output", false);
  if (no_transmission_output==false)
  {

    if (debug_output&&!no_file_output)
    {
      std::stringstream convert_to_string;
      convert_to_string << myrank;
      std::string filename=get_name()+convert_to_string.str() +"_transmission.dat";

      std::ofstream out_file;
      out_file.open(filename.c_str());
      std::map<double,std::complex<double> >::const_iterator c_it=transmission.begin();
      for(; c_it!=transmission.end(); ++c_it)
      {
        out_file<<c_it->first<<"\t"<<c_it->second.real()<<"\n";
      }
      out_file.close();
    }
    if(myrank==0&&!no_file_output)
    {
      std::string filename=get_name()+"_fulltransmission.dat";
      std::ofstream out_file;
      out_file.open(filename.c_str());
      for(unsigned int i=0; i<all_energies.size(); i++)
      {
        out_file<<all_energies[i]<<"\t"<<result_transmission[i]<<"\n";
      }
      out_file.close();
    }


  }

  *transmission_energy=result_transmission;
  //store the total (mpi-reduced) transmission in the Propagation::energy_resolved_transmission
  for(unsigned int i=0; i<all_energies.size(); i++)
  {
    std::map<double,double>::iterator it=energy_resolved_transmission.find(all_energies[i]);
    if(it==energy_resolved_transmission.end())
      energy_resolved_transmission[all_energies[i]]=result_transmission[i];
    else
      it->second=result_transmission[i];
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable,const std::vector<unsigned int>& index,std::vector<double>& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data8 ");
  NemoUtils::tic(tic_toc_prefix);
  const Domain* temp_domain=get_const_simulation_domain();
  const double temp_volume=temp_domain->return_periodic_space_volume();//to get a number of particles rather then a density
  if (variable=="electron_density" || variable=="free_charge")
  {
    if(density.size()==0)
      calculate_density();
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      data[i] =( -(this->get_density(index[i])) * temp_volume) ;


    }
  }
  else if (variable=="hole_density")
  {
    if(density.size()==0)
      calculate_density();
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
      data[i] =  (this->get_density(index[i]) * temp_volume );
  }
  else if (variable=="derivative_electron_density_over_potential"||variable=="derivative_total_charge_density_over_potential")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
      data[i] = ( -(this->get_density_Jacobian(index[i])) * temp_volume);
  }
  else if (variable=="derivative_hole_density_over_potential")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      data[i] =  (this->get_density_Jacobian(index[i]) * temp_volume);
    }
  }
  else
    throw std::runtime_error("Propagation(\""+get_name()+"\")::get_data unknown density type \""+variable+"\"\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable,std::map<double, std::map<unsigned int, double>,Compare_double_or_complex_number >& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data9 ");
  NemoUtils::tic(tic_toc_prefix);
  if(variable=="energy_resolved_density")
  {
    if(density.size()==0)
      calculate_density();
    NEMO_ASSERT(energy_resolved_density_for_output.size()!=0,"energy_resolved_density was not calculated, please set propagator output options correctly\n");
    data = energy_resolved_density_for_output;
  }
  else
    throw std::runtime_error("Propagation(\""+get_name()+"\")::get_data unknown density type \""+variable+"\"\n");
  NemoUtils::toc(tic_toc_prefix);
}

double Propagation::get_density (unsigned int atom_id)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_density ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+get_name()+"\")::get_density ";
  NEMO_ASSERT(atomic_output_only,prefix+"called when orbital resolved information is calculated\n");
  if(density.size()==0)
    calculate_density();
  std::map<unsigned int, double>::const_iterator it=density.find(atom_id);
  if(it==density.end())
    throw std::runtime_error(prefix+"reached end of density\n");
  NemoUtils::toc(tic_toc_prefix);
  return it->second;
}

double Propagation::get_density_Jacobian(const unsigned int atom_id)
{
  std::string tic_toc_prefix = "Propagation(\""+tic_toc_name+"\")::get_density_Jacobian ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix=NEMOUTILS_PREFIX("Propagation(\""+get_name()+"\")::get_density_Jacobian ");
  NEMO_ASSERT(atomic_output_only,prefix+"called when orbital resolved information is calculated\n");
  calculate_density_Jacobian();
  std::map<unsigned int, double>::const_iterator it=density_Jacobian.find(atom_id);
  if(it==density_Jacobian.end())
    throw std::runtime_error(prefix+"reached end of density\n");
  NemoUtils::toc(tic_toc_prefix);
  return it->second;
}

void Propagation::calculate_energy_interpolated_atom_density()
{

  // 1 find the source of the denisty
  std::string prefix="Propagation(\""+get_name()+"\")::calculate_energy_interpolated_atom_density ";


  //1.1 find out of which Green's function the density should be calculated (either the writeable lesser_Green of this Propagation or inputdeck given)
  std::string name_of_propagator("");
  if(writeable_Propagator!=NULL)
    name_of_propagator=name_of_writeable_Propagator;
  else if(options.check_option("density_of"))
  {
    name_of_propagator=options.get_option("density_of",std::string(""));
  }
  else
  {
    //loop over all writeable Green's functions
    std::map<std::string, const Propagator*>::iterator it=Propagators.begin();
    for(; it!=Propagators.end(); ++it)
    {
      if(Propagator_types.find(it->first)->second==NemoPhys::Fermion_lesser_Green || Propagator_types.find(it->first)->second==NemoPhys::Boson_lesser_Green)
      {
        //check that there is only one lesser Green
        name_of_propagator=it->first;
      }
    }
  }
  NEMO_ASSERT(name_of_propagator!=std::string(""),prefix+"the propagator name not find");
  //2. get the source simulation of the density
  Simulation* source_of_data = find_source_of_data(name_of_propagator);

  //3. get access propagator that hold density matrix
  Propagator* Propagator_pointer;
  source_of_data->get_data(name_of_propagator,Propagator_pointer);

  //4. loop over all momenta and collect the density at each energy point.
  //holder of the local density
  std::map<std::vector<NemoMeshPoint>,std::vector<double> > temp_map2;
  std::map<std::vector<NemoMeshPoint>,std::vector<double> >& reference_to_density_map=temp_map2;
  //temp holder of teh density matrix
  PetscMatrixParallelComplex* temp_matrix=NULL;
  //diagonal of the density matrix
  std::vector<std::complex<double> > diagonal;
  //real of the diagonal of the density matrix
  std::vector<double > diagonal_real;
  //the prefactor is real in case of density of electron
  //the prefactor is imag in case of density of states
  double real_prefactor=options.get_option(name_of_propagator+"_energy_resolved_output_real",1.0);
  double imag_prefactor=options.get_option(name_of_propagator+"_energy_resolved_output_imag",0.0);
  //tinvert the prefactor to get the required data.
  std::complex<double> prefactor_total(real_prefactor,imag_prefactor);
  std::complex<double> prefactor_temp(0.0,-1);
  prefactor_total = prefactor_total*prefactor_temp;
  //iterate over the local momentum to get the data
  Propagator::PropagatorMap::const_iterator momentum_c_it=Propagator_pointer->propagator_map.begin();
  for(; momentum_c_it!=Propagator_pointer->propagator_map.end(); ++momentum_c_it)
  {
    source_of_data->get_data(name_of_propagator,&(momentum_c_it->first),temp_matrix,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    if(temp_matrix->if_container())
        temp_matrix->assemble();
    temp_matrix->get_diagonal(&diagonal);
    diagonal_real.resize(diagonal.size(),0.0);
    for(unsigned int i =0; i< diagonal.size();i++)
    {
      diagonal_real[i]=(diagonal[i]*prefactor_total).real();
    }
    temp_map2[(momentum_c_it->first)]=diagonal_real;
  }

  //5 index the data by atome and do teh interpolation atom by atom
  int number_of_interpolation_points=options.get_option("energy_interpolation_points",100);
  std::map< int , std::map<double,double> > energy_density_atom_map_output;
  std::map<std::vector<NemoMeshPoint >, double > momentum_density_map;
  std::map<std::vector<NemoMeshPoint >, std::vector<double> >:: iterator momentum_density_atom_map_it =reference_to_density_map.begin();

  momentum_density_atom_map_it =reference_to_density_map.begin();
  unsigned int number_of_DOFs=momentum_density_atom_map_it->second.size();
  for(unsigned int j=0; j<number_of_DOFs; j++)
  {

    momentum_density_atom_map_it =reference_to_density_map.begin();
    for(; momentum_density_atom_map_it !=reference_to_density_map.end(); momentum_density_atom_map_it++)
    {
      momentum_density_map[momentum_density_atom_map_it->first]=momentum_density_atom_map_it->second[j];
    }
    std::map<double, double> temp;
    energy_density_atom_map_output[j] =temp;
    interpolate_nonrectangular_energy(momentum_density_map, energy_density_atom_map_output[j] ,number_of_interpolation_points);
  }

  //6 communicate the data atom by atom
  int myrank;
  MPI_Comm momentum_tupel_communicator=Mesh_tree_topdown.begin()->first->get_global_comm();
  MPI_Comm_rank(momentum_tupel_communicator,&myrank);
  std::map<double,double>::iterator energy_it=energy_density_atom_map_output[0].begin();
  int index=0;
  std::vector<double> send_temp;
  std::vector<double> receive_temp;
  send_temp.resize(number_of_interpolation_points, 0.0);
  receive_temp.resize(number_of_interpolation_points, 0.0);

  for(unsigned int j=0; j<number_of_DOFs; j++)
  {
    for(index=0, energy_it=energy_density_atom_map_output[j].begin(); energy_it!=energy_density_atom_map_output[j].end();energy_it++,index++)
    {
      send_temp[index]=energy_it->second;
    }


    MPI_Reduce(&(send_temp[0]),&(receive_temp[0]),send_temp.size(),MPI_DOUBLE, MPI_SUM, 0, momentum_tupel_communicator);//get_simulation_domain()->get_one_partition_total_communicator());

    if(myrank==0)
    {
      for(index=0, energy_it=energy_density_atom_map_output[j].begin(); energy_it!=energy_density_atom_map_output[j].end();energy_it++,index++)
      {
        energy_it->second=receive_temp[index];
      }
    }

  }

  //6 print the data
  if(myrank==0)
  {
    // file name
    std::map<int,int> index_to_atom_id_map;
    std::string nondistributed_hamiltonian_name=options.get_option("nondistributed_Hamilton_constructor",Hamilton_Constructor->get_name());
    Simulation* nondistributed_Hamilton_constructor = find_simulation(nondistributed_hamiltonian_name);
    NEMO_ASSERT(nondistributed_Hamilton_constructor!=NULL,prefix+"have not found simulation \""+nondistributed_hamiltonian_name+"\"\n");
    const DOFmapInterface& temp_DOFmap=nondistributed_Hamilton_constructor->get_const_dof_map();
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);
    double chemical_potential1, chemical_potential2;
    if(options.check_option("Source_chemical_potential"))
      chemical_potential1=options.get_option("Source_chemical_potential", 0.0);
    else
      chemical_potential1=options.get_option("source_chemical_potential", 0.0);
    if(options.check_option("Drain_chemical_potential"))
      chemical_potential2=options.get_option("Drain_chemical_potential", 0.0);
    else
      chemical_potential2=options.get_option("drain_chemical_potential", 0.0);

    std::stringstream stm;
    stm<<chemical_potential1-chemical_potential2;
    std::string potential_diff = stm.str();

    // convert the data to be indexed by energy to do teh translation by dof maps
    std::map<double, std::vector< double>,Compare_double_or_complex_number > energy_resolved_density_map;
    for(unsigned int i =0;i<energy_density_atom_map_output.size();i++)
    {
      energy_it=energy_density_atom_map_output[i].begin();

      for(;energy_it!=energy_density_atom_map_output[i].end();energy_it++)
      {
        std::map<double, std::vector< double>,Compare_double_or_complex_number >::iterator x_it= energy_resolved_density_map.find(energy_it->first);

        if(x_it==energy_resolved_density_map.end())
        {
          std::vector<double> temp(energy_density_atom_map_output.size(),0.0);
          energy_resolved_density_map[energy_it->first] = temp;
        }
        x_it= energy_resolved_density_map.find(energy_it->first);
        x_it->second[i]=(energy_it->second);
      }
    }

    // translate using dof maps
    std::map<double, std::map<unsigned int , double>,Compare_double_or_complex_number > energy_atom_density_map;
    std::map<double, std::vector< double>,Compare_double_or_complex_number >::iterator x_it= energy_resolved_density_map.begin();
    for( ; x_it!=energy_resolved_density_map.end();x_it++)
    {
      std::map<unsigned int, double> temp_map;
      translate_vector_into_map_real( x_it->second, temp_map);
      energy_atom_density_map[x_it->first]=temp_map;
    }

    //print the denisty
    string file_name_input = "energy_interpolated_"+potential_diff+"_"+name_of_propagator;
    reset_output_counter();
    reset_output_data();
    print_atomic_maps(energy_atom_density_map,file_name_input);
    reset_output_counter();
    reset_output_data();
  }
}



void Propagation::calculate_energy_interpolated_density()
{
  std::string prefix="Propagation::(\""+this->get_name()+"\")::calculate_density_interp: ";
  if(!options.get_option("energy_interpolated_output",false))
  {
    return;
  }
  NemoMesh* energy_mesh=Mesh_tree_topdown.begin()->first;
  MPI_Barrier(energy_mesh->get_global_comm());
  int myrank;
  myrank=get_simulation_domain()->get_geometry_replica();
  int energy_rank;
  MPI_Comm_rank(energy_mesh->get_global_comm(),&energy_rank);
  if((myrank==0)&&(energy_rank==0)&&!no_file_output)
  {
    std::map<std::vector<NemoMeshPoint>, std::complex<double> > momentum_density_map;
    std::set<std::vector<NemoMeshPoint> >::iterator c_it= pointer_to_all_momenta->begin();
    std::vector <std::complex<double> > density_vector(pointer_to_all_momenta->size(),0);
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      momentum_density_map[*c_it]=0;
    }
    std::map<std::vector<NemoMeshPoint>, std::complex<double>  >::iterator c_it_2 =density_momentum_map_interp.begin();

    for(; c_it_2!=density_momentum_map_interp.end(); c_it_2++)
    {
      momentum_density_map[c_it_2->first]=c_it_2->second;
    }
    c_it_2 =momentum_density_map.begin();
    for(unsigned int i =0; i< density_vector.size(); i++,c_it_2++)
    {
      density_vector[i]=c_it_2->second;
    }
    std::vector <std::complex<double> > recive_density_vector(pointer_to_all_momenta->size(),0);
    NEMO_ASSERT(Mesh_tree_topdown.size()>0,prefix+"Mesh_tree_topdown is not ready for usage\n");
    const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
    if(!solve_on_single_replica)
      MPI_Allreduce(&(density_vector[0]),&(recive_density_vector[0]),recive_density_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
    else
      MPI_Reduce(&(density_vector[0]),&(recive_density_vector[0]),recive_density_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM , 0, topcomm);

    std::map< std::vector<NemoMeshPoint>, double > real_density;
    std::map< std::vector<NemoMeshPoint>, double > imag_density;
    c_it_2 =momentum_density_map.begin();
    for(unsigned int i=0; i<recive_density_vector.size(); i++)
    {
      c_it_2->second=recive_density_vector[i];
      real_density[c_it_2->first]=real(recive_density_vector[i]);
      imag_density[c_it_2->first]=imag(recive_density_vector[i]);
    }
    double number_of_interpolation_points=options.get_option("number_of_interpolation_points",7584.0);
    interpolate_nonrectangular_energy(real_density, real_energy_resolved_density_interpolated ,number_of_interpolation_points);
    interpolate_nonrectangular_energy(imag_density, imag_energy_resolved_density_interpolated ,number_of_interpolation_points);
    std::string filename=get_name()+"_energy_interpolated_density_"+get_output_suffix()+".dat";
    ofstream out_file;
    out_file.open(filename.c_str());
    out_file.precision(10);
    std::map<double,double>::iterator  it_2_real=real_energy_resolved_density_interpolated.begin() ;
    std::map<double,double>::iterator  it_2_imag=imag_energy_resolved_density_interpolated.begin() ;
    for(; it_2_real !=  real_energy_resolved_density_interpolated.end(); it_2_real++,it_2_imag++)
    {
      out_file<<it_2_real->first<<"\t"<<it_2_real->second<<"\t"<<it_2_imag->second<<"\n";
    }
    out_file.close();
  }
}


void Propagation::calculate_density(std::map<unsigned int,double>* result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_density ");
  NemoUtils::tic(tic_toc_prefix);
  std::string error_prefix="Propagation(\""+get_name()+"\")::calculate_density ";
  msg<<error_prefix<<"\n";
  Propagator* lesser_Green=NULL;

  //1. find out of which Green's function the density should be calculated (either the writeable lesser_Green of this Propagation or inputdeck given)
  std::string name_of_propagator;
  Simulation* source_of_matrices = this;
  if(writeable_Propagator!=NULL)
    name_of_propagator=name_of_writeable_Propagator;
  else if(options.check_option("density_of"))
  {
    name_of_propagator=options.get_option("density_of",std::string(""));
    //2. find the source of the Green's function
    source_of_matrices = find_source_of_data(name_of_propagator);
  }
  else
  {
    //throw std::invalid_argument(error_prefix+"please define \"density_of\"\n");
    //loop over all writeable Green's functions
    std::map<std::string, const Propagator*>::iterator it=Propagators.begin();
    for(; it!=Propagators.end(); ++it)
    {
      if(Propagator_types.find(it->first)->second==NemoPhys::Fermion_lesser_Green || Propagator_types.find(it->first)->second==NemoPhys::Boson_lesser_Green)
      {
        //check that there is only one lesser Green
        NEMO_ASSERT(lesser_Green==NULL,error_prefix+"found more than one lesser Green's function; please define \"density_of\"\n");
        //lesser_Green=it->second;
        name_of_propagator=it->first;
        //2. find the source of the Green's function
        //source_of_matrices = find_source_of_data(name_of_propagator);
      }
    }
  }

  source_of_matrices = find_source_of_data(name_of_propagator);
  
  //4. integrate the diagonal of the lesser Green's function

  bool k_resolved_data = options.get_option(name_of_propagator+"_k_resolved_output", false) || k_resolved_density || options.get_option("k_resolved_density",false);

  bool energy_resolved_data=options.get_option(name_of_propagator+"_energy_resolved_output",false) ||
                                      options.get_option(name_of_propagator + "_energy_resolved",bool(false)) || options.get_option("energy_resolved_density",false);
  bool energy_resolved_per_k_data = false;
  if(options.get_option("energy_resolved_density",false))
    {
      //loop over all mesh_constructors and check whether one of them has the option ("non_rectangular = true")

      std::map<std::string, Simulation*>::const_iterator mesh_cit=Mesh_Constructors.begin();
      for(; mesh_cit!=Mesh_Constructors.end() && !energy_resolved_per_k_data; ++mesh_cit)
      {
        InputOptions& mesh_options = mesh_cit->second->get_reference_to_options();
        if(mesh_options.get_option(std::string("non_rectangular"),false))
        {
          energy_resolved_per_k_data = true;
          //don't want to store both energy resolved all k and energy resolved per k. This won't fit into memory.
          energy_resolved_data = false;
        }
      }
    }
  NemoPhys::Propagator_type input_type = get_Propagator_type(name_of_propagator);
  integrate_diagonal(source_of_matrices, input_type, name_of_propagator, energy_resolved_data, k_resolved_data, energy_resolved_per_k_data);
  //5. get access to the integrated diagonal
  PropagatorInterface* temp_interface=dynamic_cast<PropagatorInterface*>(source_of_matrices);
  NEMO_ASSERT(temp_interface!=NULL,error_prefix+source_of_matrices->get_name()+" is not a PropagatorInterface\n");
  //temp_interface->get_Propagator(lesser_Green);
  source_of_matrices->get_data(name_of_propagator,lesser_Green);
  std::vector<std::complex<double> > temp_density(*(lesser_Green->get_readable_integrated_diagonal()));

  //6. multiply with prefactors (depending on particle type)
  //get the prefactor according to the cell volume (depending on the momentum dimensionality)
  //for this read the momentum names (-1D,2D...)
  double prefactor = 1;
  const std::vector<std::string>& temp_mesh_names=lesser_Green->momentum_mesh_names;
  for (unsigned int i=0; i<temp_mesh_names.size(); i++)
  {
    if(temp_mesh_names[i].find("energy")!=std::string::npos || temp_mesh_names[i].find("momentum_1D")!=std::string::npos)
      prefactor/=2.0*NemoMath::pi;
    else if(temp_mesh_names[i].find("momentum_2D")!=std::string::npos)
      prefactor/=4.0*NemoMath::pi*NemoMath::pi;
    else if(temp_mesh_names[i].find("momentum_3D")!=std::string::npos)
      prefactor/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
  }
  if(result!=NULL)
    result->clear();
  else
    result=&density;

  double real_part_of_prefactor2(options.get_option("real_part_of_density_prefactor2",0.0));
  double imaginary_part_of_prefactor2(options.get_option("imaginary_part_of_density_prefactor2",-1.0));
  std::complex<double> prefactor2(real_part_of_prefactor2,imaginary_part_of_prefactor2);

  store_results_in_result_map(source_of_matrices,  name_of_propagator, lesser_Green,prefactor2, prefactor, temp_density,result);
  
  
  if (solve_the_spin_polarization)
  {
    unsigned int spin_factor = 1;
    Hamilton_Constructor->get_data("spin_degeneracy", spin_factor);
    if(spin_factor==1)
    {
	    NemoPhys::Propagator_type temp_type = Propagator_types.find(name_of_propagator)->second;
	    solve_spin_polarization(source_of_matrices, temp_type);
    }
    else
      OUT_WARNING<<error_prefix<<"spin polarization requested, but no spin included in the Hamiltonian. Skipping the polarization calculation...\n";
  }
  density_is_ready = true;
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::calculate_density_Jacobian(std::map<unsigned int,double>* result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_density_Jacobian ");
  NemoUtils::tic(tic_toc_prefix);
  std::string error_prefix="Propagation(\""+get_name()+"\")::calculate_density_Jacobian ";
  //throw std::runtime_error(error_prefix+"this method should not be called\n");


  if(result==NULL)
    density_Jacobian.clear();
  else
  {
    result->clear();
    result=&density_Jacobian;
  }

  //for a first implementation, we use the simple approximation given in older codes from TUMunich
  //i.e. Jacobian = density * exp(potential)
  //we assume that the density, the density_Jacobian and the potential are all given on the same mesh

  if(density.size()==0)
    calculate_density();

  density_Jacobian=density;

  NemoUtils::toc(tic_toc_prefix);
}

double Propagation::calculate_current_between_neighbors(const map<short, unsigned int>*& dof1, const map<short, unsigned int>*& dof2,
    const std::vector<NemoMeshPoint>& momentum)
{
  std::string prefix = "Propagation(\""+get_name()+"\")::calculate_current_between_neighbors ";
  //J(k,E)=2*Real(Trace(t_L1,L2(k)*G<_L2,l1(k,E)))
  //Ref: Lake et al, J.Appl.Phys. 81, 7845 (1997) Eq (20)
  //1. get t_L1,L2 from dof1 and dof2
  //1.1 get Hamiltonian of this domain
  PetscMatrixParallelComplex* Hamiltonian=NULL; //Hamiltonian matrix of this domain
  //std::set<unsigned int> Hamilton_momentum_indices;
  //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
  //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
  //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
  //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=momentum[*(pointer_to_Hamilton_momentum_indices->begin())];
  //bool avoid_copying_hamiltonian=options.get_option("avoid_copying_hamiltonian",false);

  std::vector<NemoMeshPoint> sorted_momentum;
  QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,momentum_mesh_types,Hamilton_Constructor);

  Hamilton_Constructor->get_data(std::string("Hamiltonian"),Hamiltonian,sorted_momentum,avoid_copying_hamiltonian); //obtain Hamiltonian

  const DOFmapInterface& domain_DOF = Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()); //obtain DOFmap for this domain

  //1.2 get coupling Hamiltonian t_L1,L2 from this domain Hamiltonian
  PetscMatrixParallelComplex* couple_L1L2=new PetscMatrixParallelComplex(dof1->size(),
      dof2->size(),get_simulation_domain()->get_communicator());
  couple_L1L2->set_num_owned_rows(dof1->size());
  for(unsigned int i=0; i<dof1->size(); i++)
    couple_L1L2->set_num_nonzeros_for_local_row(i,dof2->size(),0);
  std::vector<int> temp_rows(dof1->size());
  std::vector<int> temp_cols(dof2->size());

  int start_own_source_rows;
  int end_own_source_rows_p1;
  Hamiltonian->get_ownership_range(start_own_source_rows,end_own_source_rows_p1);
  std::map<short, unsigned int>::const_iterator temp_it1=dof1->begin();
  for(; temp_it1!=dof1->end(); temp_it1++)
  {
    int i = temp_it1->first;
    int j = temp_it1->second;//translated row index
    temp_rows[i]=j;
    if(j>=start_own_source_rows && j<end_own_source_rows_p1)
    {
      std::map<short, unsigned int>::const_iterator temp_it2=dof2->begin();
      for(; temp_it2!=dof2->end(); temp_it2++)
      {
        unsigned int ii = temp_it2->first;
        unsigned int jj = temp_it2->second;//translated column index
        if(jj<Hamiltonian->get_num_cols())
          temp_cols[ii]=jj;
      }
    }
  }
  Hamiltonian->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,couple_L1L2);
  couple_L1L2->assemble();
  //Hamiltonian->save_to_matlab_file("H.m");

  //2. get G<_L2,L1 matrix
  //2.1 get G< of this domain
  std::string name_of_propagator;
  if(options.check_option("density_of"))
    name_of_propagator=options.get_option("density_of",std::string(""));
  else
  {
    //loop over all writeable Green's functions
    //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.begin();
    //for(; it!=writeable_Propagators.end(); ++it)
    {
      if(Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Fermion_lesser_Green ||
          Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Boson_lesser_Green)
      {
        name_of_propagator=name_of_writeable_Propagator;
      }
    }
  }
  Simulation* source_of_matrices = find_source_of_data(name_of_propagator);
  Propagator* Propagator_pointer;
  source_of_matrices->get_data(name_of_propagator,Propagator_pointer);
  PetscMatrixParallelComplex* Glesser=NULL;
  source_of_matrices->get_data(name_of_propagator,&momentum,Glesser,&domain_DOF);
  if(Glesser->if_container())
    Glesser->assemble();
  //Glesser->save_to_matlab_file("Gl.m");

  //2.2 get G<_L2,L1 from G<
  PetscMatrixParallelComplex* Glesser_L2L1=new PetscMatrixParallelComplex(dof2->size(),
      dof1->size(),get_simulation_domain()->get_communicator());
  Glesser_L2L1->set_num_owned_rows(dof2->size());
  for(unsigned int i=0; i<dof2->size(); i++)
    Glesser_L2L1->set_num_nonzeros_for_local_row(i,dof1->size(),0);
  temp_rows.resize(dof2->size());
  temp_cols.resize(dof1->size());

  Glesser->get_ownership_range(start_own_source_rows,end_own_source_rows_p1);
  temp_it1=dof2->begin();
  for(; temp_it1!=dof2->end(); temp_it1++)
  {
    int i = temp_it1->first;
    int j = temp_it1->second;//translated row index
    temp_rows[i]=j;
    if(j>=start_own_source_rows && j<end_own_source_rows_p1)
    {
      std::map<short, unsigned int>::const_iterator temp_it2=dof1->begin();
      for(; temp_it2!=dof1->end(); temp_it2++)
      {
        unsigned int ii = temp_it2->first;
        unsigned int jj = temp_it2->second;//translated column index
        if(jj<Glesser->get_num_cols())
          temp_cols[ii]=jj;
      }
    }
  }
  Glesser->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,Glesser_L2L1);
  Glesser_L2L1->assemble();

  //3. calculate J=2*Re(Tr(t_L1,L2*G<_L2,L1))
  //3.1. mult t_L1,L2*G<_L2,L1
  PetscMatrixParallelComplex* HG=NULL;
  PetscMatrixParallelComplex::mult(*couple_L1L2,*Glesser_L2L1,&HG); //t_L1,L2*G<_L2,L1

  //3.2. obtain J=2*Re(Tr(t_L1,L2*G<_L2,L1))
  std::complex<double> trace(std::complex<double>(0.0,0.0));
  trace=HG->get_trace();
  if(avoid_copying_hamiltonian)
    delete Hamiltonian;
  delete HG;
  delete couple_L1L2;
  delete Glesser_L2L1;
  double J=2*trace.real();
  return J;
}

double Propagation::calculate_current_between_neighbors_RGF2(const std::vector<NemoMeshPoint>& momentum,PetscMatrixParallelComplex*& result)
{
  std::string prefix = "Propagation(\""+get_name()+"\")::calculate_current_between_neighbors_RGF2 ";
  //J(k,E)=2*Real(Trace(t_L1,L2(k)*G<_L2,l1(k,E)))
  //Ref: Lake et al, J.Appl.Phys. 81, 7845 (1997) Eq (20)
  //t_L1,L2*G<_L2,L1 is solved by -G<_L1,L1*t_L1,L2*gA_L2,L2*t_L2,L1-GR_L1,L1*t_L1,L2*g<_L2,L2*t_L2,L1
  //1. get t_L1,L2
  //1.1 get coupling Hamiltonian of this domain
  PetscMatrixParallelComplex* couple_Hamiltonian=NULL; //Hamiltonian matrix of this domain
  //std::set<unsigned int> Hamilton_momentum_indices;
  //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
  //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
  //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
  //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=momentum[*(pointer_to_Hamilton_momentum_indices->begin())];

  NEMO_ASSERT(options.check_option("exact_DOFmap_solver"),prefix+"please define \"exact_DOFmap_solver\"\n");
  Simulation* exact_DOFmap_solver=find_simulation(options.get_option("exact_DOFmap_solver",std::string("")));
  NEMO_ASSERT(exact_DOFmap_solver!=NULL,prefix+"have not found \""+options.get_option("exact_DOFmap_solver",std::string(""))+"\"\n");
  const DOFmapInterface& exact_DOFmap=exact_DOFmap_solver->get_const_dof_map(get_const_simulation_domain()); //dofmap for this domain
  NEMO_ASSERT(options.check_option("half_way_DOFmap_solver"),prefix+"please define \"half_way_DOFmap_solver\"\n");
  Simulation* half_DOFmap_solver=find_simulation(options.get_option("half_way_DOFmap_solver",std::string("")));
  NEMO_ASSERT(half_DOFmap_solver!=NULL,prefix+"have not found \""+options.get_option("half_way_DOFmap_solver",std::string(""))+"\"\n");
  const DOFmapInterface& half_DOFmap=half_DOFmap_solver->get_const_dof_map(get_const_simulation_domain()); //dofmap for neighbor domain

  DOFmapInterface* coupling_DOFmap = NULL;

  std::vector<NemoMeshPoint> sorted_momentum;
  QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,momentum_mesh_types,exact_DOFmap_solver);

  exact_DOFmap_solver->get_data(std::string("Hamiltonian"),sorted_momentum,half_DOFmap_solver->get_const_simulation_domain(),
                                couple_Hamiltonian,coupling_DOFmap); //coupling Hamiltonian between this domain and neighbor domain
  delete coupling_DOFmap;


  unsigned int num_rows = exact_DOFmap.get_global_dof_number(); //total number of dofs of this domain
  unsigned int num_local_rows = exact_DOFmap.get_number_of_dofs(); //local number of dofs of this domain
  unsigned int super_rows = couple_Hamiltonian->get_num_rows();
  unsigned int num_cols = super_rows-num_rows;

  //1.2 get coupling Hamiltonian t_L1,L2 from this domain coupling Hamiltonian
  PetscMatrixParallelComplex* couple_L1L2=new PetscMatrixParallelComplex(num_rows,
      num_cols,get_simulation_domain()->get_communicator());
  couple_L1L2->set_num_owned_rows(num_local_rows);
  for(unsigned int i=0; i<num_local_rows; i++)
    couple_L1L2->set_num_nonzeros_for_local_row(i,couple_Hamiltonian->get_nz_diagonal(i),couple_Hamiltonian->get_nz_offdiagonal(i));
  couple_L1L2->allocate_memory();
  couple_L1L2->set_to_zero();

  std::vector<int> temp_rows(num_rows);
  std::vector<int> temp_cols(num_cols);
  for(unsigned i=0; i<num_rows; i++)
    temp_rows[i]=i;
  for(unsigned i=0; i<num_cols; i++)
    temp_cols[i]=i+num_rows;
  couple_Hamiltonian->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,couple_L1L2);
  couple_L1L2->assemble();
  couple_L1L2->save_to_matlab_file("coupleH.m");
  delete couple_Hamiltonian;

  PetscMatrixParallelComplex couple_trans(couple_L1L2->get_num_cols(),couple_L1L2->get_num_rows(),
                                          couple_L1L2->get_communicator());
  couple_L1L2->hermitian_transpose_matrix(couple_trans, MAT_INITIAL_MATRIX); //t_L1,L2'

  //2. get G<_L1 matrix
  //2.1 get G< of this domain
  std::string Gl_propagator;
  NEMO_ASSERT(options.check_option("exact_lesser_green"),prefix+"please define \"exact_lesser_green\"\n");
  Gl_propagator=options.get_option("exact_lesser_green",std::string(""));
  Simulation* source_of_Gl = find_source_of_data(Gl_propagator);
  Propagator* Gl_pointer;
  source_of_Gl->get_data(Gl_propagator,Gl_pointer);
  PetscMatrixParallelComplex* Glesser=NULL;
  source_of_Gl->get_data(Gl_propagator,&momentum,Glesser,&exact_DOFmap);
  Glesser->save_to_matlab_file("GL.m");

  //2.2 G<_L1*t_L1,L2
  PetscMatrixParallelComplex* GLT=NULL;
  PetscMatrixParallelComplex::mult(*Glesser,*couple_L1L2,&GLT);

  //3. get gR_L2 matrix
  //3.1 get gR of this domain
  std::string gr_propagator;
  NEMO_ASSERT(options.check_option("half_way_retarded_green"),prefix+"please define \"half_way_retarded_green\"\n");
  gr_propagator=options.get_option("half_way_retarded_green",std::string(""));
  Simulation* source_of_gr = find_source_of_data(gr_propagator);
  Propagator* gr_pointer;
  source_of_gr->get_data(gr_propagator,gr_pointer);
  PetscMatrixParallelComplex* gR=NULL;
  source_of_gr->get_data(gr_propagator,&momentum,gR,&half_DOFmap);
  gR->save_to_matlab_file("gR.m");

  //obtain gA
  PetscMatrixParallelComplex gA(gR->get_num_cols(),gR->get_num_rows(),
                                gR->get_communicator());
  gR->hermitian_transpose_matrix(gA, MAT_INITIAL_MATRIX); //gA=gR'

  //3.2 gA_L2,L2*t_L2,L1
  PetscMatrixParallelComplex* gAT=NULL;
  PetscMatrixParallelComplex::mult(gA,couple_trans,&gAT);

  //3.3 G<_L1*t_L1,L2*gA_L2,L2*t_L2,L1
  PetscMatrixParallelComplex* temp1=NULL;
  PetscMatrixParallelComplex::mult(*GLT,*gAT,&temp1);
  delete GLT;
  delete gAT;

  //4. get GR_L1 matrix
  //4.1 get GR of this domain
  std::string GR_propagator;
  NEMO_ASSERT(options.check_option("exact_retarded_green"),prefix+"please define \"exact_retarded_green\"\n");
  GR_propagator=options.get_option("exact_retarded_green",std::string(""));
  Simulation* source_of_GR = find_source_of_data(GR_propagator);
  Propagator* GR_pointer;
  source_of_GR->get_data(GR_propagator,GR_pointer);
  PetscMatrixParallelComplex* GR=NULL;
  source_of_GR->get_data(GR_propagator,&momentum,GR,&exact_DOFmap);
  GR->save_to_matlab_file("GR.m");

  //4.2 GR_L1*t_L1,L2
  PetscMatrixParallelComplex* GRT=NULL;
  PetscMatrixParallelComplex::mult(*GR,*couple_L1L2,&GRT);

  //5. get g<_L2 matrix
  //5.1 get g< of this domain
  std::string gl_propagator;
  NEMO_ASSERT(options.check_option("half_way_lesser_green"),prefix+"please define \"half_way_lesser_green\"\n");
  gl_propagator=options.get_option("half_way_lesser_green",std::string(""));
  Simulation* source_of_gl = find_source_of_data(gl_propagator);
  Propagator* gl_pointer;
  source_of_gl->get_data(gl_propagator,gl_pointer);
  PetscMatrixParallelComplex* glesser=NULL;
  source_of_gl->get_data(gl_propagator,&momentum,glesser,&half_DOFmap);
  glesser->save_to_matlab_file("gL.m");

  //5.2 g<_L2,L2*t_L2,L1
  PetscMatrixParallelComplex* gLT=NULL;
  PetscMatrixParallelComplex::mult(*glesser,couple_trans,&gLT);

  //5.3 GR_L1*t_L1,L2*g<_L2,L2*t_L2,L1
  PetscMatrixParallelComplex* temp2=NULL;
  PetscMatrixParallelComplex::mult(*GRT,*gLT,&temp2);
  delete GRT;
  delete gLT;

  //6. calculate J=2*Re(Tr(t_L1,L2*G<_L2,L1))
  //6.1. -G<_L1,L1*t_L1,L2*gA_L2,L2*t_L2,L1-GR_L1,L1*t_L1,L2*g<_L2,L2*t_L2,L1
  temp1->add_matrix(*temp2,DIFFERENT_NONZERO_PATTERN);
  delete temp2;
  *temp1*=std::complex<double>(1.0,0.0); //HG
  temp1->save_to_matlab_file("HG.m");
  result = new PetscMatrixParallelComplex(*temp1); //store HG into result

  ////6.2. solve G<_L2,L1 offdiagonal with linear equation t_L1,L2*G<_L2,L1=HG
  ////set up the Linear solver
  //PetscMatrixParallelComplex temp_solution(couple_L1L2->get_num_rows(),temp1->get_num_cols(),
  //    get_simulation_domain()->get_communicator() );
  //temp_solution.consider_as_full();
  //temp_solution.allocate_memory();
  //LinearSolverPetscComplex solver(*couple_L1L2,*temp1,&temp_solution);
  ////prepare input options for the linear solver
  //InputOptions options_for_linear_solver;
  //if (options.check_option("linear_system_solver_for_Glesser"))
  //{
  //  string key("solver");
  //  string value = options.get_option("linear_system_solver_for_Glesser", string("mumps"));
  //  options_for_linear_solver[key] = value;
  //}
  //solver.set_options(options_for_linear_solver);
  //solver.solve();
  //result=new PetscMatrixParallelComplex(temp_solution); //store G<_L2,L1 into result

  //6.3. obtain J=2*Re(Tr(t_L1,L2*G<_L2,L1))
  std::complex<double> trace(std::complex<double>(0.0,0.0));
  trace=temp1->get_trace();
  delete temp1;
  double J=2*trace.real();
  J=J*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar/NemoMath::pi;
  return J;
}

void Propagation::calculate_spatial_current()
{
  std::string prefix = "Propagation(\""+get_name()+"\")::calculate_spatial_current ";
  //J=sum_k,E{J(k,E))*(2q/hbar*A)/(2pi)}
  //Ref: Lake et al, J.Appl.Phys. 81, 7845 (1997) Eq (20)
  std::map<double, std::map< pair<unsigned int, unsigned int>, double > > energy_current;
  std::map< pair<unsigned int, unsigned int>, double > integrated_current;
  //1. loop over energy (and k if any) and solve current
  //1.1 find G< propagator and solver
  std::string name_of_propagator;
  if(options.check_option("density_of"))
    name_of_propagator=options.get_option("density_of",std::string(""));
  else
  {
    //loop over all writeable Green's functions
    //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.begin();
    //for(; it!=writeable_Propagators.end(); ++it)
    {
      if(Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Fermion_lesser_Green ||
          Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Boson_lesser_Green)
      {
        name_of_propagator=name_of_writeable_Propagator;
      }
    }
  }
  Simulation* source_of_matrices = find_source_of_data(name_of_propagator);
  Propagator* Propagator_pointer;
  source_of_matrices->get_data(name_of_propagator,Propagator_pointer);

  if(Mesh_tree_topdown.size()==0)
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  //some MPI information required in multiple places
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,prefix+"Mesh_tree_topdown is not ready for usage\n");
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  int myrank;
  MPI_Comm_rank(topcomm, &myrank);
  std::stringstream convert_rank;
  convert_rank<<myrank;

  //1.2 obtain momentum_mesh from G< propagator
  Propagator::PropagatorMap::const_iterator momentum_c_it=Propagator_pointer->propagator_map.begin();
  //loop over all momentum k,E, solve current for each k,E
  for(; momentum_c_it!=Propagator_pointer->propagator_map.end(); ++momentum_c_it)
  {
    const std::vector<NemoMeshPoint> temp_vector_pointer=momentum_c_it->first;
    double energy=PropagationUtilities::read_energy_from_momentum(this,momentum_c_it->first,Propagator_pointer);
    //get the integration weight for this momentum from the momentum constructor(s)
    double integration_weight=1.0;
    double total_integration_weight=1.0;

    //1.3 find the integration weight for this specific momentum
    std::string momentum_name=std::string("");
    for(unsigned int i=0; i<Propagator_pointer->momentum_mesh_names.size(); i++)
    {
      momentum_name=Propagator_pointer->momentum_mesh_names[i];
      //find the momentum mesh constructor and ask it for the weight
      std::map<std::string,Simulation*>::const_iterator c_it2 = Propagation::Mesh_Constructors.begin();
      c_it2 = Propagation::Mesh_Constructors.find(momentum_name);
      NEMO_ASSERT(c_it2!=Propagation::Mesh_Constructors.end(),prefix+"have not found constructor for momentum mesh \""+momentum_name+"\"\n");
      if(Propagator_pointer->momentum_mesh_names[i].find("energy")==std::string::npos)
      {
        double temp=1.0;
        //check whether the momentum solver is providing a non rectangular mesh
        InputOptions& mesh_options=c_it2->second->get_reference_to_options();

        if(mesh_options.get_option(std::string("non_rectangular"),false))
        {
          c_it2->second->get_data(std::string("integration_weight"),momentum_c_it->first,(momentum_c_it->first)[i],temp);
        }
        else
        {
          c_it2->second->get_data("integration_weight",(momentum_c_it->first)[i],temp);
        }
        total_integration_weight*=temp;
        integration_weight*=temp;
        if(momentum_name.find("momentum_1D")!=std::string::npos)
        {
          total_integration_weight/=2.0*NemoMath::pi;
          integration_weight/=2.0*NemoMath::pi;
        }
        else if(momentum_name.find("momentum_2D")!=std::string::npos)
        {
          total_integration_weight/=4.0*NemoMath::pi*NemoMath::pi;
          integration_weight/=4.0*NemoMath::pi*NemoMath::pi;
        }
        else if(momentum_name.find("momentum_3D")!=std::string::npos)
        {
          total_integration_weight/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
          integration_weight/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
        }
      }
      else
      {
        double temp=1.0/2.0/NemoMath::pi;
        //double temp=1.0;
        //check whether the momentum solver is providing a non rectangular mesh
        InputOptions& mesh_options=c_it2->second->get_reference_to_options();
        if(mesh_options.get_option(std::string("non_rectangular"),false))
        {
          std::vector<NemoMeshPoint> temp_momentum(1,(momentum_c_it->first)[0]); //this assumes that i==1 and adaptiveGrid is called
          c_it2->second->get_data(std::string("integration_weight"),temp_momentum,(momentum_c_it->first)[i],temp);
        }
        else
          c_it2->second->get_data("integration_weight",(momentum_c_it->first)[i],temp);
        total_integration_weight*=temp;
      }
    }


    //2. calculate J(k,E)
    std::map< pair<unsigned int, unsigned int>, double >* current=NULL;
    calculate_spatial_current_for_momentum(momentum_c_it->first,current); //[A/eV]












    //2.2 fill the energy resolved current density
    std::map<double, std::map< pair<unsigned int, unsigned int>, double > >::iterator fill_energy_current_it=energy_current.find(energy);
    if(fill_energy_current_it==energy_current.end())
    {
      std::map< pair<unsigned int, unsigned int>, double >::const_iterator temp_cit=current->begin();
      std::map< pair<unsigned int, unsigned int>, double > temp_map;
      for(; temp_cit!=current->end(); ++temp_cit)
        temp_map[temp_cit->first]=0.0;
      energy_current[energy]=temp_map;
      fill_energy_current_it=energy_current.find(energy);
    }
    std::map< pair<unsigned int, unsigned int>, double >::iterator fill_int_current_it=current->begin();
    for(; fill_int_current_it!=current->end(); ++fill_int_current_it)
    {
      std::map< pair<unsigned int, unsigned int>, double >::iterator temp_it=fill_energy_current_it->second.find(fill_int_current_it->first);
      if(temp_it!=fill_energy_current_it->second.end())
        temp_it->second+=fill_int_current_it->second*integration_weight;
      else
        (fill_energy_current_it->second)[fill_int_current_it->first]=fill_int_current_it->second*integration_weight;
    }

    delete current;
  }//end of loop over (E,k)

  //test output
  if(debug_output&&!no_file_output)
  {
    std::string fname="fully_resolved_spatial_current_"+convert_rank.str()+".dat";
    const char* filename = fname.c_str();
    ofstream fo;
    fo.open(filename);
    std::map<double, std::map< pair<unsigned int, unsigned int>, double > >::const_iterator out_cit=energy_current.begin();
    for(; out_cit!=energy_current.end(); ++out_cit)
    {
      double energy=out_cit->first;
      fo<< "%energy point: ";
      fo<< energy<< "\n";
      std::map< pair<unsigned int, unsigned int>, double >::const_iterator c_it=out_cit->second.begin();
      for(; c_it!=out_cit->second.end(); c_it++)
        fo<<(c_it->first).first<<" "<<(c_it->first).second<<" "<<c_it->second<<"\n";
      fo<<"\n";
    }
    fo.close();
  }

  // MPI-reduce the result and store result locally
  //find all existing energies - on all MPI processes
  std::set<std::vector<NemoMeshPoint> >::iterator c_it2=pointer_to_all_momenta->begin();
  std::set<double> temp_all_energies;
  for(; c_it2!=pointer_to_all_momenta->end(); c_it2++)
  {
    double temp_energy = PropagationUtilities::read_energy_from_momentum(this,*c_it2, Propagator_pointer);
    //double temp_energy = read_energy_from_momentum(*c_it2, writeable_Propagators.begin()->second);
    temp_all_energies.insert(temp_energy);
  }
  //save the ordered energies into a vector and save the index-mapping
  std::vector<double> all_energies(temp_all_energies.size());
  std::map<double, int> translation_map_energy_index;
  int counter = 0;
  std::set<double>::iterator e_it;
  for(e_it=temp_all_energies.begin(); e_it!=temp_all_energies.end(); e_it++)
  {
    double temp_energy = *e_it;
    translation_map_energy_index[temp_energy]=counter;
    all_energies[counter]=temp_energy;
    counter++;
  }

  //fill the result vector with zeros or the values calculated within this MPI-process
  std::map<double, std::map< std::pair<unsigned int, unsigned int>, double > >::const_iterator c_it=energy_current.begin();
  std::map< std::pair<unsigned int, unsigned int>, double > temp_current(c_it->second);
  std::map< std::pair<unsigned int, unsigned int>, double >::iterator temp_it=temp_current.begin();
  for(; temp_it!=temp_current.end(); temp_it++)
    temp_it->second=0.0; //initialize all pairs to zero
  std::vector<std::map< std::pair<unsigned int, unsigned int>, double > > result_current(all_energies.size(),temp_current);
  std::map<double, std::map< std::pair<unsigned int, unsigned int>, double > >::const_iterator current_cit;
  counter=0;
  for(unsigned int i=0; i<result_current.size(); i++)
  {
    current_cit=energy_current.find(all_energies[i]);
    if(current_cit!=energy_current.end())
    {
      result_current[i]=current_cit->second;
      counter++;
    }
  }
  //translate the map-results into vectors
  //we use the fact that the keys of energy_current are sorted always the same way (on all MPI ranks)
  std::vector<double> current_vector(energy_current.begin()->second.size(),0.0);
  std::vector<std::vector<double> > energy_resolved_current_vector(all_energies.size(),current_vector);
  for(unsigned int i=0; i<all_energies.size(); i++)
  {
    double temp_energy=all_energies[i];
    std::map<double, std::map< pair<unsigned int, unsigned int>, double > >::const_iterator fill_c_it=energy_current.find(temp_energy);
    if(fill_c_it!=energy_current.end())
    {
      std::map< pair<unsigned int, unsigned int>, double >::const_iterator loop_cit=fill_c_it->second.begin();
      int j_counter=0;
      for(; loop_cit!=fill_c_it->second.end(); ++loop_cit)
      {
        if (particle_type_is_Fermion)
          energy_resolved_current_vector[i][j_counter]=loop_cit->second;
        else
          energy_resolved_current_vector[i][j_counter]=loop_cit->second*temp_energy;
        j_counter++;
      }
    }
    else
    {
      //create the energy entry when needed (setting all values to 0.0)
      std::map< pair<unsigned int, unsigned int>, double >::const_iterator loop_cit=energy_current.begin()->second.begin();
      std::map< pair<unsigned int, unsigned int>, double > temp_map;
      for(; loop_cit!=energy_current.begin()->second.end(); ++loop_cit)
        temp_map[loop_cit->first]=0.0;
      energy_current[temp_energy]=temp_map;
    }
  }








  //MPI-reduce the vectors
  if(solve_on_single_replica)
  {
    for(unsigned int i=0; i<energy_resolved_current_vector.size(); i++)
    {
      std::vector<double> temp_vector(energy_resolved_current_vector[i]);
      MPI_Reduce(&(temp_vector[0]),&(energy_resolved_current_vector[i][0]),energy_resolved_current_vector[0].size(),MPI_DOUBLE,MPI_SUM,0,topcomm);
    }


  }
  else
  {
    for(unsigned int i=0; i<energy_resolved_current_vector.size(); i++)
    {
      std::vector<double> temp_vector(energy_resolved_current_vector[i]);
      MPI_Allreduce(&(temp_vector[0]),&(energy_resolved_current_vector[i][0]),energy_resolved_current_vector[0].size(),MPI_DOUBLE,MPI_SUM,topcomm);
    }


  }



  //YuHe: fill the integrated current density
  for(unsigned int i=0; i<all_energies.size()-1; i++)
  {
    for(unsigned int j=0; j<current_vector.size(); j++)
      current_vector[j]+=(energy_resolved_current_vector[i][j]+energy_resolved_current_vector[i+1][j])*(all_energies[i+1]-all_energies[i])/2;
  }
  //if have one energy no reason to integrate
  if(all_energies.size()==1)
    for(unsigned int j=0; j<current_vector.size(); j++)
      current_vector[j] = energy_resolved_current_vector[0][j];

  //translate the vectors into maps
  std::map<double, std::map< std::pair<unsigned int, unsigned int>, double > >::iterator translate_back_it=energy_current.begin();
  for(unsigned int i=0; i<energy_resolved_current_vector.size(); i++)
  {
    NEMO_ASSERT(translate_back_it!=energy_current.end(),prefix+"reached end of energy_current\n");
    std::map< std::pair<unsigned int, unsigned int>, double >::iterator small_translate_back_it=translate_back_it->second.begin();
    NEMO_ASSERT(translate_back_it->second.size()==energy_resolved_current_vector[i].size(),prefix+"inconsistent size\n");
    for(unsigned int j=0; j<energy_resolved_current_vector[i].size(); j++)
    {
      NEMO_ASSERT(small_translate_back_it!=translate_back_it->second.end(),prefix+"reached end of energy_current-subsection\n");
      small_translate_back_it->second=energy_resolved_current_vector[i][j];
      ++small_translate_back_it;
    }
    ++translate_back_it;
  }
  translate_back_it=energy_current.begin();
  std::map< std::pair<unsigned int, unsigned int>, double >::iterator small_translate_back_it=translate_back_it->second.begin();
  for(unsigned int i=0; i<current_vector.size(); i++)
  {

    integrated_current[small_translate_back_it->first]=current_vector[i];
    ++small_translate_back_it;
  }

  if(options.get_option("check_current_conservation",false))
  {
    //3.check current conservation
    std::map<unsigned int, double >* summed_current_p=NULL;
    check_current_conservation(integrated_current,summed_current_p);
    std::map<unsigned int, double > summed_current(*summed_current_p);

    if(myrank==0&&!no_file_output)
    {
      std::string fname="current_conservation_";
      if(options.check_option("output_suffix"))
        fname=fname + "_" + get_output_suffix();
        //fname=fname+get_output_suffix("0");
      fname = fname + ".dat";
      const char* filename = fname.c_str();
      ofstream fo;
      fo.open(filename);
      fo<<"%check current conservation unit [A]\n";
      std::map<unsigned int, double >::iterator cout_it=summed_current.begin();
      for(; cout_it!=summed_current.end(); cout_it++)
        fo<<cout_it->first<<" "<<cout_it->second<<"\n";
      fo<<"\n";
      fo.close();

    //  print_atomic_map(summed_current,fname);
    }
    delete summed_current_p;
  }

  //test purpose
  if(myrank==0&&!no_file_output)
  {
    std::string fname="integrated_spatial_current";
    if(options.check_option("output_suffix"))
      fname=fname+"_"+ get_output_suffix();//options.get_option("output_suffix",std::string("0"));
    fname = fname + ".dat";
    const char* filename = fname.c_str();
    ofstream fo;
    fo.open(filename);
    fo<<"%integrated spatial current unit [A]\n";
    std::map< pair<unsigned int, unsigned int>, double >::iterator cout_it=integrated_current.begin();
    for(; cout_it!=integrated_current.end(); cout_it++)
      fo<<(cout_it->first).first<<" "<<(cout_it->first).second<<" "<<cout_it->second<<"\n";
    fo<<"\n";
    fo.close();
  }

  if(myrank==0&&options.get_option("spatial_energy_resolved_current",false))
  {
    std::string fname="energy_resolved_spatial_current.dat";
    const char* filename = fname.c_str();
    ofstream fo;
    fo.open(filename);

    fo << "%\t";
    //create header
    for(unsigned int i = 0; i < all_energies.size(); ++i)
    {
      fo<< all_energies[i] << "[eV] \t";
    }
    fo << "\n";
    //std::vector<std::vector<double> > energy_resolved_current_vector(all_energies.size(),current_vector);
    std::map< pair<unsigned int, unsigned int>, double >::iterator out_cit=integrated_current.begin();
    int count = 0;
    for(; out_cit!=integrated_current.end(); ++out_cit)
    {
      //std::map< pair<unsigned int, unsigned int>, double >::const_iterator c_it=out_cit->second.begin();
      //atom 1 and atom 2
      fo<<(out_cit->first).first<<"\t"<<(out_cit->first).second<<"\t";

      for(unsigned int i = 0; i < all_energies.size(); i++)
        fo<<energy_resolved_current_vector[i][count]<<"\t";
      fo<<"\n";
      ++count;
    }
    fo.close();
  }

}

void Propagation::calculate_spatial_current_for_momentum(const std::vector<NemoMeshPoint>& momentum_point,
    std::map< pair<unsigned int, unsigned int>, double >*& current)
{
  std::string prefix = "Propagation(\""+get_name()+"\")::calculate_spatial_current_for_momentum ";
  //1. decide resolution of spatial resolved current: atom-resolved or slab-resolved
  //bool atom_resolved = options.get_option("solve_atomic_resolved_current",true);
  //if(!atom_resolved)
  //{
    //   //1.1. if slab-resolved current is requested
    //   vector<string> subdomains; //vector of subdomains
    //   //1.1.1. read in vector of slabs (subdomains) from repartitioner
    //   if(options.check_option("repartition_solver_name"))
    //   {
    //     string repartition_solver_name = options.get_option("repartition_solver_name", string(""));
    //     Simulation* _repartition_solver = find_simulation(repartition_solver_name);
    //     _repartition_solver->get_data("subdomains",subdomains);
    //}
    //else
    //     throw std::runtime_error(prefix+"repartition solver not defined\n");

    //   //1.1.2. loop over neighbor slabs, solve J(k,E)
    //   current.resize(subdomains.size()-1);
    //for(unsigned int i=0;i<subdomains.size()-1;i++)
    //{
    //     //obtain dofmap of two neighbor subdomains
    //     NEMO_ASSERT(options.check_option("DOFmap_solver1"),prefix+"please define DOFmap solver for first slab\n");
    //     std::string schroed1_name = options.get_option("DOFmap_solver1",std::string(""));
    //     Simulation* schroed1 = find_simulation(schroed1_name);
    //     NEMO_ASSERT(schroed1!=NULL,prefix+"simulation \""+schroed1_name+"\" not found\n");
    //     NEMO_ASSERT(options.check_option("DOFmap_solver2"),prefix+"please define DOFmap solver for second slab\n");
    //     std::string schroed2_name = options.get_option("DOFmap_solver2",std::string(""));
    //     Simulation* schroed2 = find_simulation(schroed2_name);
    //     NEMO_ASSERT(schroed2!=NULL,prefix+"simulation \""+schroed2_name+"\" not found\n");

    //     const DOFmap& dof1 = schroed1->get_const_dof_map(get_const_simulation_domain());
    //     const DOFmap& dof2 = schroed2->get_const_dof_map(get_const_simulation_domain());
    //     double J=calculate_current_between_neighbors(&dof1,&dof2,momentum_c_it->first);
    //  current[i]=J*integration_weight*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar/NemoMath::pi;//[A/eV];
    //}
    //throw std::runtime_error(prefix+"slab-resolved current not support yet\n");
  //}
  //else
  {
    bool chk_current=options.get_option("check_current_conservation",false);
    //1.2. if atom-resolved current is requested
    //1.2.1. loop over all atoms within this domain, solve J(k,E)
    unsigned int atom_i_id; //atom i whose current is solved
    const AtomisticDomain* atomdomain  = dynamic_cast<const AtomisticDomain*> (Hamilton_Constructor->get_const_simulation_domain() ); //obtain this domain pointer
    const AtomicStructure& atoms = atomdomain->get_atoms(); //obtain all atoms of this domain
    const DOFmapInterface& dof_map = Hamilton_Constructor->get_dof_map(get_const_simulation_domain()); //obtain dofmap of this domain
    vector<const Atom_neighbour*> active_neighbors;    // active neighbours of atom i
    vector<const Atom_neighbour*> active_neighbors_2nd;    // active neighbours of atom j
    ConstActiveAtomIterator it  = atoms.active_atoms_begin(); //first active atom of this domain
    ConstActiveAtomIterator end = atoms.active_atoms_end(); //last active atom of this domain
    std::map< pair<unsigned int, unsigned int>, double > current_temp;
    std::map< pair<unsigned int, unsigned int>, double >::iterator c_it; //iterator for current map
    std::map< pair<unsigned int, unsigned int>, double >::iterator c_it2;
    for(; it!=end; ++it)
    {
      const AtomStructNode& atom_i_nd = it.node();
      atom_i_id = atom_i_nd.id; //id of atom i
      const map<short, unsigned int>* atom_i_dofs = dof_map.get_atom_dof_map(atom_i_id); //dofmap of atom i
      atomdomain->get_active_neighbours(atom_i_id, &active_neighbors);
      int NN = active_neighbors.size(); //all active neighbor atoms of atom i

      for(short jj=0; jj<NN; ++jj) //loop over all neighbors of atom i
      {
        unsigned int atom_j_id = active_neighbors[jj]->atom_id;
        pair<unsigned int, unsigned int> index_ij=std::make_pair(atom_i_id,atom_j_id); //atom ij pairs
        pair<unsigned int, unsigned int> index_ji=std::make_pair(atom_j_id,atom_i_id); //atom ji pairs

        c_it=current_temp.find(index_ij); //check if atom ij pair is solved
        if(c_it==current_temp.end()) //if atom ij pair is not solved
        {
          if(chk_current) //if check current conservation, then needs to solve all pairs
          {
            c_it2=current_temp.find(index_ji); //check if atom ji pair is solved
            if(c_it2==current_temp.end()) //if atom ji pair is not solved
            {
              // both pair ij and ji not solved, then solve it
              const map<short, unsigned int>* atom_j_dofs = dof_map.get_atom_dof_map(atom_j_id);
              double J=0;
              J=calculate_current_between_neighbors(atom_i_dofs,atom_j_dofs,momentum_point);//current between pari ij
              J=J*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar/NemoMath::pi/2;//[A/eV]
              current_temp[index_ij]=J; //store current in pair ij
            }
            else //if atom ji pair is solved
              current_temp[index_ij]=-c_it2->second; //store result of ji pair into pair ij, since they are the same
          }
          else //if not check current conservation
          {
            if(atom_i_id<atom_j_id) //only solve i<j, since Iij=Iji
            {
              c_it2=current_temp.find(index_ji); //check if atom ji pair is solved
              if(c_it2==current_temp.end()) //if atom ji pair is not solved
              {
                // both pair ij and ji not solved, then solve it
                const map<short, unsigned int>* atom_j_dofs = dof_map.get_atom_dof_map(atom_j_id);
                double J=0;
                J=calculate_current_between_neighbors(atom_i_dofs,atom_j_dofs,momentum_point);
                J=J*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar/NemoMath::pi/2;//[A/eV]
                current_temp[index_ij]=J; //store current in pair ij
              }
              else //if atom ji pair is solved
                current_temp[index_ij]=c_it2->second; //store result of ji pair into pair ij, since they are the same
            }
          }
        }
        atomdomain->get_active_neighbours(atom_j_id, &active_neighbors_2nd);
        int NN2nd = active_neighbors_2nd.size(); //all active neighbor atoms of atom i

        //following considers when 2NN exists
        for(short kk=0; kk<NN2nd; ++kk) //loop over all neighbors of atom i
        {
          unsigned int atom_k_id = active_neighbors_2nd[kk]->atom_id;
          pair<unsigned int, unsigned int> index_ik=std::make_pair(atom_i_id,atom_k_id); //atom ij pairs
          pair<unsigned int, unsigned int> index_ki=std::make_pair(atom_k_id,atom_i_id); //atom ji pairs
          if (atom_k_id != atom_i_id)
          {
            c_it=current_temp.find(index_ik); //check if atom ij pair is solved
            if(c_it==current_temp.end()) //if atom ij pair is not solved
            {
              if(chk_current) //if check current conservation, then needs to solve all pairs
              {
                c_it2=current_temp.find(index_ki); //check if atom ji pair is solved
                if(c_it2==current_temp.end()) //if atom ji pair is not solved
                {
                  // both pair ij and ji not solved, then solve it
                  const map<short, unsigned int>* atom_k_dofs = dof_map.get_atom_dof_map(atom_k_id);
                  double J=0;
                  J=calculate_current_between_neighbors(atom_i_dofs,atom_k_dofs,momentum_point);
                  J=J*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar/NemoMath::pi/2;//[A/eV]
                  current_temp[index_ik]=J; //store current in pair ij
                }
                else //if atom ji pair is solved
                current_temp[index_ik]=-c_it2->second; //store result of ji pair into pair ij, since they are the same
              }
              else //if not check current conservation
              {
                if(atom_i_id<atom_k_id) //only solve i<j, since Iij=Iji
                {
                  c_it2=current_temp.find(index_ki); //check if atom ji pair is solved
                  if(c_it2==current_temp.end()) //if atom ji pair is not solved
                  {
                    // both pair ij and ji not solved, then solve it
                    const map<short, unsigned int>* atom_k_dofs = dof_map.get_atom_dof_map(atom_k_id);
                    double J=0;
                    J=calculate_current_between_neighbors(atom_i_dofs,atom_k_dofs,momentum_point);
                    J=J*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar/NemoMath::pi/2;//[A/eV]
                    current_temp[index_ik]=J; //store current in pair ij
                  }
                  else //if atom ji pair is solved
                    current_temp[index_ik]=c_it2->second; //store result of ji pair into pair ij, since they are the same
                }
              }
            }
          }
        }
      }
    }
    current = new std::map< pair<unsigned int, unsigned int>, double >(current_temp);
  }
}


void Propagation::check_current_conservation(std::map< pair<unsigned int, unsigned int>, double > input_current, std::map<unsigned int, double >*& current)
{
  //YuHe: if current conserves, then the summed current over each atom is zero, except for the atoms at the device/contact interface
  std::string prefix="Propagation(\""+get_name()+"\")::check_current_conservation: ";
  //read-in information of atoms in this domain
  unsigned int atom_i_id; //atom i whose current is solved
  const AtomisticDomain* atomdomain  = dynamic_cast<const AtomisticDomain*> (Hamilton_Constructor->get_const_simulation_domain() ); //obtain this domain pointer
  const AtomicStructure& atoms = atomdomain->get_atoms(); //obtain all atoms of this domain
  //const DOFmap& dof_map = Hamilton_Constructor->get_dof_map(); //obtain dofmap of this domain
  vector<const Atom_neighbour*> active_neighbors;    // active neighbours of atom i
  vector<const Atom_neighbour*> active_neighbors_2nd;    // active 2nd nearest neighbours of atom i
  ConstActiveAtomIterator it  = atoms.active_atoms_begin(); //first active atom of this domain
  ConstActiveAtomIterator end = atoms.active_atoms_end(); //last active atom of this domain
  std::map<unsigned int, double > current_temp;

  std::map< pair<unsigned int, unsigned int>, double >::iterator c_it; //iterator for current map
  for(; it!=end; ++it)
  {
    const AtomStructNode& atom_i_nd = it.node();
    atom_i_id = atom_i_nd.id; //id of atom i
    //      const map<short, unsigned int>* atom_i_dofs = dof_map.get_atom_dof_map(atom_i_id); //dofmap of atom i
    atomdomain->get_active_neighbours(atom_i_id, &active_neighbors);
    int NN = active_neighbors.size(); //all active neighbor atoms of atom i
    double sum_current=0.0;

    for(short jj=0; jj<NN; ++jj) //loop over all neighbors of atom i
    {
      unsigned int atom_j_id = active_neighbors[jj]->atom_id;
      pair<unsigned int, unsigned int> index_ij=std::make_pair(atom_i_id,atom_j_id); //atom ij pairs
      //        pair<unsigned int, unsigned int> index_ji=std::make_pair(atom_j_id,atom_i_id); //atom ji pairs

      c_it=input_current.find(index_ij); //check if atom ij pair is solved
      if(c_it!=input_current.end()) //if atom ij pair is found
        sum_current=sum_current+c_it->second; //sum over current of ij pair

      atomdomain->get_active_neighbours(atom_j_id, &active_neighbors_2nd);
      int NN2nd = active_neighbors_2nd.size(); //all active neighbor atoms of atom j
      for(short kk=0; kk<NN2nd; ++kk) //loop over all neighbors of atom j
      {
        unsigned int atom_k_id = active_neighbors_2nd[kk]->atom_id;
        pair<unsigned int, unsigned int> index_ik=std::make_pair(atom_i_id,atom_k_id); //atom ik pairs

        c_it=input_current.find(index_ik); //check if atom ik pair is solved
        if(c_it!=input_current.end()) //if atom ij pair is found
          sum_current=sum_current+c_it->second; //sum over current of ik pair
      }
    }
    current_temp[atom_i_id]=sum_current;
  }
  current = new std::map<unsigned int, double >(current_temp);
}

void Propagation::calculate_local_current()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_local_current ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+get_name()+"\")::calculate_local_current: ";
  std::map<double, std::vector<std::complex<double> > > current_x;
  std::map<double, std::vector<std::complex<double> > > current_y;
  std::map<double, std::vector<std::complex<double> > > current_z;

  //J(E)=real(diag(v_matrix*S*G_lesser))*(q/h)
  //1 get the lesser Green's function G_lesser at energy point
  //1.1 find out of which Green's function the density should be calculated (either the writeable lesser_Green of this Propagation or inputdeck given)
  std::string name_of_propagator;
  if(options.check_option("density_of"))
    name_of_propagator=options.get_option("density_of",std::string(""));
  else
  {
    //loop over all writeable Green's functions
    //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.begin();
    //for(; it!=writeable_Propagators.end(); ++it)
    {
      if(Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Fermion_lesser_Green ||
          Propagator_types.find(name_of_writeable_Propagator)->second==NemoPhys::Boson_lesser_Green)
      {
        name_of_propagator=name_of_writeable_Propagator;
      }
    }
  }

  //1.2 find the source of the lesser Green's function
  Simulation* source_of_matrices = find_source_of_data(name_of_propagator);
  Propagator* Propagator_pointer;
  source_of_matrices->get_data(name_of_propagator,Propagator_pointer);
  Propagator::PropagatorMap::const_iterator momentum_c_it=Propagator_pointer->propagator_map.begin();
  //PetscMatrixParallelComplex* Glesser_matrix;
  //source_of_matrices->get_data(name_of_propagator,&(momentum_c_it->first),Glesser_matrix,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));



  //integrate with respect to the (non-energy) momenta
  //1.2.1 loop over all momenta
  std::vector<double> result_x1;
  std::vector<double> result_y1;
  std::vector<double> result_z1;
  for(; momentum_c_it!=Propagator_pointer->propagator_map.end(); ++momentum_c_it)
  {
    const std::vector<NemoMeshPoint> temp_vector_pointer=momentum_c_it->first;
    //1.2.2 use source_of_data->get_data to get reading access to the matrix of this respective momentum
    PetscMatrixParallelComplex* Glesser_matrix;
    source_of_matrices->get_data(name_of_propagator,&(momentum_c_it->first),Glesser_matrix,
                                 &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    //get the integration weight for this momentum from the momentum constructor(s)
    double integration_weight=1.0;
    std::string momentum_name=std::string("");

    for (unsigned int i = 0; i < Propagator_pointer->momentum_mesh_names.size(); i++)
    {
      //find the momentum name that does not contintegration_weightain "energy"
      if (Propagator_pointer->momentum_mesh_names[i].find("energy") != std::string::npos)
      {
        momentum_name = Propagator_pointer->momentum_mesh_names[i];
        //find the momentum mesh constructor and ask it for the weight
        std::map<std::string, Simulation*>::const_iterator c_it2 = Propagation::Mesh_Constructors.begin();
        c_it2 = Propagation::Mesh_Constructors.find(momentum_name);
        NEMO_ASSERT(c_it2!=Propagation::Mesh_Constructors.end(), prefix+"have not found constructor for momentum mesh \""+momentum_name+"\"\n");
        double temp;
        //check whether the momentum solver is providing a non rectangular mesh
        InputOptions& mesh_options = c_it2->second->get_reference_to_options();

        if (mesh_options.get_option(std::string("non_rectangular"), false))
          c_it2->second->get_data(std::string("integration_weight"), (momentum_c_it->first), (momentum_c_it->first)[i], temp);
        else
          c_it2->second->get_data("integration_weight", (momentum_c_it->first)[i], temp);

        integration_weight *= temp;

        if(momentum_name.find("momentum_1D")!=std::string::npos)
          integration_weight/=2.0*NemoMath::pi;
        if(momentum_name.find("momentum_2D")!=std::string::npos)
          integration_weight/=4.0*NemoMath::pi*NemoMath::pi;
        else if(momentum_name.find("momentum_3D")!=std::string::npos)
          integration_weight/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
      }
    }



    //2. get velocity operator
    //2.1 find the indices of the momenta that are constructed by the Hamiltonian_Constructor
    PetscMatrixParallelComplex* velocity_x = NULL;
    PetscMatrixParallelComplex* velocity_y = NULL;
    PetscMatrixParallelComplex* velocity_z = NULL;
    std::set<unsigned int> Hamilton_momentum_indices;
    std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
    PropagationUtilities::find_Hamiltonian_momenta(this,writeable_Propagator,pointer_to_Hamilton_momentum_indices);
    NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
    if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=temp_vector_pointer[*(pointer_to_Hamilton_momentum_indices->begin())];
    Hamilton_Constructor->get_data(std::string("velocity_operator_x"),temp_NemoMeshPoint,velocity_x);
    Hamilton_Constructor->get_data(std::string("velocity_operator_y"),temp_NemoMeshPoint,velocity_y);
    Hamilton_Constructor->get_data(std::string("velocity_operator_z"),temp_NemoMeshPoint,velocity_z);
    if(options.get_option("debug_output",false))
    {
      velocity_x->save_to_matlab_file("local_current_velocity_x.m");
      velocity_y->save_to_matlab_file("local_current_velocity_y.m");
      velocity_z->save_to_matlab_file("local_current_velocity_z.m");
    }

    //get the S matrix
    PetscMatrixParallelComplex* SxGL=NULL;
    PetscMatrixParallelComplex* S_matrix=NULL;
    Hamilton_Constructor->get_data(std::string("dimensional_S_matrix"),S_matrix);//for n-dim periodicity: [1/nm^(3-n)]
    PetscMatrixParallelComplex::mult(*S_matrix,*Glesser_matrix,&SxGL);
    if(options.get_option("debug_output",false))
    {
      S_matrix->save_to_matlab_file("local_current_S_matrix.m");
      Glesser_matrix->save_to_matlab_file("local_current_Glesser_matrix.m");
    }
    delete S_matrix;
    S_matrix = NULL;
    //delete Glesser_matrix;
    //Glesser_matrix = NULL;

    PetscMatrixParallelComplex* temp1 = NULL;
    PetscMatrixParallelComplex::mult(*velocity_x,*SxGL,&temp1);//v_matrix*G_lesser
    //velocity_x->hermitian_transpose_matrix(*velocity_x,MAT_REUSE_MATRIX);//v_matrix'
    //PetscMatrixParallelComplex::mult(*Gl1,*velocity_x,&temp2);//G_lesser*v_matrix'
    //temp1->add_matrix(*temp2, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));//v_matrix*G_lesser-G_lesser*v_matrix'
    *temp1 *=std::complex<double>(integration_weight*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::h,0.0);//[A*m/eV]
   //temp1->save_to_matlab_file("Jx.m");
    std::vector<std::complex<double> > diagonal;
    temp1->get_diagonal(&diagonal);//J=diag(v_matrix*G_lesser-G_lesser*v_matrix')//[A*m/eV]


    if(result_x1.size()==0)
    {
      result_x1.resize(diagonal.size(),0.0);

    }
    for(unsigned int i=0;i<diagonal.size();i++)
    {
      result_x1[i]+=diagonal[i].real();
    }


    //y direction
    delete temp1;
    temp1=NULL;
    //delete temp2;
    //temp2=NULL;
    PetscMatrixParallelComplex::mult(*velocity_y,*SxGL,&temp1);//v_matrix*G_lesser
   // velocity_y->hermitian_transpose_matrix(*velocity_y,MAT_REUSE_MATRIX);//v_matrix'
    //PetscMatrixParallelComplex::mult(*Glesser_matrix,*velocity_y,&temp2);//G_lesser*v_matrix'
    //temp1->add_matrix(*temp2, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));//v_matrix*G_lesser-G_lesser*v_matrix'
    *temp1 *=std::complex<double>(integration_weight*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::h,0.0);//[A*m/eV]
    temp1->get_diagonal(&diagonal);//J=diag(v_matrix*G_lesser-G_lesser*v_matrix')//[A*m/eV]
    //temp1->save_to_matlab_file("Jy.m");
    if(result_y1.size()==(unsigned int)0)
    {
      result_y1.resize(diagonal.size(),0.0);

    }
    for(unsigned int i=0;i<diagonal.size();i++)
    {
      result_y1[i]+=diagonal[i].real();
    }



    //z direction
    delete temp1;
    temp1=NULL;
    //delete temp2;
    //temp2=NULL;
    PetscMatrixParallelComplex::mult(*velocity_z,*SxGL,&temp1);//v_matrix*G_lesser
    //velocity_z->hermitian_transpose_matrix(*velocity_z,MAT_REUSE_MATRIX);//v_matrix'
    //PetscMatrixParallelComplex::mult(*Glesser_matrix,*velocity_z,&temp2);//G_lesser*v_matrix'
    //temp1->add_matrix(*temp2, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));//v_matrix*G_lesser-G_lesser*v_matrix'
    *temp1 *=std::complex<double>(integration_weight*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::h,0.0);//[A*m/eV]
    temp1->get_diagonal(&diagonal);//J=diag(v_matrix*G_lesser-G_lesser*v_matrix')//[A*m/eV]
    //temp1->save_to_matlab_file("Jz.m");

    if(result_z1.size()==(unsigned int)0)
    {
      result_z1.resize(diagonal.size(),0.0);

    }
    for(unsigned int i=0;i<diagonal.size();i++)
    {
      result_z1[i]+=diagonal[i].real();
    }



    delete SxGL;
    SxGL = NULL;
    delete temp1;
    temp1 = NULL;
    delete velocity_x;
    delete velocity_y;
    delete velocity_z;
  }




   //call MPI_allreduce for this vector
   NemoMesh* energy_mesh=Mesh_tree_topdown.begin()->first;
   MPI_Barrier(energy_mesh->get_global_comm());
   MPI_Allreduce(MPI_IN_PLACE,&(result_x1[0]),result_x1.size(),MPI_DOUBLE, MPI_SUM, energy_mesh->get_global_comm());
   MPI_Allreduce(MPI_IN_PLACE,&(result_y1[0]),result_y1.size(),MPI_DOUBLE, MPI_SUM, energy_mesh->get_global_comm());
   MPI_Allreduce(MPI_IN_PLACE,&(result_z1[0]),result_z1.size(),MPI_DOUBLE, MPI_SUM, energy_mesh->get_global_comm());


  //7. store the results in the result map
  //if the DOFmap size does not agree with the size of vector, we ask the matrix source how to translate the vector index into the atom_id
  //otherwise, we assume that there is a one to one correspondence
  //7a. get the DOFmap of the matrix source
  const DOFmapInterface& temp_DOFmap=source_of_matrices->get_const_dof_map(get_const_simulation_domain());
  //7b. compare the size of the DOFMap with the vector dimension
  const unsigned int number_of_DOFs=temp_DOFmap.get_number_of_dofs();
  //x direction
  if(number_of_DOFs>result_x1.size())
  {
    atomic_output_only = true;
    //here, a transformation has contracted the result onto atoms only
    //7c. get the map between vector index (which is the iterator number of the respective active atom) and the atom_id from DOFmap
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);

    //7d. multiply result with prefactor and store in map (key is atom_id, value is density)
    for (unsigned int i=0; i<result_x1.size(); i++)
    {
      std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(i);
      NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),prefix+"have not found the atom_id\n");
      local_current_x[c_it->second]=result_x1[i];
    }
  }
  else if(number_of_DOFs==result_x1.size())
  {
    atomic_output_only = false;
    //this is for orbital resolved output
    for (unsigned int i=0; i<result_x1.size(); i++)
      local_current_x[i]=result_x1[i];

    //one exception is possible: if the original representation has only one DOF per atom (as in effective mass)
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map);
    const std::map<short,unsigned int>* temp_map=temp_DOFmap.get_atom_dof_map(index_to_atom_id_map.begin()->first);
    if(temp_map->size()==1)
      atomic_output_only=true;
  }
  else
    throw std::runtime_error(prefix+"mismatch of number of DOFs and resulting density size\n");

  //y direction
  if(number_of_DOFs>result_y1.size())
  {
    atomic_output_only = true;
    //here, a transformation has contracted the result onto atoms only
    //7c. get the map between vector index (which is the iterator number of the respective active atom) and the atom_id from DOFmap
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);

    //7d. multiply result with prefactor and store in map (key is atom_id, value is density)
    for (unsigned int i=0; i<result_y1.size(); i++)
    {
      std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(i);
      NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),prefix+"have not found the atom_id\n");
      local_current_y[c_it->second]=result_y1[i];
    }
  }
  else if(number_of_DOFs==result_y1.size())
  {
    atomic_output_only = false;
    //this is for orbital resolved output
    for (unsigned int i=0; i<result_y1.size(); i++)
      local_current_y[i]=result_y1[i];

    //one exception is possible: if the original representation has only one DOF per atom (as in effective mass)
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map);
    const std::map<short,unsigned int>* temp_map=temp_DOFmap.get_atom_dof_map(index_to_atom_id_map.begin()->first);
    if(temp_map->size()==1)
      atomic_output_only=true;
  }
  else
    throw std::runtime_error(prefix+"mismatch of number of DOFs and resulting density size\n");

  //z direction
  if(number_of_DOFs>result_z1.size())
  {
    atomic_output_only = true;
    //here, a transformation has contracted the result onto atoms only
    //7c. get the map between vector index (which is the iterator number of the respective active atom) and the atom_id from DOFmap
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);

    //7d. multiply result with prefactor and store in map (key is atom_id, value is density)
    for (unsigned int i=0; i<result_z1.size(); i++)
    {
      std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(i);
      NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),prefix+"have not found the atom_id\n");
      local_current_z[c_it->second]=result_z1[i];
    }
  }
  else if(number_of_DOFs==result_z1.size())
  {
    atomic_output_only = false;
    //this is for orbital resolved output
    for (unsigned int i=0; i<result_z1.size(); i++)
      local_current_z[i]=result_z1[i];

    //one exception is possible: if the original representation has only one DOF per atom (as in effective mass)
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map);
    const std::map<short,unsigned int>* temp_map=temp_DOFmap.get_atom_dof_map(index_to_atom_id_map.begin()->first);
    if(temp_map->size()==1)
      atomic_output_only=true;
  }
  else
    throw std::runtime_error(prefix+"mismatch of number of DOFs and resulting density size\n");

  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::calculate_energy_interpolated_current()
{
  if(!options.get_option("energy_interpolated_output",false))
  {
    return;
  }
  NemoMesh* energy_mesh=Mesh_tree_topdown.begin()->first;
  MPI_Barrier(energy_mesh->get_global_comm());
  int myrank;
  myrank=get_simulation_domain()->get_geometry_replica();
  int energy_rank;
  MPI_Comm_rank(energy_mesh->get_global_comm(),&energy_rank);
  if((myrank==0)&&(energy_rank==0)&&!no_file_output)
  {
    //read in the temperature and the chemical potential for all contacts -- assume there are only two contacts at this moment
    //double temperature1=options.get_option("temperature1", NemoPhys::temperature);
    //temperature1*=NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
    //double chemical_potential1=options.get_option("chemical_potential1", 0.0);
    //double temperature2=options.get_option("temperature2", NemoPhys::temperature);
    //temperature2*=NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
    //double chemical_potential2=options.get_option("chemical_potential2", 0.0);
    energy_resolved_current_interp.clear();
    double temp_energy_resolved_current=0.0;
    std::stringstream stm;
    stm<<chemical_potential1-chemical_potential2;
    std::string potential_diff = stm.str();
    ofstream out_file;
    string filename=get_name()+"_"+potential_diff+"_energy_interpolated_current_" + get_output_suffix() + ".dat";
    out_file.open(filename.c_str());
    out_file.precision(10);
    //ofstream out_file_debug;
    // filename=get_name()+"_"+potential_diff+"_energy_interpolated_current_debug_" + get_output_suffix() + ".dat";
    //out_file_debug.open(filename.c_str());
    //out_file_debug.precision(10);
    out_file<<"Energy (ev)"<<"\t"<<"Current (A)"<<"\n";
    //out_file_debug<<"energy"<<",\t "<<"temperature1"<<",\t "<<"chemical_potential1"<<",\t "<<"fermi_factor_1"<<",\t "<<"temperature2"
    //    <<",\t "<<"chemical_potential2"<<",\t "<<"fermi_factor_2"<<",\t "<< "T "<< ",\t "<<"current"<<"\n";
    std::map<double,double>::iterator   it =energy_resolved_transmission_interpolated.begin();
    double temp_energy,fermi_factor_1 ,fermi_factor_2;
    for(; it!=energy_resolved_transmission_interpolated.end(); it++)
    {
      temp_energy=it->first;
      fermi_factor_1 = NemoMath::fermi_distribution(chemical_potential1,temperature1_in_eV,temp_energy);
      fermi_factor_2 = NemoMath::fermi_distribution(chemical_potential2,temperature2_in_eV,temp_energy);
      temp_energy_resolved_current=(it->second*NemoPhys::elementary_charge/NemoPhys::h*fermi_factor_1);
      temp_energy_resolved_current-=(it->second*NemoPhys::elementary_charge/NemoPhys::h*fermi_factor_2);
      energy_resolved_current_interp[temp_energy]=temp_energy_resolved_current*NemoPhys::elementary_charge;
      out_file<<temp_energy<<"\t"<<energy_resolved_current_interp[temp_energy]<<"\n";
      // out_file_debug<<temp_energy<<",\t "<<temperature1<<",\t "<<chemical_potential1<<",\t "<<fermi_factor_1<<",\t "<<temperature2<<",\t "<<chemical_potential2<<",\t "<<fermi_factor_2<<",\t "<< it->second<<",\t" <<energy_resolved_current_interp[temp_energy]<<",\t "<< NemoPhys::elementary_charge<<",\t "<< NemoPhys::h<<"\n";
    }
    out_file.close();
    //out_file_debug.close();
  }
}



void Propagation::calculate_current()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_current ");
  NemoUtils::tic(tic_toc_prefix);

  if(momentum_transmission.empty() || energy_resolved_transmission.empty())
    calculate_transmission();

  //0. determine whether all meshes are rectangular or not
  bool rectangular_mesh_used=false;
  //0.1 loop over all mesh_constructors and check whether one of them has the option ("non_rectangular = true")

  std::map<std::string, Simulation*>::const_iterator mesh_cit=Mesh_Constructors.begin();
  for(; mesh_cit!=Mesh_Constructors.end() && !rectangular_mesh_used; ++mesh_cit)
  {
    InputOptions& mesh_options = mesh_cit->second->get_reference_to_options();
    if(mesh_options.get_option(std::string("non_rectangular"),false))
      rectangular_mesh_used=true;
  }

  if(!rectangular_mesh_used)
    //we can use the T(E) as is and integrate
    calculate_rectangular_mesh_current();
  else
    //need to re-do the integration of transmission and integrate appropriately
    calculate_nonrectangular_mesh_current();

  NemoUtils::toc(tic_toc_prefix);

}


void Propagation::calculate_rectangular_mesh_current()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_rectangular_mesh_current ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::calculate_rectangular_mesh_current ";

  //double temperature1_eV = temperature1*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  //double temperature2_eV = temperature1*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  //calculate the current density at each energy point
  current=0.0;
  std::map<double,double> energy_resolved_current;

  //1.2 find the energy mesh pointer
  NemoMesh* energy_mesh=NULL;
  Simulation* energy_mesh_constructor=NULL;
  std::map<std::string, NemoMesh*>::const_iterator temp_cit=Momentum_meshes.begin();
  for (; temp_cit!=Momentum_meshes.end(); temp_cit++)
    if(temp_cit->first.find("energy")!=std::string::npos)
    {
      energy_mesh=temp_cit->second;
      std::map<std::string, Simulation*>::const_iterator temp_c_it2=Mesh_Constructors.find(temp_cit->first);
      NEMO_ASSERT(temp_c_it2!=Mesh_Constructors.end(),prefix+"no mesh constructor found for \""+temp_cit->first+"\"\n");
      energy_mesh_constructor = temp_c_it2->second;
      InputOptions& mesh_options=energy_mesh_constructor->get_reference_to_options();
      if(!mesh_options.get_option(std::string("non_rectangular"),false))
        //if(!options.get_option("non_rectangular_"+temp_cit->first,false))
        energy_mesh_constructor->get_data("e_space_"+temp_cit->first,energy_mesh);
      else
      {
        //create a vector of NemoMeshPoint with the k-point only
        std::vector<double> temp_k_point(3,0.0);
        NemoMeshPoint temp_point(0,temp_k_point);
        std::vector<NemoMeshPoint> temp_vector(1,temp_point);
        energy_mesh_constructor->get_data("e_space_"+temp_cit->first,temp_vector,energy_mesh);
      }

    }
  NEMO_ASSERT(energy_mesh!=NULL,prefix+"have not found the energy mesh\n");
  NEMO_ASSERT(energy_mesh_constructor!=NULL,prefix+"have not found the energy mesh constructor\n");


  //1.3 get the energy resolved current and the energy integrated current
  std::map<double,double>::const_iterator cit=energy_resolved_transmission.begin();
  for(; cit!=energy_resolved_transmission.end(); cit++)
  {
    double energy = cit->first;
    std::map<double,double>::const_iterator trans_cit=energy_resolved_transmission.find(energy);
    NEMO_ASSERT(trans_cit!=energy_resolved_transmission.end(),prefix+"have not found energy in transmission map\n");

    double integration_weight = 1.0;
    //get_fermi_factor
    double fermi_factor_1 = 0.0;//get_fermi_factor(chemical_potential1, energy, temperature1_in_eV, use_analytical_momenta);
    double fermi_factor_2 = 0.0;//get_fermi_factor(chemical_potential2, energy, temperature2_in_eV, use_analytical_momenta);

    if(particle_type_is_Fermion)
    {
      fermi_factor_1 = get_fermi_factor(chemical_potential1, energy, temperature1_in_eV, use_analytical_momenta);
      fermi_factor_2 = get_fermi_factor(chemical_potential2, energy, temperature2_in_eV, use_analytical_momenta);

    }
    else
    {
      fermi_factor_1 = NemoMath::bose_distribution(chemical_potential1,temperature1_in_eV,energy);
      fermi_factor_2 = NemoMath::bose_distribution(chemical_potential2,temperature2_in_eV,energy);
    }
    if(particle_type_is_Fermion)
      energy_resolved_current[cit->first]= trans_cit->second*NemoPhys::elementary_charge/NemoPhys::h*(fermi_factor_1-fermi_factor_2);
    else
      energy_resolved_current[cit->first]= trans_cit->second*NemoPhys::elementary_charge*energy/NemoPhys::h*(fermi_factor_1-fermi_factor_2);


    double temp_double=1.0;
    // InputOptions& mesh_options=energy_mesh_constructor->get_reference_to_options();
    std::vector<double> temp_vector(1,energy);
    NemoMeshPoint temp_momentum(0,temp_vector);
    energy_mesh_constructor->get_data("integration_weight",temp_momentum,temp_double);

    integration_weight*=temp_double;

    //1.3.2 perform the integral sum
    current+=energy_resolved_current[cit->first]*NemoPhys::elementary_charge*integration_weight; //[A]
  }


  //Suppressed by default - ruins scaling - EMW
  //Option isn't currently read correctly from input deck
  //Think option isn't being read correctly from RGF module
  bool no_current_output=options.get_option("no_current_output",false);
  if(output_done_here(std::string("end"))&&options.get_option("energy_resolved_current_output",true) && !no_current_output && !no_file_output)
  {
    //3. output current to file
    std::stringstream stm;
    stm<<chemical_potential1-chemical_potential2;
    std::string potential_diff = stm.str();
    std::string filename;
    std::ofstream out_file;
    filename=get_name()+"_"+potential_diff+"_energy_resolved_current" + get_output_suffix() + ".dat";
    out_file.open(filename.c_str());
    std::map<double,double>::const_iterator cit_ej=energy_resolved_current.begin();
    for(; cit_ej!=energy_resolved_current.end(); cit_ej++)
    {
      out_file<<cit_ej->first<<"\t"<<cit_ej->second* NemoPhys::elementary_charge<<"\n";
    }
    out_file.close();
    filename=get_name()+"_"+potential_diff+"_totalcurrent" + get_output_suffix() + ".dat";
    out_file.open(filename.c_str());
    out_file<<"total current is: "<<current<<"\n";
    out_file.close();
  }

  NemoUtils::toc(tic_toc_prefix);

}

double Propagation::get_fermi_factor(double chemical_potential, double energy, double temperature,  bool analytical_momenta)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_fermi_factor ");
  std::string prefix="Propagation(\""+this->get_name()+"\")::get_fermi_factor ";

  NemoUtils::tic(tic_toc_prefix);

  double fermi_factor = 0.0;
  if (analytical_momenta==true)
  {
    NemoUtils::MsgLevel prev_level = msg.get_level();
    NemoUtils::msg.set_level(NemoUtils::MsgLevel(3));
    msg <<"Propagation current integration performing analytical momentum\n";
    //5.1.2.1.1 find the type of analytical momentum, i.e. 1D or 2D
    std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
    NEMO_ASSERT(momentum_type=="1D"||
                momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n"); // check prefix equivalent or define
    msg <<"analytical momentum type is " << momentum_type <<"\n";

    NemoUtils::msg.set_level(prev_level);
    //analytical_momenta_mh*NemoPhys::electron_mass
    if(options.check_option("analytical_momenta_me"))
    {
      double analytical_momenta_me = options.get_option("analytical_momenta_me",1.0);
      fermi_factor = get_analytically_integrated_distribution(energy, temperature, chemical_potential, momentum_type, true, false, false,
                     analytical_momenta_me*NemoPhys::electron_mass);
    }
    else
      fermi_factor = get_analytically_integrated_distribution(energy, temperature, chemical_potential, momentum_type, true, false, false);
  }
  else
    fermi_factor = NemoMath::fermi_distribution(chemical_potential,temperature,energy);

  NemoUtils::toc(tic_toc_prefix);
  return fermi_factor;

}


void Propagation::calculate_nonrectangular_mesh_current()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_nonrectangular_mesh_current ");
  std::string prefix="Propagation(\""+this->get_name()+"\")::calculate_nonrectangular_mesh_current ";

  NemoUtils::tic(tic_toc_prefix);

  //double temperature1_eV = temperature1*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  //double temperature2_eV = temperature2*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  //calculate the current density at each energy point
  current=0.0;
  std::map<double,double> energy_resolved_current;

  vector<std::string> Hamilton_momenta;
  NEMO_ASSERT(options.check_option("Hamilton_momenta"),prefix+"Hamilton_momenta is not defined\n");
  options.get_option("Hamilton_momenta",Hamilton_momenta);
  bool valley_only = 0;
  if(Hamilton_momenta.size()==1&&Hamilton_momenta[0].find("valley")!=std::string::npos)
    valley_only = !valley_only;

  NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"received empty all_momenta\n");
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
  std::set<double> temp_all_energies;

  std::set<std::vector<double> > temp_all_kvectors; //Bozidar (k-resolved current)
  std::map<std::vector<double>, int> translation_map_kvector_index;
  //std::vector<double> k_resolved_current_vector;
  std::vector<std::vector<double> > all_kvectors;
  // int k_index;

  //std::map<std::vector<NemoMeshPoint>,double> map_momentum_energy;
  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    std::complex<double> temp_complex_energy = PropagationUtilities::read_complex_energy_from_momentum(this,*c_it, writeable_Propagator);
    NEMO_ASSERT(temp_complex_energy.imag() == 0.0, prefix + "nonrectangular_mesh_current is solving on energy with non-zero imaginary part\n");
    double temp_energy = temp_complex_energy.real();
    temp_all_energies.insert(temp_energy);
    if(!valley_only)
    {
      //Bozidar (k-resolved current)
      std::vector<double> temp_kvector = read_kvector_from_momentum(*c_it, writeable_Propagator, NULL);
      temp_all_kvectors.insert(temp_kvector);
    }
    //map_momentum_energy[c_it->first]=temp_energy;
  }
  //1.2 save the ordered energies into a vector and save the index-mapping
  std::vector<double> all_energies(temp_all_energies.size());
  std::map<double, int> translation_map_energy_index;
  int counter = 0;
  std::set<double>::iterator e_it;
  for(e_it=temp_all_energies.begin(); e_it!=temp_all_energies.end(); e_it++)
  {
    double temp_energy = *e_it;
    translation_map_energy_index[temp_energy]=counter;
    all_energies[counter]=temp_energy;
    counter++;
  }
  if(!valley_only)
  {
    //Bozidar (k-resolved current)
    std::vector<double> temptemp((*temp_all_kvectors.begin()).size(),0.0);
    all_kvectors.resize(temp_all_kvectors.size(),temptemp);
    counter = 0;
    std::set<std::vector<double> >::iterator k_it;
    for(k_it=temp_all_kvectors.begin(); k_it!=temp_all_kvectors.end(); k_it++)
    {
      std::vector<double> temp_kvector = *k_it;
      translation_map_kvector_index[temp_kvector]=counter;
      all_kvectors[counter]=temp_kvector;
      counter++;
    }
  }

  //1.3 fill a vector for energy resolved transmission with the local results or 0 otherwise
  std::vector<double> energy_resolved_transmission2(translation_map_energy_index.size(),0.0);


  //Bozidar (k-resolved current)
  std::vector<double> k_resolved_current_vector(translation_map_kvector_index.size(),0.0);


  for(c_it=pointer_to_all_momenta->begin(); c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    const Propagator* temp_Propagator = writeable_Propagator;
    std::complex<double> temp_complex_energy = PropagationUtilities::read_complex_energy_from_momentum(this, *c_it, temp_Propagator);
    NEMO_ASSERT(temp_complex_energy.imag() == 0.0, prefix + "nonrectangular_mesh_current is solving on energy with non-zero imaginary part\n");
    double temp_energy = temp_complex_energy.real();
    int index = translation_map_energy_index.find(temp_energy)->second;
    int k_index = 0;
    if(!valley_only)
    {
      //Bozidar (k-resolved current)
      std::vector<double> temp_kvector = read_kvector_from_momentum(*c_it, temp_Propagator,NULL);
      k_index = translation_map_kvector_index.find(temp_kvector)->second;
    }
    //1.4 find the integration weight for this specific momentum
    double integration_weight=1.0;
    double integration_weight_without_dk=1.0;
    {
      std::string momentum_name=std::string("");
      for(unsigned int i=0; i<temp_Propagator->momentum_mesh_names.size(); i++)
      {
        //find the momentum mesh constructor and ask it for the weight
        momentum_name=temp_Propagator->momentum_mesh_names[i];
        std::map<std::string,Simulation*>::const_iterator c_it2 = Propagation::Mesh_Constructors.find(momentum_name);
        NEMO_ASSERT(c_it2!=Propagation::Mesh_Constructors.end(),prefix+"have not found constructor for momentum mesh \""+momentum_name+"\"\n");
        //find the momentum name that does not contain "energy"
        if(temp_Propagator->momentum_mesh_names[i].find("energy")==std::string::npos)
        {
          momentum_name=temp_Propagator->momentum_mesh_names[i];
          double temp = find_integration_weight(momentum_name, (*c_it), temp_Propagator);
          integration_weight*=temp;

          if(momentum_name.find("momentum_1D")!=std::string::npos)
          {
            integration_weight/=2.0*NemoMath::pi;
            integration_weight_without_dk/=2.0*NemoMath::pi;
          }
          else if(momentum_name.find("momentum_2D")!=std::string::npos)
          {
            integration_weight/=4.0*NemoMath::pi*NemoMath::pi;
            integration_weight_without_dk/=4.0*NemoMath::pi*NemoMath::pi;
          }
          else if(momentum_name.find("momentum_3D")!=std::string::npos)
          {
            integration_weight/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
            integration_weight_without_dk/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
          }

        }
        else
        {
          double temp= find_integration_weight(momentum_name, (*c_it), temp_Propagator);
          integration_weight*=temp;
          integration_weight_without_dk*=temp;
        }

      }
    }
    //note that this is the momentum integrated current, multiplied with the energy integration weight!
    std::map<std::vector<NemoMeshPoint>,double>::const_iterator transmission_cit=momentum_transmission.find(*c_it);
    if(transmission_cit!=momentum_transmission.end())
    {
      energy_resolved_transmission2[index]+=transmission_cit->second*integration_weight;
      if(!valley_only)
      {
        //Bozidar (k-resolved current without k integration weights)
        //Aryan: do not change k resolved current if performing analytical integration (current is the correct current for the only k defined by user (e.g. k=0)
        k_resolved_current_vector[k_index]+=transmission_cit->second*integration_weight_without_dk*NemoPhys::elementary_charge/NemoPhys::h*
                                            NemoMath::fermi_distribution(chemical_potential1,temperature1_in_eV,temp_energy);
        k_resolved_current_vector[k_index]-=transmission_cit->second*integration_weight_without_dk*NemoPhys::elementary_charge/NemoPhys::h*
                                            NemoMath::fermi_distribution(chemical_potential2,temperature2_in_eV,temp_energy);
        //1.3.2 perform the integral sum
        //current_k+=k_resolved_current_vector[index]*NemoPhys::elementary_charge; //[A]
      }
    }
  }

  //2. get the energy resolved current and the energy integrated current
  //std::map<double,double>::const_iterator cit=energy_resolved_transmission2.begin();
  std::vector<double> energy_resolved_current_vector(energy_resolved_transmission2.size(),0.0);

  //Analytical integration add by Aryan 2014/09/01
  //bool analytical_momenta = false;
  std::string momentum_type=std::string(""); //container for analytical integration
  double effective_mass=NemoPhys::me0_nemo;//* me0/q

  std::string filename;
  filename=get_name()+"_fermi_factor_analytical" + get_output_suffix() + ".dat";
  std::ofstream out_file;
  if(use_analytical_momenta)//options.check_option("analytical_momenta"))
  {
    //analytical_momenta =true;
    if(options.check_option("analytical_momenta_me")) //analytical_momenta_me_CB_S
    {
      effective_mass*=options.get_option("analytical_momenta_me",1.0);//* me0/q
      msg << "effective mass read from input deck with value " << (effective_mass/NemoPhys::me0_nemo) << "\n";
      //effective_mass= 0.05*NemoPhys::me0_nemo;
    }

    NemoUtils::MsgLevel prev_level = msg.get_level();
    NemoUtils::msg.set_level(NemoUtils::MsgLevel(3));
    msg <<"Propagation current integration performing analytical momentum method 2\n";
    //5.1.2.1.1 find the type of analytical momentum, i.e. 1D or 2D
    momentum_type=options.get_option("analytical_momenta",std::string(""));
    NEMO_ASSERT(momentum_type=="1D"||
                momentum_type=="2D",prefix+"called with unknown analytical_momenta \""+momentum_type+"\"\n"); // check prefix equivalent or define
    msg <<"analytical momentum type is " << momentum_type <<"\n";
    msg <<"effective mass, effective mass/me0 are " << effective_mass << " " << (effective_mass/NemoPhys::me0_nemo) <<"\n";
    NemoUtils::msg.set_level(prev_level);

    //add by Aryan

    //std::string filename = output_collector.get_file_path(filename,"checking fermi factor analytical integration" , NemoFileSystem::DEBUG);
    if(!no_file_output)
    {
      out_file.open(filename.c_str());
      out_file << "temp_energy chemical_potential1 fermi_facor_1"<<"\n";
    }
  }
  for(unsigned int i=0; i<energy_resolved_transmission2.size(); i++)
  {
    double temp_energy=all_energies[i];

    // Analytical integration added by Aryan 2014/09/01
    double fermi_factor_1 = 0.0;
    double fermi_factor_2 = 0.0;

    //if(options.check_option("analytical_momenta"))
    if (use_analytical_momenta)
    {
      // a factor 1/q seems to be missing for current ( maybe come from eV unit and me0 =me0/q ?
      // or come from integration weight ...
      fermi_factor_1 = get_analytically_integrated_distribution(temp_energy, temperature1_in_eV, chemical_potential1, momentum_type, true, false,false,
                       effective_mass);
      fermi_factor_2 = get_analytically_integrated_distribution(temp_energy, temperature2_in_eV, chemical_potential2, momentum_type, true, false,false,
                       effective_mass);
      //fermi_factor_1 = NemoMath::fermi_distribution(chemical_potential1,temperature1,temp_energy);
      //fermi_factor_2 = NemoMath::fermi_distribution(chemical_potential2,temperature2,temp_energy);

      if(!no_file_output)
        out_file << temp_energy <<"\t "<<chemical_potential1<<"\t "<<fermi_factor_1<<"\n";

      NemoUtils::MsgLevel prev_level = msg.get_level();
      NemoUtils::msg.set_level(NemoUtils::MsgLevel(3));
      msg <<"analytical momentum type is " << momentum_type <<"\n";
      msg<<"energy fermi_factor_1  fermi_factor_2 " << temp_energy << " " << fermi_factor_1 <<"  " << fermi_factor_2 << "\n";
      NemoUtils::msg.set_level(prev_level);
    }
    else
    {
      fermi_factor_1 = NemoMath::fermi_distribution(chemical_potential1,temperature1_in_eV,temp_energy);
      fermi_factor_2 = NemoMath::fermi_distribution(chemical_potential2,temperature2_in_eV,temp_energy);
    }

    double temp_energy_resolved_current=energy_resolved_transmission2[i]*NemoPhys::elementary_charge/NemoPhys::h*(fermi_factor_1-fermi_factor_2);

    energy_resolved_current[temp_energy]=temp_energy_resolved_current;
    energy_resolved_current_vector[i]=temp_energy_resolved_current;
    ////1.3.2 perform the integral sum
    current+=temp_energy_resolved_current*NemoPhys::elementary_charge; //[A]

  }

  //add by Aryan
  if(!no_file_output)
    out_file.close();

  //2. MPI_Allreduce(SUM) the resulting transmission per energy
  if(debug_output&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(get_simulation_communicator(), &myrank);

    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::ofstream out_file;
    std::string filename=get_name()+convert_to_string.str() +"_energy_resolved_current.dat";
    //output of the energy resolved transmission per MPI process - for test purposes
    out_file.open(filename.c_str());
    for(unsigned int i=0; i<energy_resolved_current_vector.size(); i++)
    {
      out_file<<all_energies[i]<<"\t"<<energy_resolved_current_vector[i]<<"\n";
    }
    out_file.close();
  }
  //2.2 perform the sum, i.e. MPI_(All)reduce
  NemoMesh* energy_mesh=NULL;
  energy_mesh=Mesh_tree_topdown.begin()->first;
  MPI_Barrier(energy_mesh->get_global_comm());

  NemoUtils::tic(tic_toc_prefix+"communication of doubles 2");
  if(solve_on_single_replica)
  {
    NemoUtils::tic(tic_toc_prefix+"communication of doubles 2 Reduce");
    std::vector<double> temp2(energy_resolved_current_vector);
    MPI_Reduce(&(temp2[0]),&(energy_resolved_current_vector[0]),energy_resolved_current_vector.size(),MPI_DOUBLE, MPI_SUM, 0,
               energy_mesh->get_global_comm());
    if(!valley_only)
    {
      //Bozidar (k-resolved current)
      std::vector<double> temp2_k(k_resolved_current_vector);
      MPI_Reduce(&(temp2_k[0]),&(k_resolved_current_vector[0]),k_resolved_current_vector.size(),MPI_DOUBLE, MPI_SUM, 0,
                 energy_mesh->get_global_comm());
    }
    NemoUtils::toc(tic_toc_prefix+"communication of doubles 2 Reduce");
    //2.3 if the user wants to have the energy resolved current output...
    if(output_done_here(std::string("end"))&&options.get_option("energy_resolved_current_output",true)&&!no_file_output)
    {
      NemoUtils::tic(tic_toc_prefix+"communication of doubles 2 energy resolved current output");
      std::stringstream stm;
      stm<<chemical_potential1-chemical_potential2;
      std::string potential_diff = stm.str();
      std::string filename;
      filename=get_name()+"_"+potential_diff+"_energy_resolved_current" + get_output_suffix() + ".dat";
      std::ofstream out_file;
      out_file.open(filename.c_str());
      for(unsigned int i=0; i<energy_resolved_current_vector.size(); i++)
      {
        out_file<<all_energies[i]<<"\t"<<energy_resolved_current_vector[i]*NemoPhys::elementary_charge<<"\n";
      }
      out_file.close();

      if(!valley_only)
      {
        //Bozidar (k-resolved current)
        filename=get_name()+"_"+potential_diff+"_k_resolved_current" + get_output_suffix() + ".dat";
        out_file.open(filename.c_str());
        //all_kvectors = *pointer_to_all_kvectors;
        for(unsigned int i=0; i<k_resolved_current_vector.size(); i++)
        {
          for(unsigned int j=0; j<all_kvectors[i].size(); j++)
          {
            out_file<<all_kvectors[i][j]<<"\t";
          }
          out_file<<k_resolved_current_vector[i]*NemoPhys::elementary_charge<<"\n";;
        }
        out_file.close();
      }

      NemoUtils::toc(tic_toc_prefix+"communication of doubles 2 energy resolved current output");
    }
  }
  else
  {
    NemoUtils::tic(tic_toc_prefix+"communication of doubles 2 MPIBarrier 1");
    MPI_Barrier(energy_mesh->get_global_comm());
    NemoUtils::toc(tic_toc_prefix+"communication of doubles 2 MPIBarrier 1");
    NemoUtils::tic(tic_toc_prefix+"communication of doubles 2 MPIAllreduce 1");
    MPI_Allreduce(MPI_IN_PLACE,&(energy_resolved_current_vector[0]),energy_resolved_current_vector.size(),MPI_DOUBLE, MPI_SUM,
                  energy_mesh->get_global_comm());
    NemoUtils::toc(tic_toc_prefix+"communication of doubles 2 MPIAllreduce 1");
  }
  NemoUtils::tic(tic_toc_prefix+"communication of doubles 2 MPIBarrier 2");
  MPI_Barrier(energy_mesh->get_global_comm());
  NemoUtils::toc(tic_toc_prefix+"communication of doubles 2 MPIBarrier 2");
  NemoUtils::tic(tic_toc_prefix+"communication of doubles 2 MPIAllreduce 2");
  MPI_Allreduce(MPI_IN_PLACE,&current,1,MPI_DOUBLE,MPI_SUM,energy_mesh->get_global_comm());
  NemoUtils::toc(tic_toc_prefix+"communication of doubles 2 MPIAllreduce 2");
  NemoUtils::toc(tic_toc_prefix+"communication of doubles 2");

  //3. store the result on file by one MPI process, with the appropriate voltage in the filename
  //bool no_file_output = options.get_option("no_file_output",false);
  //if(!no_file_output)
  if(output_done_here(std::string("end"))&&options.get_option("energy_resolved_current_output",true)&&!no_file_output)
  {
    //int myenergyrank;
    //MPI_Comm_rank(energy_mesh->get_global_comm(),&myenergyrank);
    //if(output_done_here(std::string("end"))&&myenergyrank==0)
    //{
    //3. output current to file
    std::stringstream stm;
    stm<<chemical_potential1-chemical_potential2;
    std::string potential_diff = stm.str();
    std::string filename;
    std::ofstream out_file;
    filename=get_name()+"_"+potential_diff+"_energycurrent.dat";
    out_file.open(filename.c_str());
    std::map<double,double>::const_iterator cit_ej=energy_resolved_current.begin();
    for(; cit_ej!=energy_resolved_current.end(); cit_ej++)
    {
      out_file<<cit_ej->first<<"\t"<<cit_ej->second<<"\n";
    }
    out_file.close();
    filename=get_name()+"_"+potential_diff+"_totalcurrent.dat";
    out_file.open(filename.c_str());
    out_file<<"total current is: "<<current<<"\n";
    out_file.close();
    //}
  }

  NemoUtils::toc(tic_toc_prefix);


}


/*
void Propagation::calculate_current()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_current ");
  NemoUtils::tic(tic_toc_prefix);

  //0. find all existing energies
  NEMO_ASSERT(pointer_to_all_momenta->size()>0,prefix+"received empty all_momenta\n");
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
  std::set<double> temp_all_energies;
  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    double temp_energy = read_energy_from_momentum(*c_it, Propagators.begin()->second);
    temp_all_energies.insert(temp_energy);
  }
  std::vector<double> all_energies(temp_all_energies.size(),0.0);
  int counter=0;
  for(std::set<double>::iterator it=temp_all_energies.begin(); it!=temp_all_energies.end(); ++it)
  {
    all_energies[counter]=*it;
    counter++;
  }

  //calculate the current density at each energy point
  current=0.0;
  std::vector<double> energy_resolved_current(all_energies.size(),0.0);

  //1. Landau formula J(E)=T(E)*(f1(E)-f2(E))*(q/h)
  //1.1 check if transmission is calculated, if not, calculate transmission
  std::vector<double> transmission;
  std::vector<double>* transmission_energy = &transmission;
  calculate_transmission_energy(transmission_energy);
  NEMO_ASSERT(transmission_energy->size()==all_energies.size(),
      "Propagation(\""+get_name()+"\")::calculate_current number of energy points and number of transmission points not match\n");

  //1.2 obtain the Fermi distribution at all contacts
  //read in the temperature and the chemical potential for all contacts -- assume there are only two contacts at this moment
  double temperature1=options.get_option("temperature1", NemoPhys::temperature);
  temperature1*=NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
  double chemical_potential1=options.get_option("chemical_potential1", 0.0);
  double temperature2=options.get_option("temperature2", NemoPhys::temperature);
  temperature2*=NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
  double chemical_potential2=options.get_option("chemical_potential2", 0.0);

  //1. loop through all momenta
  //same order as transmission_energy

  std::vector<double> energy_resolved_current(all_energies.size(),0.0);

  //for output without communication
  for(unsigned int i=0; i<all_energies.size(); i++)
  {
    energy_resolved_current[i] = (*transmission_energy)[i]*NemoPhys::elementary_charge/NemoPhys::h*(NemoMath::fermi_distribution(chemical_potential1,
            temperature1_in_eV,all_energies[i])-NemoMath::fermi_distribution(chemical_potential2,temperature2_in_eV,all_energies[i]));  //[A/J]
  }

  //do integration locally and MPI reduce
  Propagator *Propagator_pointer = Propagators.begin();
  Propagator::PropagatorMap::const_iterator momentum_c_it = Propagator_pointer->propagator_map.begin();
  //avoid access in the loop
  Propagator::PropagatorMap::const_iterator momentum_c_it_end = Propagator_pointer->propagator_map.end();
  if(; momentum_c_it!=momentum_c_it_end; ++momentum_c_it)
  {

  }


  NemoUtils::toc(tic_toc_prefix);

}
*/
void Propagation::calculate_energy_current(void)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_energy_current ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::calculate_energy_current: ";

  //This method is very similar to calculate_energy_current. Differences are:
  //a) We apply the Landauer formula with the energy as an extra weighting factor
  //b) We check whether this current is meant for Phonons (Bose distribution) or for Electrons (Fermi distribution)
  //NOTE for Kai: check whether Propagators.begin() is a Fermion or a Boson type, using get_Propagator_type(Propagators.begin()->first)

  NemoUtils::toc(tic_toc_prefix);
}


const DOFmapInterface& Propagation::get_special_const_dof_map(const std::string& input_name, const Domain* input_domain) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_special_const_dof_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::get_special_const_dof_map: ";
  //1. find the source of matrix
  std::string variable_name ="DOFmap";
  //1. get the name of the simulation that provides variable_name
  std::string temp_variable_name= input_name + variable_name + "_solver";
  std::string simulation_name;
  if(options.check_option(temp_variable_name))
    simulation_name = options.get_option(temp_variable_name,std::string(""));
  else
  {
    simulation_name = Hamilton_Constructor->get_name();
    //throw std::invalid_argument(prefix+"please define \""+temp_variable_name+"\"\n");
  }
  //2. get the pointer to the simulation that provides variable_name
  const Simulation* source_of_data = this->find_const_simulation(simulation_name);
  NEMO_ASSERT(source_of_data!=NULL, prefix+"have not found simulation \""+simulation_name+"\"\n");
  //3. query get_const_dof_map from that simulation
  if(input_domain==NULL)
  {
    const DOFmapInterface& temp_DOFmap=source_of_data->get_const_dof_map(get_const_simulation_domain());
    //4. return the result
    NemoUtils::toc(tic_toc_prefix);
    return temp_DOFmap;
  }
  else
  {
    const DOFmapInterface& temp_DOFmap=source_of_data->get_const_dof_map(input_domain);
    //4. return the result
    NemoUtils::toc(tic_toc_prefix);
    return temp_DOFmap;
  }
}

DOFmapInterface& Propagation::get_special_dof_map(const std::string& input_name, const Domain* input_domain)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_special_dof_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::get_special_dof_map: ";
  //1. find the source of matrix
  std::string variable_name ="DOFmap";
  //1. get the name of the simulation that provides variable_name
  std::string temp_variable_name= input_name + variable_name + "_solver";
  std::string simulation_name;
  if(options.check_option(temp_variable_name))
    simulation_name = options.get_option(temp_variable_name,std::string(""));
  else
  {
    simulation_name = Hamilton_Constructor->get_name();
    //throw std::invalid_argument(prefix+"please define \""+temp_variable_name+"\"\n");
  }
  //2. get the pointer to the simulation that provides variable_name
  Simulation* source_of_data = this->find_simulation(simulation_name);
  NEMO_ASSERT(source_of_data!=NULL, prefix+"have not found simulation \""+simulation_name+"\"\n");
  //3. query get_dof_map from that simulation
  if(input_domain==NULL)
  {
    DOFmapInterface& temp_DOFmap=source_of_data->get_dof_map(get_const_simulation_domain());
    //4. return the result
    NemoUtils::toc(tic_toc_prefix);
    return temp_DOFmap;
  }
  else
  {
    DOFmapInterface& temp_DOFmap=source_of_data->get_dof_map(input_domain);
    //4. return the result
    NemoUtils::toc(tic_toc_prefix);
    return temp_DOFmap;
  }
}

void Propagation::get_data(const std::string& variable, double&  data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data11 ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::get_data ";
  if (variable=="current")
  {
    calculate_current();
    data = current;
  }
  else
    throw runtime_error(prefix+"unknown variable " + variable + "\n");
  NemoUtils::toc(tic_toc_prefix);
}

const DOFmapInterface& Propagation::get_const_dof_map(const Domain* input_domain) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_const_dof_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::get_const_dof_map: ";
  //1. find the source of matrix
  std::string variable_name ="DOFmap";
  //1. get the name of the simulation that provides variable_name
  std::string temp_variable_name= variable_name + "_solver";
  std::string simulation_name;
  if(options.check_option(temp_variable_name))
    simulation_name = options.get_option(temp_variable_name,std::string(""));
  else
    simulation_name = Hamilton_Constructor->get_name();
  //2. get the pointer to the simulation that provides variable_name
  const Simulation* source_of_data = this->find_const_simulation(simulation_name);
  NEMO_ASSERT(source_of_data!=NULL, prefix+"have not found simulation \""+simulation_name+"\"\n");
  //3. query get_const_dof_map from that simulation
  const DOFmapInterface& temp_DOFmap=source_of_data->get_const_dof_map(input_domain);
  //4. return the result
  NemoUtils::toc(tic_toc_prefix);
  return temp_DOFmap;
}

DOFmapInterface& Propagation::get_dof_map(const Domain* input_domain)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_dof_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::get_const_dof_map: ";
  //1. find the source of matrix
  std::string variable_name ="DOFmap";
  //1. get the name of the simulation that provides variable_name
  std::string temp_variable_name= variable_name + "_solver";
  std::string simulation_name;
  if(options.check_option(temp_variable_name))
    simulation_name = options.get_option(temp_variable_name,std::string(""));
  else
    simulation_name = Hamilton_Constructor->get_name();
  //2. get the pointer to the simulation that provides variable_name
  Simulation* source_of_data = find_simulation(simulation_name);
  NEMO_ASSERT(source_of_data!=NULL, prefix+"have not found simulation \""+simulation_name+"\"\n");
  //3. query get_dof_map from that simulation
  DOFmapInterface& temp_DOFmap=source_of_data->get_dof_map(input_domain);
  //4. return the result
  NemoUtils::toc(tic_toc_prefix);
  return temp_DOFmap;
}

void Propagation::do_solve_combine(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                   PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve_combine ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::do_solve_combine: ";
  msg<<prefix<<"of \""<<output_Propagator->get_name()<<"\" called for momentum:";
  for(unsigned int i=0; i<momentum_point.size(); i++)
    momentum_point[i].print();
  //NOTE that the code below is meant for the non-LRA case (with LRA, we require the transformed DOFmap)
  //the domain of this Propagation should be the full device (largest domain)

  if (options.check_option("Large_DOFmap_solver"))
  {
    std::string temp_name = options.get_option("Large_DOFmap_solver", std::string(""));
    Simulation* temp_simulation = find_simulation(temp_name);
    NEMO_ASSERT(temp_simulation != NULL, prefix + "have not found simulation \"" + temp_name + "\"\n");
    //large_DOFmap = temp_simulation->get_dof_map();
    Hamilton_Constructor = temp_simulation;
  }
  const DOFmapInterface& large_DOFmap = Hamilton_Constructor->get_dof_map(get_const_simulation_domain());

  //1. get the DOFmap for the rows
  std::string variable_name = "row_";
  const Domain* row_domain=NULL;
  if(options.check_option("row_domain"))
  {
    std::string row_domain_name=options.get_option("row_domain",std::string(""));
    row_domain=Domain::get_domain(row_domain_name);
  }
  //const DOFmapInterface& row_DOFmap=get_special_const_dof_map(variable_name,row_domain);
  DOFmapInterface& row_DOFmap=get_special_dof_map(variable_name,row_domain);

  //2. get the DOFmap2
  variable_name = "column_";
  const Domain* column_domain=NULL;
  if(options.check_option("column_domain"))
  {
    std::string column_domain_name=options.get_option("column_domain",std::string(""));
    column_domain=Domain::get_domain(column_domain_name);
  }
  //const DOFmapInterface& column_DOFmap=get_special_const_dof_map(variable_name,column_domain);
  DOFmapInterface& column_DOFmap=get_special_dof_map(variable_name,column_domain);

  //3. get the source of the matrix to store NOTE: we do not use a readable Propagator to make the combine-solver always "ready" (i.e. no job_done_momentum_map with false-argument)
  Simulation*  source_of_data=NULL;
  if(source_solver_for_storage!=NULL)
  {
    source_solver_for_storage->get_data("name_of_writable_Propagator", variable_name);
    source_of_data=source_solver_for_storage;
  }
  else
  {
    if(options.check_option(std::string("source_matrix_name")))
      variable_name=options.get_option(std::string("source_matrix_name"),std::string(""));
    else
    {
      variable_name = output_Propagator->get_name(); //substact the string "combined"
      std::string::iterator temp_it=variable_name.begin();
      temp_it+=variable_name.find(std::string("combined_"));
      std::string::iterator temp_it_end=temp_it+std::string("combined_").size();
      variable_name.erase(temp_it,temp_it_end);
    }
    source_of_data=find_source_of_data(variable_name);
  }

  //4. get the actual source matrix
  PetscMatrixParallelComplex* source_matrix=NULL;
  source_of_data->get_data(variable_name,&momentum_point,source_matrix); //,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));


  //6. if not done yet, set the storage matrix (allocate propagator?)
  //6.1 try to find the matrix in the writeable propagator map
  Propagator::PropagatorMap::iterator it=output_Propagator->propagator_map.find(momentum_point);
  if(it!=output_Propagator->propagator_map.end())
    result=it->second;
  std::map<std::string, NemoPhys::Propagator_type>::const_iterator type_cit=Propagator_types.find(output_Propagator->get_name());
  NEMO_ASSERT(type_cit!=Propagator_types.end(),prefix+"have not found the type of \""+output_Propagator->get_name()+"\"\n");
  NemoPhys::Propagator_type output_type=type_cit->second;
  //if(output_Propagator->get_name().find("self")==std::string::npos)
  if(output_type!=NemoPhys::Fermion_retarded_self && output_type!=NemoPhys::Fermion_lesser_self && output_type!=NemoPhys::Fermion_advanced_self
      && output_type!=NemoPhys::Fermion_greater_self && output_type!=NemoPhys::Fermion_spectral_self
      && output_type!=NemoPhys::Boson_retarded_self && output_type!=NemoPhys::Boson_lesser_self && output_type!=NemoPhys::Boson_advanced_self
      && output_type!=NemoPhys::Boson_greater_self && output_type!=NemoPhys::Boson_spectral_self)
  {
    if(result==NULL)
    {
      msg<<prefix<<"creating a matrix container\n";
      //5.1a create a new Petscmatrix
      unsigned int number_of_cols = 0;
      Hamilton_Constructor->get_data("global_size",number_of_cols);
      unsigned int number_of_rows = number_of_cols;
      // result = new PetscMatrixParallelComplex (number_of_rows,number_of_cols,holder.geometry_communicator);
      result = new PetscMatrixParallelComplexContainer(number_of_rows,number_of_cols,
          get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);

      unsigned int number_of_own_super_rows = 0;
      Hamilton_Constructor->get_data("local_size",number_of_own_super_rows);
      result->set_num_owned_rows(number_of_own_super_rows);

      output_Propagator->allocated_momentum_Propagator_map[momentum_point]=true;
    }


    //7. put the source matrix into the storage matrix
    std::vector<int> row_indexes;
    Hamilton_Constructor->get_dof_map(get_const_simulation_domain()).get_local_row_indexes(&row_indexes);
    //translate the row_indexes into a set for later usage
    std::set<double> set_local_super_row_indices;
    for(unsigned int i=0; i<row_indexes.size(); i++)
      set_local_super_row_indices.insert(row_indexes[i]);

    //7.1 get the inverse index maps
    //std::map<unsigned int, unsigned int> rows_superindex_map;
    //std::map<unsigned int, unsigned int>::const_iterator c_it;

    //std::map<unsigned int, unsigned int> cols_superindex_map;

    //large_DOFmap.get_sub_DOF_index_map(&row_DOFmap,rows_superindex_map,true);
    //large_DOFmap.get_sub_DOF_index_map(&column_DOFmap,cols_superindex_map,true);
    //NEMO_ASSERT(rows_superindex_map.size()>0,prefix+"row super index map is empty. No place to store\n");
    //NEMO_ASSERT(cols_superindex_map.size()>0,prefix+"column super index map is empty. No place to store\n");

    std::vector<int> rows; //(rows_superindex_map.size(),0);
    std::vector<int> cols; //(cols_superindex_map.size(),0);

    //c_it = rows_superindex_map.begin();

    //for(; c_it!=rows_superindex_map.end(); c_it++)
    //  rows[c_it->first]=c_it->second;

    Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,row_DOFmap,rows,get_const_simulation_domain());
    Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap,column_DOFmap,cols,get_const_simulation_domain());


    //c_it = cols_superindex_map.begin();

    //for(; c_it!=cols_superindex_map.end(); c_it++)
    //  cols[c_it->first]=c_it->second;

    PetscMatrixParallelComplexContainer* temp_pointer = dynamic_cast<PetscMatrixParallelComplexContainer*>(result);
    NEMO_ASSERT(temp_pointer!=NULL,prefix+"dynamic cast into PetscMatrixParallelComplexContainer failed\n");

    //std::cerr<<prefix<<"saving with rows[0]: "<<rows[0]<<" and cols[0]: "<<cols[0]<<"\n";
    dynamic_cast<PetscMatrixParallelComplexContainer*>(result)->set_block_from_matrix1(*source_matrix, rows,cols);

  }
  else
  {
    add_combine_solver_matrices = options.get_option("add_combine_solver_matrices",bool(false));
    if(!add_combine_solver_matrices)
    {
      delete result;
      unsigned int number_of_cols = 0;
      Hamilton_Constructor->get_data("global_size",number_of_cols);
      unsigned int number_of_rows = number_of_cols;
      result = new PetscMatrixParallelComplex(number_of_rows,number_of_cols,
          get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/); //(*source_matrix);
    }
    unsigned int number_of_own_super_rows = 0;
    Hamilton_Constructor->get_data("local_size",number_of_own_super_rows);
    result->set_num_owned_rows(number_of_own_super_rows);

    bool use_source_sparsity_pattern = options.get_option("use_source_sparsity_pattern",false);

    PropagationOptionsInterface* opts_interface = dynamic_cast<PropagationOptionsInterface*>(this);
    NEMO_ASSERT(opts_interface, prefix + ": simulation " + get_name() + " cannot be cast to type PropagationOptionsInterface");
    if (opts_interface->get_compute_blockdiagonal_self_energy())
      use_source_sparsity_pattern = true;

    if(!use_source_sparsity_pattern) //do filtering 
    {
      std::vector<int> local_rows;
      Hamilton_Constructor->get_dof_map(get_const_simulation_domain()).get_local_row_indexes(&local_rows);

      //determine how many off diagonals to store
      //and determine whether these offdiagonals are given as DOFs or as atoms
      int number_of_off_diagonals = options.get_option("store_offdiagonals",0);
      bool atomic_resolved_off_diagonals = false;
      int number_of_orbitals = 1;
      if(options.check_option("store_atomic_offdiagonals"))
      {
        atomic_resolved_off_diagonals = true;
        number_of_off_diagonals = options.get_option("store_atomic_offdiagonals",0);
        //find the number of orbitals per atom
        double temp_number_of_orbitals;
        Hamilton_Constructor->get_data("number_of_orbitals",temp_number_of_orbitals);
        number_of_orbitals=int(temp_number_of_orbitals);
      }
      int temp_locals=std::min(2*number_of_off_diagonals+1,int(number_of_own_super_rows));//number_of_off_diagonals+1;
      //whatever is left
      int temp_nonlocals=std::min(number_of_off_diagonals,int(number_of_own_super_rows-temp_locals));



      int start_row_index=0;
      int end_row_index=0;
      source_matrix->get_ownership_range(start_row_index,end_row_index);

      std::vector<int> local_nonzeros = std::vector<int> (number_of_own_super_rows,temp_locals);
      std::vector<int> nonlocal_nonzeros = std::vector<int> (number_of_own_super_rows,temp_nonlocals);

      for (unsigned int i=0; i<number_of_own_super_rows; i++)
        result->set_num_nonzeros(local_rows[i],local_nonzeros[i],nonlocal_nonzeros[i]);

      if(!add_combine_solver_matrices)
      {
        result->allocate_memory();

        for(unsigned int i=0; i<number_of_own_super_rows; i++)
          for(unsigned int j=0; j<number_of_own_super_rows; j++)
            if(!atomic_resolved_off_diagonals && std::abs(local_rows[i]-local_rows[j])<=number_of_off_diagonals)
              result->set(local_rows[i],local_rows[j],source_matrix->get(local_rows[i],local_rows[j]));
            else if(atomic_resolved_off_diagonals && std::abs(local_rows[i]-local_rows[j])<=number_of_off_diagonals*number_of_orbitals &&
                std::abs(local_rows[i]-local_rows[j])%number_of_orbitals==0)
              result->set(local_rows[i],local_rows[j],source_matrix->get(local_rows[i],local_rows[j]));
      }
      else
      {
        for(unsigned int i=0; i<number_of_own_super_rows; i++)
          for(unsigned int j=0; j<number_of_own_super_rows; j++)
            if(!atomic_resolved_off_diagonals && std::abs(local_rows[i]-local_rows[j])<=number_of_off_diagonals)
              result->add(local_rows[i],local_rows[j],source_matrix->get(local_rows[i],local_rows[j]));
            else if(atomic_resolved_off_diagonals && std::abs(local_rows[i]-local_rows[j])<=number_of_off_diagonals*number_of_orbitals &&
                std::abs(local_rows[i]-local_rows[j])%number_of_orbitals==0)
              result->add(local_rows[i],local_rows[j],source_matrix->get(local_rows[i],local_rows[j]));
      }

    }
    else //get sparsity pattern from source
    {
      //add_combine_solver_matrices  
      std::set<std::pair<int,int> > set_of_row_col_indices;        
      const std::set<std::pair<int,int> >* pointer_to_set_of_row_col_indices=&set_of_row_col_indices;
      source_of_data->get_data(std::string("sparsity_pattern"),pointer_to_set_of_row_col_indices);
      //if(pointer_to_set_of_row_col_indices==NULL)
      //  pointer_to_set_of_row_col_indices=&set_of_row_col_indices;
      NEMO_ASSERT(!pointer_to_set_of_row_col_indices->empty(), prefix + "did not get sparsity pattern from source");
      std::set<std::pair<int,int> >::const_iterator set_cit=pointer_to_set_of_row_col_indices->begin();
      std::map<int, int> nonzero_map; //key is the row index, value is the count 
      std::map<int, int>::iterator nonzero_it = nonzero_map.begin();
      for(; set_cit!=pointer_to_set_of_row_col_indices->end(); set_cit++)
      {
        int row_index = (*set_cit).first;
        //int col_index = (*set_cit).second;
        nonzero_it = nonzero_map.find(row_index);
        if(nonzero_it==nonzero_map.end())
          nonzero_map[row_index] = 1; //first count
        else
          nonzero_it->second += 1; //increment count 
      }
     vector<int> rows(source_matrix->get_num_owned_rows());
     vector<int> cols(source_matrix->get_num_cols());
     
     for(unsigned int i = 0; i < rows.size(); ++i)
       rows[i] = i; 
     for(unsigned int i = 0; i < cols.size(); ++i)
       cols[i] = i; 
     
     int start_row_index=0;
     int end_row_index=0;
     source_matrix->get_ownership_range(start_row_index,end_row_index);
     result->set_num_owned_rows(source_matrix->get_num_owned_rows());
     for (int i = 0; i < start_row_index; i++)
       result->set_num_nonzeros(i,0,0);
     for (int i = start_row_index; i < end_row_index; i++)
     {
       nonzero_it = nonzero_map.find(i);
       double num_nonzeros = 0; 
       if(nonzero_it != nonzero_map.end())
         num_nonzeros = nonzero_it->second;
       result->set_num_nonzeros(i,num_nonzeros,0);
     }
     for (unsigned int i = end_row_index; i < result->get_num_rows(); i++)
       result->set_num_nonzeros(i,0,0);
     
     //set source matrix into result 
     // const std::vector<int>& rows, const std::vector<int>& cols,InsertMode mode)
     if (!add_combine_solver_matrices)
       result->allocate_memory();

     cplx temp_val(0.0,0.0);
     const std::complex<double>* pointer_to_data= NULL;
     vector<cplx> data_vector;
     vector<int> col_index;
     int n_nonzeros=0;
     const int* n_col_nums=NULL;
     for(int i=(start_row_index); i<(end_row_index); i++)
     {
       source_matrix->get_row(i-start_row_index,&n_nonzeros,n_col_nums,pointer_to_data);
       col_index.resize(n_nonzeros,0);
       data_vector.resize(n_nonzeros,cplx(0.0,0.0));
       for(int j=0; j<n_nonzeros; j++)
       {
         col_index[j]=n_col_nums[j]+start_row_index;
         temp_val=pointer_to_data[j];
         data_vector[j]=temp_val;
       }
       if (n_nonzeros > 0)
       {
         if(!add_combine_solver_matrices)
           result->set(i,col_index,data_vector);
         else
           result->add(i,col_index,data_vector);
       }
       source_matrix->store_row(i-start_row_index,&n_nonzeros,n_col_nums,pointer_to_data);
     }
     //result->set_block_from_matrix(*source_matrix,rows,cols,mode);
     result->assemble();
     //result->save_to_matlab_file(get_name()+"container.m");

    }
    result->assemble();
    //result->save_to_matlab_file(get_name()+"container.m");
    output_Propagator->allocated_momentum_Propagator_map[momentum_point]=true;
  }
  //result->assemble();
  //result->save_to_matlab_file(get_name()+"container.m");
  set_job_done_momentum_map(&(output_Propagator->get_name()), &momentum_point, true);
  NemoUtils::toc(tic_toc_prefix);

}

void Propagation::fill_Propagator_type_map(const std::string& input)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::fill_Propagator_type_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation(\""+this->get_name()+"\")::fill_Propagator_type_map: ";
  // read in what Propagator type these objects are
  if (input.find("inverse_")!=std::string::npos)
    Propagator_types[input]=NemoPhys::Inverse_Green;
  else if (input.find("Green")!=std::string::npos)
  {
    if (input.find("lesser")!=std::string::npos)
    {
      if(input.find("Fermion")!=std::string::npos)
      {
        if(input.find("product")==std::string::npos)
          Propagator_types[input]=NemoPhys::Fermion_lesser_Green;
        else
          Propagator_types[input]=NemoPhys::Fermion_lesser_HGreen;
      }
      else if(input.find("Boson")!=std::string::npos)
      {
        if(input.find("product")==std::string::npos)
          Propagator_types[input]=NemoPhys::Boson_lesser_Green;
        else
          Propagator_types[input]=NemoPhys::Boson_lesser_HGreen;
      }
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else if (input.find("greater")!=std::string::npos)
    {
      if(input.find("Fermion")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Fermion_greater_Green;
      else if(input.find("Boson")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Boson_greater_Green;
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else if (input.find("retarded")!=std::string::npos)
    {
      if(input.find("Fermion")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Fermion_retarded_Green;
      else if(input.find("Boson")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Boson_retarded_Green;
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else if (input.find("advanced")!=std::string::npos)
    {
      if (input.find("Fermion")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Fermion_advanced_Green;
      else if(input.find("Boson")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Boson_advanced_Green;
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else if (input.find("spectral")!=std::string::npos)
    {
      if (input.find("one_side") != std::string::npos)
      {
        if (input.find("Fermion") != std::string::npos)
          Propagator_types[input] = NemoPhys::Fermion_one_side_spectral_Green;
        else if (input.find("Boson") != std::string::npos)
          Propagator_types[input] = NemoPhys::Boson_one_side_spectral_Green;
        else
          throw std::invalid_argument(prefix + "unknown NEGF-object name: \"" + input + "\"\n");
      }
      else 
      {
        if (input.find("Fermion") != std::string::npos)
          Propagator_types[input] = NemoPhys::Fermion_spectral_Green;
        else if (input.find("Boson") != std::string::npos)
          Propagator_types[input] = NemoPhys::Boson_spectral_Green;
        else
          throw std::invalid_argument(prefix + "unknown NEGF-object name: \"" + input + "\"\n");
      }
    }
    else
      throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
  }
  else if (input.find("self")!=std::string::npos)
  {
    if (input.find("retarded")!=std::string::npos)
    {
      if (input.find("Fermion")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Fermion_retarded_self;
      else if (input.find("Boson")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Boson_retarded_self;
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else if (input.find("advanced")!=std::string::npos)
    {
      if (input.find("Fermion")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Fermion_advanced_self;
      else if(input.find("Boson")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Boson_advanced_self;
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else if (input.find("lesser")!=std::string::npos)
    {
      if (input.find("Fermion")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Fermion_lesser_self;
      else if (input.find("Boson")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Boson_lesser_self;
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else if (input.find("greater")!=std::string::npos)
    {
      if (input.find("Fermion")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Fermion_greater_self;
      else if(input.find("Boson")!=std::string::npos)
        Propagator_types[input]=NemoPhys::Boson_greater_self;
      else
        throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
    }
    else
      throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
  }
  else
    throw std::invalid_argument(prefix+"unknown NEGF-object name: \""+input+"\"\n");
  NemoUtils::toc(tic_toc_prefix);
}
void Propagation::scattering_lambda_proportional_to_G(Propagator*& output_Propagator, const Propagator*& input_Propagator,
    Simulation* input_Propagator_solver, const double lambda,
    const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::scattering_lambda_proportional_to_G ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation("+this->get_name()+")::retarded_scattering_lambda_proportional_to_G: ";

  delete result;
  //1. copy the Green's function from the input Propagator
  PetscMatrixParallelComplex* temp_matrix_pointer=NULL;
  NEMO_ASSERT(input_Propagator_solver!=NULL,prefix+"solver of \""+input_Propagator->get_name()+"\" is NULL\n");
  //input_Propagator_solver->get_data(input_Propagator->get_name(),&momentum_point,temp_matrix_pointer);

  NemoPhys::Propagator_type input_Propagator_type;
  NemoPhys::Propagator_type output_Propagator_type = output_Propagator->get_propagator_type();
  if (output_Propagator_type == NemoPhys::Fermion_retarded_self)
    input_Propagator_type = NemoPhys::Fermion_retarded_Green;
  else if (output_Propagator_type == NemoPhys::Fermion_lesser_self)
    input_Propagator_type = NemoPhys::Fermion_lesser_Green;
  GreensfunctionInterface* GF_interface = dynamic_cast<GreensfunctionInterface*>(input_Propagator_solver);
  GF_interface->get_Greensfunction(momentum_point, temp_matrix_pointer, NULL, NULL, input_Propagator_type);

  //2. multiply the Green's function with the lambda
  //2.1 if this is a container, assemble it before copying
  if(temp_matrix_pointer->if_container())
    temp_matrix_pointer->assemble();

  result = new PetscMatrixParallelComplex(*temp_matrix_pointer);
  //self-energy is dense
  *result *= std::complex<double> (lambda,0.0);

  if(debug_output)
  {
    std::string temp_string;
    const std::vector<NemoMeshPoint>* temp_pointer=&momentum_point;
    translate_momentum_vector(temp_pointer, temp_string);
    //temp_string+="_slptG_"+options.get_option("output_suffix",std::string(""));
    temp_string+="_slptG_"+get_output_suffix();
    temp_matrix_pointer->save_to_matlab_file(input_Propagator->get_name()+temp_string+".m");
    result->save_to_matlab_file(output_Propagator->get_name()+temp_string+".m");
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::scattering_lambda_proportional_to_G(Propagator*& output_Propagator, const Propagator*& input_Propagator,
    Simulation* input_Propagator_solver, const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::scattering_lambda_proportional_to_G ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Propagation("+this->get_name()+")::retarded_scattering_lambda_proportional_to_G: ";

  delete result;
  result = NULL;
  //1. copy the Green's function from the input Propagator
  PetscMatrixParallelComplex* temp_matrix_pointer=NULL;
  NEMO_ASSERT(input_Propagator_solver!=NULL,prefix+"solver of \""+input_Propagator->get_name()+"\" is NULL\n");
  input_Propagator_solver->get_data(input_Propagator->get_name(),&momentum_point,temp_matrix_pointer);
  //2. multiply the Green's function with the lambda
  //2.1 if this is a container, assemble it before copying
  if(temp_matrix_pointer->if_container())
    temp_matrix_pointer->assemble();

  std::string Hamilton_constructor_name = options.get_option("Hamilton_constructor", std::string(""));

  Simulation* Hamilton_Constructor;
  Hamilton_Constructor = this->find_simulation(Hamilton_constructor_name);
  NEMO_ASSERT(Hamilton_Constructor!=NULL,
		  tic_toc_prefix+"Hamilton_constructor has not been found!\n");

  const Domain* dom = this->get_simulation_domain();
  const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*>(dom);
  const AtomicStructure& atoms = domain->get_atoms();

  unsigned int num_active_atoms = 0;

  for (ConstActiveAtomIterator it=atoms.active_atoms_begin(); it!=atoms.active_atoms_end(); ++it)
  {
	  num_active_atoms++;
  }

  unsigned int basis_size=0;
  unsigned int AtomNum = 0;
  HamiltonConstructor* tb_ham = NULL;
  std::vector<std::string>param_group(1);
  param_group[0] = "Transport";
  double lambda = 0.0;
  std::map<std::string,double> lambda_map;
  std::vector<std::complex<double> > lambda_vector;

  for (ConstActiveAtomIterator it=atoms.active_atoms_begin(); it!=atoms.active_atoms_end(); ++it)
  {
	  const Material* material = it.node().atom->get_material();
	  tb_ham = dynamic_cast<HamiltonConstructor*>(Hamilton_Constructor->get_material_properties(material));
	  basis_size = tb_ham->get_number_of_orbitals(it.node().atom);

	  std::string atom_tag = material->get_tag();

	  map<std::string,double >::iterator lambda_it_end = lambda_map.end();
	  map<std::string,double >::iterator lambda_it;

	  lambda_it = lambda_map.find(atom_tag);

	  if (lambda_it == lambda_it_end)
	  {
		  MaterialProperties* material_properties = Hamilton_Constructor->get_material_properties(material);
		  lambda = material_properties->query_database_for_material(material->get_name(),param_group,"scattering_lambda",atom_tag);
		  lambda_map.insert(pair<std::string,double>(atom_tag,lambda));
	  }

	  lambda_it = lambda_map.find(atom_tag);
	  for (unsigned int i= AtomNum*basis_size;i < AtomNum*basis_size + basis_size; i++)
	  {
		  lambda_vector.push_back(lambda_it->second);
	  }
	  AtomNum++;
   }

  PetscMatrixParallelComplex* LambdaMat = new PetscMatrixParallelComplex(AtomNum*basis_size,AtomNum*basis_size,get_simulation_domain()->get_communicator());
  LambdaMat->allocate_memory();
  LambdaMat->set_to_zero();
  for(unsigned int i=0;i<AtomNum*basis_size;i++)
	  LambdaMat->set(i,i,lambda_vector[i]);
  LambdaMat->assemble();
  PetscMatrixParallelComplex::mult(*LambdaMat,*temp_matrix_pointer,&result);

  if(debug_output)
  {
    std::string temp_string;
    const std::vector<NemoMeshPoint>* temp_pointer=&momentum_point;
    translate_momentum_vector(temp_pointer, temp_string);
    //temp_string+="_slptG_"+options.get_option("output_suffix",std::string(""));
    temp_string+="_slptG_"+get_output_suffix();
    temp_matrix_pointer->save_to_matlab_file(input_Propagator->get_name()+temp_string+".m");
    result->save_to_matlab_file(output_Propagator->get_name()+temp_string+".m");
    LambdaMat->save_to_matlab_file(output_Propagator->get_name()+"_lambda_matrix_"+temp_string+".m");
  }

  delete LambdaMat;
  LambdaMat = NULL;

  NemoUtils::toc(tic_toc_prefix);
}

MPI_Comm Propagation::get_momentum_communicator(const std::set<std::vector<NemoMeshPoint> >& input_momentum_set) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_momentum_communicator ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation("+this->get_name()+")::get_momentum_communicator: ";
  //1. loop over all NemoMeshPoints and store their NemoMeshes in a set
  std::set<NemoMesh*> temp_mesh_container;

  std::set<std::vector<NemoMeshPoint> >::const_iterator cit=input_momentum_set.begin();

  NEMO_ASSERT(input_momentum_set.size()<=1,prefix+"debugging stop\n");

  std::set<std::vector<NemoMeshPoint> >::iterator it=input_momentum_set.begin();
  for(; it!=input_momentum_set.end(); ++it)
  {
    for(unsigned int i=0; i<(*it).size(); i++)
    {
      NemoMeshPoint temp_point = (*it)[i];
      NemoMesh* temp_NemoMesh = temp_point.get_parent_mesh();
      NEMO_ASSERT(temp_NemoMesh!=NULL,prefix+"pointer to the parent mesh of a NemoMeshPoint was found to be NULL\n");
      temp_mesh_container.insert(temp_NemoMesh);
    }
  }
  NEMO_ASSERT(temp_mesh_container.size()<=1,prefix+"temp_mesh_container debugging stop\n");

  //2. translate the set into a vector
  std::vector<const MPIVariable*> Mesh_container(temp_mesh_container.size());
  it=input_momentum_set.begin();
  std::set<NemoMesh*>::iterator it2=temp_mesh_container.begin();
  int counter=0;
  for(; it2!=temp_mesh_container.end(); it2++)
  {
    Mesh_container[counter]=(*it2);
    counter++;
  }
  //3. get the communicator according to the found NemoMeshes
  MPI_Comm result_comm;
  int temp_rank;
  MPIEnvironment* temp_MPIenvironment=dynamic_cast<MPIEnvironment*> (Parallelizer->get_mpi_variable());
  NEMO_ASSERT(temp_MPIenvironment!=NULL, prefix+"cast into MPIEnvironment failed\n");



  temp_MPIenvironment->Comm_create(Mesh_container,result_comm,temp_rank);

  int temp_size;
  MPI_Comm_size(result_comm,&temp_size);

  NemoUtils::toc(tic_toc_prefix);
  //4. return the result
  return result_comm;
}

int Propagation::translate_MPI_rank(const std::vector<NemoMeshPoint>* queried_momentum, const MPI_Comm& input_communicator) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::translate_MPI_rank ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation("+this->get_name()+")::translate_MPI_rank: ";
  int result=0;
  //1. check whether this momentum is solved on this MPI-process
  std::map<std::vector<NemoMeshPoint>, int>::const_iterator c_it=global_job_list.find(*queried_momentum);
  NEMO_ASSERT(c_it!=global_job_list.end(),prefix+"inconsistent find result\n");
  int temp2;
  MPI_Comm_rank(MPI_COMM_WORLD,&temp2);

  int integration_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator,&integration_rank);
  //2. if 1. then store the local rank according to the input_communicator in result
  if(integration_rank==c_it->second)
  {
    MPI_Comm_rank(input_communicator,&result);
  }
  //3. sum the result of all involved MPI-processes and distribute (MPI_ALLreduce)
  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_SUM,input_communicator);
  NemoUtils::toc(tic_toc_prefix);
  //4. return the result
  return result;
}


void Propagation::get_data(const std::string& variable, std::map<unsigned int, double>& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data9 ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+get_name()+"\")::get_data() ";
  //const Domain* temp_domain=get_const_simulation_domain();

  if (variable=="electron_density" || variable == "free_charge")
  {
    if(density.size()==0)
      calculate_density();
    for (std::map<unsigned int, double>::const_iterator it = density.begin();
         it != density.end(); it++)
    {

      data[it->first] = - it->second;

    }
  }
  else if (variable=="derivative_electron_density_over_potential"||variable=="derivative_total_charge_density_over_potential")
  {
    calculate_density();
    /*calculate_density_Jacobian();*/
    density_Jacobian=density;
    density.clear();
    for (std::map<unsigned int, double>::const_iterator it = density_Jacobian.begin();
         it != density_Jacobian.end(); it++)
      data[it->first] = - it->second;
  }
  else
    throw runtime_error(prefix+"unknown variable " + variable + "\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::map<double, double>& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data10 ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+get_name()+"\")::get_data() ";
  if(variable=="transmission")
  {
    if(momentum_transmission.empty() || energy_resolved_transmission.empty())
      calculate_transmission();
    data=energy_resolved_transmission;
  }
  else
    throw std::invalid_argument(prefix+"called with unknown variable \""+variable+"\"\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::clean_all()
{
  std::string prefix = "Propagation(\""+get_name()+"\")::clean_all() ";
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::clean_all ");
  NemoUtils::tic(tic_toc_prefix);

  //1. loop over all writeable propagator
  //std::map<std::string, Propagator*>::iterator it=writeable_Propagators.begin();
  //for(; it!=writeable_Propagators.end(); it++)
  if(writeable_Propagator!=NULL)
  {
    ////1.1 check if this is the constructor of this propagator
    //std::map<std::string, Simulation*>::iterator constructor_it=pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator);
    //NEMO_ASSERT(constructor_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found the constructor of \""+name_of_writeable_Propagator+"\"\n");
    //if(constructor_it->second==this)
    {
      Propagator_is_initialized[name_of_writeable_Propagator]=false;
      //1.1.1 check that this propagator is initialized
      //std::map<std::string, bool>::iterator temp_it=Propagator_is_initialized.find(name_of_writeable_Propagator);
      //1.2 delete all propagator matrices
      delete_propagator_matrices(writeable_Propagator);
      //1.3 delete the propagator
      writeable_Propagator->Propagator::~Propagator();
      writeable_Propagator=NULL;
    }
    if(Propagation_is_initialized)
    {
      //std::map<std::string, std::map<std::vector<NemoMeshPoint>, bool> >::iterator map_it=job_done_momentum_map.find(name_of_writeable_Propagator);
      //NEMO_ASSERT(map_it!=job_done_momentum_map.end(),prefix+"have not found the job_done_momentum_map of \""+name_of_writeable_Propagator+"\"\n");
      job_done_momentum_map.clear();
    }
  }

  //2. delete member matrices/elements
  energy_resolved_transmission.clear();
  delete temporary_sub_matrix;
  temporary_sub_matrix=NULL;

  delete into_device_propagating_modes;
  into_device_propagating_modes = NULL;
  delete into_device_decaying_modes;
  into_device_decaying_modes = NULL;
  delete out_of_device_propagating_modes;
  out_of_device_propagating_modes = NULL;
  delete out_of_device_decaying_modes;
  out_of_device_decaying_modes = NULL;

  delete into_device_propagating_phase;
  into_device_propagating_phase = NULL;
  delete into_device_decaying_phase;
  into_device_decaying_phase = NULL;
  delete out_of_device_propagating_phase;
  out_of_device_propagating_phase = NULL;
  delete out_of_device_decaying_phase;
  out_of_device_decaying_phase = NULL;

  delete  into_device_modes;
  into_device_modes = NULL;
  delete  out_of_device_modes;
  out_of_device_modes = NULL;
  delete  into_device_phase;
  into_device_phase = NULL;
  delete  out_of_device_phase;
  out_of_device_phase = NULL;

  delete  into_device_velocity;
  into_device_velocity = NULL;
  delete  out_of_device_velocity;
  out_of_device_velocity = NULL;
  delete  into_device_propagating_velocity;
  into_device_propagating_velocity = NULL;
  delete  out_of_device_propagating_velocity;
  out_of_device_propagating_velocity = NULL;

  PropagationUtilities::destroy_parallelization(this,momentum_mesh_types, Mesh_Constructors);
  bool was_this_parallelizer = parallelized_by_this(); //Bozidar added. Look a few lines below for explanation.
  _parallelized_by_this=false;

  Momentum_type_map.clear();
  momentum_mesh_types.clear();

  for(std::map<std::string,Simulation*>::iterator it = Mesh_Constructors.begin(); it!=Mesh_Constructors.end(); ++it)
  {
    if(it->second==this)
    {
      std::map<std::string,NemoMesh*>::iterator Mesh_it=Momentum_meshes.find(it->first);
      if(Mesh_it!=Momentum_meshes.end())
        if(Mesh_it->second!=NULL)
          delete Mesh_it->second;
    }
  }
  Momentum_meshes.clear();

  //Bozidar modified mesh deletion. First, take into account that mesh might already be deleted by external constructor so do not attempt to find mesh name
  //directly, but by using combination of top down mesh hierarchy map and Mesh_tree_names variable. Second, do this only if the mesh tree is constructed and if
  //this solver is parallelizer. The rule is: grid can be deleted either by grid parallelizer (indirectly through setting job to false) or grid constructor
  //(directly).
  if(Mesh_tree_topdown.size() > 0 && was_this_parallelizer)
  {
    std::map<NemoMesh*, std::vector<NemoMesh*> >::iterator topdown_it = Mesh_tree_topdown.begin();
    if(Mesh_Constructors.find(Mesh_tree_names[0])->second == this)
    {
      //DM: there is a HUGE memory leak here causing segfaults when valeys are enabled, added temporal option to disable meshes' destruction.
      if(options.get_option("remove_internal_meshes", true))
        delete topdown_it->first;
    }
    else
    {
      if(options.get_option("remove_meshes", true))
        Mesh_Constructors.find(Mesh_tree_names[0])->second->init_job_false();
    }
    for(unsigned int i = 1; i < Mesh_tree_names.size(); i++)
    {
      NEMO_ASSERT(topdown_it != Mesh_tree_topdown.end(), prefix + "Size of Mesh_tree_names is different than size of slim Mesh_tree_topdown.\n");
      if(Mesh_Constructors.find(Mesh_tree_names[i])->second == this)
      {
        for(unsigned int j = 0; j < topdown_it->second.size(); j++)
        {
          //DM: there is a HUGE memory leak here causing segfaults when valeys are enabled, added temporal option to disable meshes' destruction.
          if(options.get_option("remove_internal_meshes", true))
            delete topdown_it->second[j];
        }
      }
      else
      {
        if(options.get_option("remove_meshes", true))
          Mesh_Constructors.find(Mesh_tree_names[i])->second->init_job_false();
      }
      topdown_it = Mesh_tree_topdown.find(topdown_it->second[0]);
    }
  }
  //Old way!
  //std::map<NemoMesh*, NemoMesh*>::iterator downtop_it=Mesh_tree_downtop.begin();
  //for(; downtop_it!=Mesh_tree_downtop.end(); downtop_it++)
  //{
  //  string mesh_const_name = downtop_it->first->get_name();
  //  if(Mesh_Constructors.find(mesh_const_name)->second==this)
  //  {
  //    //DM: there is a HUGE memory leak here causing segfaults when valeys are enabled, added temporal option to disable meshes' destruction.
  //    if (options.get_option("remove_internal_meshes",true))
  //      delete downtop_it->first;
  //  }
  //}

  Mesh_tree_downtop.clear();
  Mesh_tree_topdown.clear();

  Mesh_tree_names.clear();
  Mesh_Constructors.clear();

  global_job_list.clear();
  energy_resolved_transmission.clear();
  momentum_transmission.clear();
  Propagation_is_initialized=false;

  momentum_meshes_are_set=false;
  momentum_dependencies.clear();
  analytical_momentum_integration.clear();
  input_Propagator_map.clear();
  Propagators.clear();
  Propagator_types.clear();
  density_Jacobian.clear();
  energy_resolved_per_k_density_for_output.clear();
  complex_energy_resolved_density_for_output.clear();
  energy_resolved_density_for_output.clear();
  density.clear();
  _wave_function_map.clear(); //Bozidar
  _pivot_map.clear(); //Bozidar
  Propagator_is_initialized.clear();
  //Propagator_Callers.clear();
  pointer_to_Propagator_Constructors->clear();
  pointer_to_all_momenta->clear();
  pointer_to_local_momenta->clear();
  energy_orbital_resolved_result.clear();
  output_counter = 0;
  Hamilton_Constructor=NULL;

  NemoUtils::toc(tic_toc_prefix);
}

const Domain* Propagation::get_simulation_domain_for_output() const
{
  std::string prefix="Propagation(\""+get_name()+"\")::get_simulation_domain_for_output() ";
  std::string nondistributed_hamiltonian_name=options.get_option("nondistributed_Hamilton_constructor",Hamilton_Constructor->get_name());
  const Simulation* nondistributed_Hamilton_constructor = find_const_simulation(nondistributed_hamiltonian_name);
  NEMO_ASSERT(nondistributed_Hamilton_constructor!=NULL,prefix+"have not found simulation \""+nondistributed_hamiltonian_name+"\"\n");
  return nondistributed_Hamilton_constructor->get_const_simulation_domain();
}

void Propagation::activate_regions_domain_for_output()
{
  std::string prefix="Propagation(\""+get_name()+"\")::get_simulation_domain_for_output() ";
  std::string nondistributed_hamiltonian_name=options.get_option("nondistributed_Hamilton_constructor",Hamilton_Constructor->get_name());
  Simulation* nondistributed_Hamilton_constructor = find_simulation(nondistributed_hamiltonian_name);
  NEMO_ASSERT(nondistributed_Hamilton_constructor!=NULL,prefix+"have not found simulation \""+nondistributed_hamiltonian_name+"\"\n");
  nondistributed_Hamilton_constructor->activate_regions();
}

bool Propagation::output_done_here(const std::string& option) const
{
  //1. calculation of the size of the geometry_communicator
  int size;
  MPI_Comm_size(get_const_simulation_domain()->get_communicator(),&size);
  //2. calculation of the rank of the geometry_communicator
  int geometry_rank;
  MPI_Comm_rank(get_const_simulation_domain()->get_communicator(),&geometry_rank);
  //3. get the geometry replica number
  int replication_number = get_const_simulation_domain()->get_geometry_replica();
  //4.a if option!=end - if 2. and 3.==0 return true, else false
  if(option!=std::string("end"))
  {
    if(geometry_rank==0 && replication_number==0) return true;
    else return false;
  }
  //4.b if option==end - if 3. == 0 and 2. == size-1 return true, else false
  else
  {
    if(geometry_rank==size-1 && replication_number==0) return true;
    else return false;
  }
}

void Propagation::get_data(const std::string& variable, std::set<unsigned int>& data)
{
  std::string prefix="Propagation(\""+get_name()+"\")::get_data() ";
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data ");
  NemoUtils::tic(tic_toc_prefix);
  if(variable=="Hamilton_momenta_indices")
  {
    data.clear();
    NEMO_ASSERT(Propagators.size()>0,prefix+"called with empty Propagators-map\n");
    std::set<unsigned int>* temp_pointer=&data;
    PropagationUtilities::find_Hamiltonian_momenta(this,Propagators.begin()->second,temp_pointer);
  }
  else
    throw std::runtime_error(prefix+"called with unknown variable \""+variable+"\"\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, Simulation*& result)
{
  std::string prefix="Propagation(\""+get_name()+"\")::get_data() ";
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data ");
  NemoUtils::tic(tic_toc_prefix);
  if(variable=="Hamilton_constructor"||variable==Hamilton_Constructor->get_name())
    result=get_Hamilton_constructor();
  else
    throw std::runtime_error(prefix+"called with unknown variable \""+variable+"\"\n");
  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::get_data(const std::string& variable, std::map<std::string, Simulation* >*& data)
{
  std::string prefix="Propagation(\""+get_name()+"\")::get_data() std::map<std::string, std::set<Simulation*> > ";
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data std::map<std::string, std::set<Simulation*> >");
  NemoUtils::tic(tic_toc_prefix);
  if(variable=="Propagator_Constructors")
    data=pointer_to_Propagator_Constructors;
  else
    throw std::runtime_error(prefix+"called with unknown variable \""+variable+"\"\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::calculate_optical_absorption(void)
{
  std::string prefix="Propagation(\""+get_name()+"\")::calculate_optical_absorption() ";
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::calculate_optical_absorption ");
  NemoUtils::tic(tic_toc_prefix);


  //1. Get photon energy range, number of points from inputdeck (options.get_option())
  //local energy points (from infrastructure)
  //nonlocal energy points (from infrastructure)
  //Get k space, number of k points

  //Define variables and allocate memory for:
  //sigmaM, epsilonM, gainM, temp_sigmaM,  % They all have size of photon_energy_points * lattice points

  //get or define the "nabla" operator, dz is a matrix of size N by N
  //for all "photon energies"
  //set temp_sigmaM empty
  //  for all "E points"
  //    locate "E+hv"
  //    for all "k points"
  //      calculate Ez
  //      if Ez < E
  //        locate "Ez+hv"
  //        obtain G_r, G_A, G_less(E, k)
  //        obtain Gr_ph, GA_ph, Gless_ph(E+hv, k)
  //        calculate tempT3 = -(dz*Gr_ph*dz*G_less+dz*G_less_ph*dz*G_A)+(Gr_ph*(dz*G_less*dz')+G_less_ph*(dz*G_A*dz'))
  //        (a) take the diagonal of tempT3 and multiply by a prefactor
  //        (b) calculate the diagnonal of G_less(E,k), multiply by another prefactor
  //          assemble the equation for sigma of this k point, store it.
  //       else
  //          result is zero
  //       end if
  //
  //    end all" k points"
  //    perform the k integration, store result for this energy point

  //  do the E integration, store result for this photon energy

  //end all "photon energies"

  //Calculate epsilonM(z,w)

  //calculate alpha(z,w)

  //store results, de-allocate memories, done!

  NemoUtils::toc(tic_toc_prefix);
}


void Propagation::set_description()
{
  description = "Solver to calculate phonon modes, spectra, and related quantities";
}



void Propagation::fill_offloading_momentum_map(const std::vector<NemoMeshPoint>& momentum_point,
    std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>& offloading_momentum_map)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\"" + tic_toc_name + "\")::fill_offloading_momentum_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\"" + this->get_name() + "\")::fill_offloading_momentum_map ";

  std::map<const std::vector<NemoMeshPoint>, ResourceUtils::OffloadInfo>::iterator it_momentum_offload_info =
    offloading_momentum_map.find(momentum_point);

  if (it_momentum_offload_info == offloading_momentum_map.end())
  {
    // Struct containing information regarding offloading such as load, coprocessor number, etc.
    ResourceUtils::OffloadInfo offload_info = ResourceUtils::get_offload_info();

    offload_solver->get_offload_task(Nemo::instance().total_communicator, offload_info);
    offloading_momentum_map[momentum_point] = offload_info;
  }
  // else, this momentum at it_momenta has already been given an offload task

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::set<std::vector<NemoMeshPoint> >*& data)
{
  std::string prefix = "Propagation(\""+get_name()+"\")::get_data() ";
  if(variable=="all_momenta")
    data=pointer_to_all_momenta;
  else if (variable=="local_momenta")
    data=pointer_to_local_momenta;
  else
    throw std::runtime_error(prefix+"called with unknown variable \""+variable+"\"\n");
}

void Propagation::get_data(const std::string& variable, unsigned int& data)
{
  std::string prefix = "Propagation(\""+get_name()+"\")::get_data() ";
  if(variable == "number_of_momenta")
    data = number_of_momenta;
  else
    throw std::runtime_error(prefix+"called with unknown variable \""+variable+"\"\n");
}

void Propagation::get_data(const std::string& variable, std::map<std::string, NemoMesh*>*& data)
{
  std::string prefix = "Propagation(\""+get_name()+"\")::get_data() ";
  if(variable == "Momentum_meshes")
    data = pointer_to_Momentum_meshes;
  else
    throw std::runtime_error(prefix+"called with unknown variable \""+variable+"\"\n");
}

//void Propagation::test_debug()
//{
//  std::string prefix = "Propagation(\""+get_name()+"\")::test_debug() ";
//  //throw std::runtime_error("stop\n");
//  std::set<Simulation*>::iterator it=Propagation::list_of_Propagators.begin();
//  for(;it!=Propagation::list_of_Propagators.end();++it)
//  {
//    std::cerr<<prefix<<"____________________________\n";
//    std::cerr<<prefix<<(*it)->get_name()<<"\n";
//    //initialized output
//    std::map<std::string, bool>* temp_map_pointer=NULL;
//    (*it)->get_data(std::string("Propagator_is_initialized"),temp_map_pointer);
//    std::cerr<<prefix<<(*it)->get_name()<<" has stored: \n";
//    if(temp_map_pointer!=NULL)
//    {
//      std::map<std::string, bool>::iterator it2=temp_map_pointer->begin();
//      for(;it2!=temp_map_pointer->end();++it2)
//      {
//        std::string first=it2->first;
//        bool second=it2->second;
//        std::cerr<<prefix<<" is_initialized_map:"<<first<<" "<<second<<"\n";
//      }
//    }
//    //constructor output
//    std::map<std::string, Simulation* >* temp_map_pointer0=NULL;
//    (*it)->get_data(std::string("Propagator_Constructors"),temp_map_pointer0);
//    if(temp_map_pointer0!=NULL)
//    {
//      std::map<std::string, Simulation* >::iterator it0=temp_map_pointer0->begin();
//      for(;it0!=temp_map_pointer0->end();++it0)
//      {
//        std::string first=it0->first;
//        std::string second=it0->second->get_name();
//        std::cerr<<prefix<<" Constructor_map:"<<first<<" "<<second<<"\n";
//      }
//    }
//    //caller output
//    std::map<std::string, std::set<Simulation*> >* temp_map_pointer3=NULL;
//    (*it)->get_data(std::string("Propagator_Callers"),temp_map_pointer3);
//    if(temp_map_pointer3!=NULL)
//    {
//      std::map<std::string, std::set<Simulation*> >::iterator it4=temp_map_pointer3->begin();
//      for(;it4!=temp_map_pointer3->end();++it4)
//      {
//        std::set<Simulation*>::iterator set_it=it4->second.begin();
//        for(;set_it!=it4->second.end();++set_it)
//        {
//          std::string first=it4->first;
//          std::string second=(*set_it)->get_name();
//          std::cerr<<prefix<<" Caller_map:"<<first<<" "<<second<<"\n";
//        }
//      }
//    }
//    std::cerr<<prefix<<"____________________________\n";
//  }
//}


void Propagation::reinit_material_properties()
{
}

void Propagation::init_material_properties()
{
}

void Propagation::do_solve_H_lesser(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve");
  NemoUtils::tic(tic_toc_prefix);
  activate_regions();
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=output_Propagator->propagator_map.find(momentum_point);
  if(matrix_it!=output_Propagator->propagator_map.end())
    temp_matrix=matrix_it->second;
  //do_solve_H_lesser(output_Propagator,momentum_point,temp_matrix);
  PropagationUtilities::do_solve_H_lesser(this, output_Propagator,momentum_point,temp_matrix);
  set_job_done_momentum_map(&(output_Propagator->get_name()), &momentum_point, true);
  NEMO_ASSERT(is_ready(output_Propagator->get_name(),momentum_point),tic_toc_prefix+"still not ready\n");
  write_propagator(output_Propagator->get_name(),momentum_point, temp_matrix);
  conclude_after_do_solve(output_Propagator,momentum_point);
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::do_solve_H_lesser(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::do_solve_H_lesser ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::do_solve_H_lesser ";

  delete result;
  result=NULL;
  PetscMatrixParallelComplex* temp_result=NULL;
  //1. solve the first summand
  //1.1 get the readable Propagator matrices of the first summand
  //1.1.1 get the forward retarded Green's function gR of the i+1 domain
  //1.1.1.1 get the name of the gR, its solver and the dofmap of it
  NEMO_ASSERT(options.check_option("half_way_retarded_green"),prefix+"please define \"half_way_retarded_green\"\n");
  std::string half_way_gRname=options.get_option("half_way_retarded_green",std::string(""));
  Simulation* source_of_half_way_gR = find_source_of_data(half_way_gRname);
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
    half_way_DOF_solver=find_simulation(half_way_dof_solver_name);
    NEMO_ASSERT(half_way_DOF_solver!=NULL,prefix+"have not found solver \"half_way_dof_solver_name\"\n");
  }
  std::string variable_name=output_Propagator->get_name()+std::string("_lead_domain");
  std::string neighbor_domain_name;
  if (options.check_option(variable_name))
    neighbor_domain_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(prefix+"define \""+variable_name+"\"\n");
  const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);

  DOFmapInterface& half_way_DOFmap=half_way_DOF_solver->get_dof_map(neighbor_domain);///variable_name,row_domain);

  //1.1.1.4 get access to the actual matrix
  PetscMatrixParallelComplex* half_way_gR_matrix=NULL;
  source_of_half_way_gR->get_data(half_way_gRname,&momentum_point,half_way_gR_matrix,&half_way_DOFmap);
  //1.1.2 get the exact lesser Green's function G< of the i domain
  //1.1.2.1 get the name of the G< and its solver
  NEMO_ASSERT(options.check_option("exact_lesser_green"),prefix+"please define \"exact_lesser_green\"\n");
  std::string exact_GLname=options.get_option("exact_lesser_green",std::string(""));
  Simulation* source_of_exact_GL = find_source_of_data(exact_GLname);
  //1.1.2.2 get the dofmap of the exact G< section needed here
  NEMO_ASSERT(options.check_option("exact_DOFmap_solver"),prefix+"please define \"exact_DOFmap_solver\"\n");
  Simulation* exact_DOfmap_solver=find_simulation(options.get_option("exact_DOFmap_solver",std::string("")));
  NEMO_ASSERT(exact_DOfmap_solver!=NULL,prefix+"have not found \""+options.get_option("exact_DOFmap_solver",std::string(""))+"\"\n");
  //const DOFmap& exact_DOFmap=exact_DOfmap_solver->get_const_dof_map(get_const_simulation_domain());
  //1.1.2.3 get access to the actual matrix
  PetscMatrixParallelComplex* exact_GL_matrix=NULL;
  source_of_exact_GL->get_data(exact_GLname,&momentum_point,exact_GL_matrix,&get_const_dof_map(get_const_simulation_domain()));
  //1.2 get the coupling Hamiltonian T=H_i,i+1
  PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;
  PetscMatrixParallelComplex* temp_coupling_Hamiltonian=NULL;
  DOFmapInterface* coupling_DOFmap = &half_way_DOFmap; //NULL; //new DOFmap;
  DOFmapInterface* temp_pointer=coupling_DOFmap;
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
  QuantumNumberUtils::sort_quantum_number(momentum_point,sorted_momentum,options,momentum_mesh_types,Hamilton_Constructor);
  Hamilton_Constructor->get_data(std::string("Hamiltonian"),sorted_momentum,neighbor_domain,temp_coupling_Hamiltonian,
                                 coupling_DOFmap,get_const_simulation_domain());
  if(temp_pointer!=coupling_DOFmap)
    delete coupling_DOFmap;
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
  if(num_cols >= (int)(temp_coupling_Hamiltonian->get_num_rows()))
    offset = 0;
  for(int i=0; i<num_cols; i++)
    columns[i]=offset+i;
  //1.2.3.3 store the submatrix
  int num_rows=exact_GL_matrix->get_num_rows();
  coupling_Hamiltonian = new PetscMatrixParallelComplex(num_rows, num_cols, get_simulation_domain()->get_communicator());
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
  //coupling_Hamiltonian->save_to_matlab_file("temp_couple.m");
  delete temp_coupling_Hamiltonian;
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
  NEMO_ASSERT(options.check_option("half_way_lesser_green"),prefix+"please define \"half_way_lesser_green\"\n");
  std::string half_way_gLname=options.get_option("half_way_lesser_green",std::string(""));
  Simulation* source_of_half_way_gL = find_source_of_data(half_way_gLname);
  //2.1.1.2 get the DOFmap this gL is solved on - same as in 1.1.1.2
  //get the dofmap corresponding to the half_way domain - same as in 1.1.1.2
  //2.1.1.3 get access to the actual matrix
  PetscMatrixParallelComplex* half_way_gL_matrix=NULL;
  source_of_half_way_gL->get_data(half_way_gLname,&momentum_point,half_way_gL_matrix,&half_way_DOFmap);
  //2.1.2 get the exact retarded Green's function GR of the i domain
  //2.1.2.1 get the name of the GR and its solver
  NEMO_ASSERT(options.check_option("exact_retarded_green"),prefix+"please define \"exact_retarded_green\"\n");
  std::string exact_GRname=options.get_option("exact_retarded_green",std::string(""));
  Simulation* source_of_exact_GR = find_source_of_data(exact_GRname);
  //2.1.2.2 the dofmap of the exact GR agrees with this dofmap...
  //2.1.2.3 get access to the actual matrix
  PetscMatrixParallelComplex* exact_GR_matrix=NULL;
  source_of_exact_GR->get_data(exact_GRname,&momentum_point,exact_GR_matrix,&get_const_dof_map(get_const_simulation_domain()));
  //2.2 solve -GR T g< T'
  //2.2.1 multiply g< and T'
  PetscMatrixParallelComplex::mult(*half_way_gL_matrix,*temp_matrix,&temp_matrix2);
  //2.2.2 multiply temp_matrix2=g< T' with T
  PetscMatrixParallelComplex::mult(*coupling_Hamiltonian,*temp_matrix2,&temp_matrix3);
  delete temp_matrix2;
  temp_matrix2=NULL;
  delete temp_matrix;
  temp_matrix=NULL;
  delete coupling_Hamiltonian;
  coupling_Hamiltonian=NULL;
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

  set_job_done_momentum_map(&output_Propagator->get_name(),&momentum_point,true);
  NemoUtils::toc(tic_toc_prefix);

}

void Propagation::translate_vector_into_map(std::vector<std::complex<double> >& input_vector, const std::complex<double>& prefactor, bool store_real_part,
    std::map<unsigned int, double>& output_map)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::translate_vector_into_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::translate_vector_into_map ";

  //store the results in the result map
  //if the DOFmap size does not agree with the size of vector, we ask the matrix source how to translate the vector index into the atom_id
  //otherwise, we assume that there is a one to one correspondence
  output_map.clear();
  //a. get the DOFmap of the matrix source
  const DOFmapInterface& temp_DOFmap=get_const_dof_map(get_const_simulation_domain());
  //b. compare the size of the DOFMap with the vector dimension
  const unsigned int number_of_DOFs=temp_DOFmap.get_number_of_dofs();
  if(number_of_DOFs>input_vector.size())
  {
    atomic_output_only = true;
    //here, a transformation has contracted the result onto atoms only
    //c. get the map between vector index (which is the iterator number of the respective active atom) and the atom_id from DOFmap
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);

    //d. multiply result with prefactor and store in map (key is atom_id, value is density)
    for (unsigned int i=0; i<input_vector.size(); i++)
    {
      std::complex<double> temp_complex=input_vector[i]*prefactor;
      std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(i);
      NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),prefix+"have not found the atom_id\n");
      if(store_real_part)
        output_map[c_it->second]=temp_complex.real();
      else
        output_map[c_it->second]=temp_complex.imag();
    }
  }
  else if(number_of_DOFs==input_vector.size())
  {
    atomic_output_only = false;
    //this is for orbital resolved output
    //one exception is possible: if the original representation has only one DOF per atom (as in effective mass)
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);
    //const std::map<short,unsigned int>* temp_map=temp_DOFmap.get_atom_dof_map(index_to_atom_id_map.begin()->first);
    const std::map<short,unsigned int>* temp_map=temp_DOFmap.get_atom_dof_map(index_to_atom_id_map.begin()->second);
    NEMO_ASSERT(temp_map!=NULL, prefix+"get_atom_dof_map has given empty atom index map\n");
    if(temp_map->size()==1)
      atomic_output_only=true;
    if(!atomic_output_only)
    {
      //this is for orbital resolved output
      for (unsigned int i=0; i<input_vector.size(); i++)
      {
        std::complex<double> temp_complex=input_vector[i]*prefactor;
        if(store_real_part)
          output_map[i]=temp_complex.real();
        else
          output_map[i]=temp_complex.imag();
      }
    }
    else
    {
      //store result in map (key is atom_id, value is density)
      for (unsigned int i=0; i<input_vector.size(); i++)
      {
        std::complex<double> temp_complex=input_vector[i]*prefactor;
        std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(i);
        NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),prefix+"have not found the atom_id\n");
        if(store_real_part)
          output_map[c_it->second]=temp_complex.real();
        else
          output_map[c_it->second]=temp_complex.imag();
      }
    }
  }
  else
    throw std::runtime_error(prefix+"mismatch of number of DOFs and input_vector size\n");

  NemoUtils::toc(tic_toc_prefix);

}

void Propagation::translate_vector_into_map_real(std::vector<double> & input_vector,   std::map<unsigned int, double>& output_map)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::translate_vector_into_map_real ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::translate_vector_into_map_real ";

  //store the results in the result map
  //if the DOFmap size does not agree with the size of vector, we ask the matrix source how to translate the vector index into the atom_id
  //otherwise, we assume that there is a one to one correspondence
  output_map.clear();
  //a. get the DOFmap of the matrix source
  const DOFmapInterface& temp_DOFmap=get_const_dof_map(get_const_simulation_domain());
  //b. compare the size of the DOFMap with the vector dimension
  const unsigned int number_of_DOFs=temp_DOFmap.get_number_of_dofs();
  if(number_of_DOFs>input_vector.size())
  {
    atomic_output_only = true;
    //here, a transformation has contracted the result onto atoms only
    //c. get the map between vector index (which is the iterator number of the respective active atom) and the atom_id from DOFmap
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);

    //d. multiply result with prefactor and store in map (key is atom_id, value is density)
    for (unsigned int i=0; i<input_vector.size(); i++)
    {
      std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(i);
      NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),prefix+"have not found the atom_id\n");
      output_map[c_it->second]=input_vector[i];
    }
  }
  else if(number_of_DOFs==input_vector.size())
  {
    atomic_output_only = false;
    //this is for orbital resolved output
    //one exception is possible: if the original representation has only one DOF per atom (as in effective mass)
    std::map<int,int> index_to_atom_id_map;
    temp_DOFmap.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map,true);
    //const std::map<short,unsigned int>* temp_map=temp_DOFmap.get_atom_dof_map(index_to_atom_id_map.begin()->first);
    const std::map<short,unsigned int>* temp_map=temp_DOFmap.get_atom_dof_map(index_to_atom_id_map.begin()->second);
    NEMO_ASSERT(temp_map!=NULL, prefix+"get_atom_dof_map has given empty atom index map\n");
    if(temp_map->size()==1)
      atomic_output_only=true;
    if(!atomic_output_only)
    {
      //this is for orbital resolved output
      for (unsigned int i=0; i<input_vector.size(); i++)
      {
        output_map[i]=input_vector[i];
      }
    }
    else
    {
      //store result in map (key is atom_id, value is density)
      for (unsigned int i=0; i<input_vector.size(); i++)
      {
        double temp =input_vector[i];
        std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(i);
        NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),prefix+"have not found the atom_id\n");
        output_map[c_it->second] = temp;

      }
    }
  }
  else
    throw std::runtime_error(prefix+"mismatch of number of DOFs and input_vector size\n");

  NemoUtils::toc(tic_toc_prefix);

}

bool Propagation::complex_energy_used(void) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::complex_energy_used ");
  NemoUtils::tic(tic_toc_prefix);
  bool result = false;
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator c_it=momentum_mesh_types.begin();
  for(; c_it!=momentum_mesh_types.end()&&!result; ++c_it)
    if(c_it->second==NemoPhys::Complex_energy)
      result=true;
  NemoUtils::toc(tic_toc_prefix);
  return result;

}

double Propagation::find_integration_weight(const std::string& mesh_name, const std::vector<NemoMeshPoint>& momentum_point, const Propagator* input_Propagator)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::find_integration_weight ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::find_integration_weight ";
  //1. find the index of the momentum_point that corresponds to the mesh_name
  unsigned int mesh_index=input_Propagator->momentum_mesh_names.size();
  bool not_found=true;
  for(unsigned int i=0; i<input_Propagator->momentum_mesh_names.size()&&not_found; i++)
    if(input_Propagator->momentum_mesh_names[i]==mesh_name)
    {
      mesh_index=i;
      not_found=false;
    }
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
      std::vector<double> temp_vector=read_kvector_from_momentum(momentum_point, input_Propagator);
      NemoMeshPoint temp_momentum(0,temp_vector);
      std::vector<NemoMeshPoint> temp_vector_momentum(1,temp_momentum);
      c_it->second->get_data("integration_weight",temp_vector_momentum,momentum_point[mesh_index],result);
    }
    else
      c_it->second->get_data("integration_weight",momentum_point,momentum_point[mesh_index],result);
  }
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

std::vector<bool> Propagation::status_of_job_map(void) const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::status_of_job_map ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::status_of_job_map ";
  std::vector<bool> result(job_done_momentum_map.size(),false);
  //std::map<std::string, std::map<std::vector<NemoMeshPoint>, bool> >::const_iterator c_it=job_done_momentum_map.begin();
  //loop over the writeable propagators
  //for(; c_it!=job_done_momentum_map.end(); ++c_it)
  {
    std::map<std::vector<NemoMeshPoint>, bool>::const_iterator c_it2=job_done_momentum_map.begin();
    unsigned int counter2=0;
    //loop over the momentup tuples
    for(; c_it2!=job_done_momentum_map.end(); ++c_it2)
    {
      result[counter2]=c_it2->second;
      counter2++;
    }
  }
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

void Propagation::print_all_status_to_screen()
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::print_all_status_to_screen ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::print_all_status_to_screen ";
  //loop over all Propagator_Constructors
  std::map<std::string,Simulation*>::const_iterator c_it=Propagator_Constructors.begin();
  for(; c_it!=Propagator_Constructors.end(); ++c_it)
  {
    std::vector<bool> temp_vector=(dynamic_cast<Propagation*>(c_it->second))->status_of_job_map();
    OUT_DETAILED<<prefix<<"Job status of "<<c_it->first<<" is:\n";
    for(unsigned int i=0; i<temp_vector.size(); i++)
      OUT_DETAILED<<temp_vector[i]<<"\t";
    OUT_DETAILED<<"\n-------------------\n";
  }
  NemoUtils::toc(tic_toc_prefix);
}

double Propagation::get_eme_analytically_integrated_distribution(const bool electron_type)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_eme_analytically_integrated_distribution ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::get_eme_analytically_integrated_distribution ";
  double effective_mass=0.0;
  std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
  Simulation* temp_simulation=find_simulation(temp_simulation_name);
  NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
  if(electron_type)
    temp_simulation->get_data("averaged_effective_mass_conduction_band",effective_mass);
  else
    temp_simulation->get_data("averaged_effective_mass_valence_band",effective_mass);

  NemoUtils::toc(tic_toc_prefix);
  return effective_mass;
}

double Propagation::get_analytically_integrated_distribution(const double energy, const double temperature_in_eV, const double chemical_potential,
    const std::string& momentum_type, const bool electron_type, const bool take_derivative_over_E, const bool take_derivative_over_mu)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_analytically_integrated_distribution ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::get_analytically_integrated_distribution ";
  double effective_mass=0.0;
  std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
  Simulation* temp_simulation=find_simulation(temp_simulation_name);
  NEMO_ASSERT(temp_simulation!=NULL,prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
  if(electron_type)
    temp_simulation->get_data("averaged_effective_mass_conduction_band",effective_mass);
  else
    temp_simulation->get_data("averaged_effective_mass_valence_band",effective_mass);

  double result= get_analytically_integrated_distribution(energy,temperature_in_eV, chemical_potential,
                 momentum_type,electron_type, take_derivative_over_E, take_derivative_over_mu, effective_mass);
  NemoUtils::toc(tic_toc_prefix);
  return result;
}

double Propagation::get_analytically_integrated_distribution(const double energy, const double temperature_in_eV, const double chemical_potential,
    const std::string& momentum_type, const bool electron_type, const bool take_derivative_over_E, const bool take_derivative_over_mu,const double effective_mass)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_analytically_integrated_distribution_wo_eme ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::get_analytically_integrated_distribution ";
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
    //if (!options.get_option("integration_to_get_density", true))
    {
      const Domain* temp_domain=get_const_simulation_domain();
      result*=temp_domain->return_periodic_space_volume();
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
    const Domain* temp_domain=get_const_simulation_domain();
    result*=temp_domain->return_periodic_space_volume();
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

    //if (!options.get_option("integration_to_get_density", true))
    // {
    //const Domain* temp_domain=get_const_simulation_domain();
    //result*=temp_domain->return_periodic_space_volume();
    //NemoUtils::MsgLevel prev_level = msg.get_level();
    //NemoUtils::msg.set_level(NemoUtils::MsgLevel(3));
    //msg <<"integration_to_get_density = false\n";
    //NemoUtils::msg.set_level(prev_level);
    //}

    //NemoUtils::msg.set_level(prev_level);

  }

  NemoUtils::toc(tic_toc_prefix);
  return result;
}

void Propagation::delete_real_or_imag_part_of_all_propagators(const std::string& type)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::delete_real_or_imag_part_of_all_propagators ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::delete_real_or_imag_part_of_all_propagators ";

  //std::map<std::string, Propagator*>::iterator temp_it=writeable_Propagators.begin();
  //for(; temp_it!=writeable_Propagators.end(); ++temp_it)
  {
    Propagator::PropagatorMap::iterator matrix_it=writeable_Propagator->propagator_map.begin();
    for(; matrix_it!=writeable_Propagator->propagator_map.end(); ++matrix_it)
    {
      if(matrix_it->second->if_container())
        matrix_it->second->assemble();
      if(type=="imaginary")
        matrix_it->second->real_part();
      else
      {
        PetscMatrixParallelComplex temp_matrix(*(matrix_it->second));
        temp_matrix.real_part();
        matrix_it->second->add_matrix(temp_matrix,SAME_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::symmetrize_all_propagators(NemoMath::symmetry_type symm_type)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::delete_real_or_imag_part_of_all_propagators ");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Propagation(\""+this->get_name()+"\")::delete_real_or_imag_part_of_all_propagators ";

  Propagator::PropagatorMap::iterator matrix_it=writeable_Propagator->propagator_map.begin();
  for(; matrix_it!=writeable_Propagator->propagator_map.end(); ++matrix_it)
  {
    if(matrix_it->second->if_container())
      matrix_it->second->assemble();
    symmetrize(matrix_it->second,symm_type);
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::create_ordered_mesh_list(std::vector<std::string>& ordered_list)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::create_ordered_mesh_list ");
  NemoUtils::tic(tic_toc_prefix);

  std::vector<std::string> temp_momentum_names;
  if (!options.check_option("momentum_names"))
    throw std::invalid_argument(tic_toc_prefix+"define momentum_names, i.e. the mesh types\n");
  options.get_option("momentum_names",temp_momentum_names);
  NEMO_ASSERT(temp_momentum_names.size()>0,tic_toc_prefix+"received empty momentum_names list\n");
  std::set<std::string> list_of_momenta;
  for(unsigned int i=0; i<temp_momentum_names.size(); i++)
    list_of_momenta.insert(temp_momentum_names[i]);

  ordered_list.clear();
  ordered_list.resize(temp_momentum_names.size());
  //1. find the mesh without parent - that is the topmost mesh
  std::string current_topmesh;
  std::set<std::string>::iterator it=list_of_momenta.begin();
  for(; it!=list_of_momenta.end(); ++it)
  {
    if(!options.check_option((*it)+"_parent"))
    {
      current_topmesh=(*it);
      list_of_momenta.erase(it);
      break;
    }
  }

  ordered_list[0]=current_topmesh;
  unsigned int counter = 1;
  //2. loop over the remaining meshes and identify the mesh order
  it=list_of_momenta.begin();
  for(; it!=list_of_momenta.end(); ++it)
  {
    NEMO_ASSERT(options.check_option((*it)+"_parent"),tic_toc_prefix+"have not found parent mesh of \""+(*it)+"\"\n");
    std::string temp_parent_name=options.get_option((*it)+"_parent",std::string(""));
    //2.1 find the mesh with the current highest one as a parent
    if(temp_parent_name==current_topmesh)
    {
      ordered_list[counter]=(*it);
      counter++;
      current_topmesh=(*it);
      list_of_momenta.erase(it);
      if(list_of_momenta.size() != 0)
        it=list_of_momenta.begin();
      else
        break;
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}


unsigned int Propagation::find_energy_index() const
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::find_energy_index ");
  NemoUtils::tic(tic_toc_prefix);

  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
  std::string energy_name=std::string("");
  for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
    if(momentum_name_it->second==NemoPhys::Energy)
      energy_name=momentum_name_it->first;

  //NEMO_ASSERT(Propagators.size()>0,tic_toc_prefix+"called with empty Propagator list\n");
  const Propagator* temp_propagator=*(known_Propagators.begin()); //Propagators.begin()->second;
  NEMO_ASSERT(temp_propagator!=NULL,tic_toc_prefix+"found NULL for a readable Propagator\n");// \""+Propagators.begin()->first+"\"\n");
  const std::vector<NemoMeshPoint>& momentum_point=temp_propagator->propagator_map.begin()->first;
  unsigned int energy_index=momentum_point.size()+1; //larger then the vector size to throw exception if energy index is not found
  for(unsigned int i=0; i<temp_propagator->momentum_mesh_names.size(); i++)
    if(temp_propagator->momentum_mesh_names[i]==energy_name)
      energy_index=i;
  NEMO_ASSERT(energy_index!=momentum_point.size()+1,tic_toc_prefix+"have not found the energy index\n");
  NemoUtils::toc(tic_toc_prefix);
  return energy_index;
}

void Propagation::interpolate_nonrectangular_energy(const std::map<std::vector<NemoMeshPoint>, double>& input_data, std::map<double, double>& output_data,
    double number_of_interpolation_points)
{
  static int counter=0;
  counter++;
  std::stringstream stm;
  stm << counter;
  std::string potential_diff = stm.str();
  std::string prefix = "Propagation(\"" + this->get_name() + "\")::interpolate_nonrectangular_energy ";
  unsigned int energy_index = find_energy_index();
  //1. sort the data for each k-point (or tupel without energy)
  std::map<std::vector<NemoMeshPoint>, std::map<double, double> > sorted_data;
  //1.1 loop over all input_data
  std::map<std::vector<NemoMeshPoint>, double>::const_iterator cit = input_data.begin();
  for (; cit != input_data.end(); ++cit)
  {
    //1.2 extract non-energy part of the momentum vector
    std::vector<NemoMeshPoint> temp_vector;
    for (unsigned int i = 0; i < cit->first.size(); i++)
      if (i != energy_index)
        temp_vector.push_back(cit->first[i]);
    //1.3 fill the inputdata into sorted data container
    double energy = (cit->first[energy_index]).get_x();
    std::map<std::vector<NemoMeshPoint>, std::map<double, double> >::iterator it = sorted_data.find(temp_vector);
    if (it != sorted_data.end())
      it->second[energy] = cit->second;
    else
    {
      std::map<double, double> temp_map;
      temp_map[energy] = cit->second;
      sorted_data[temp_vector] = temp_map;
    }
  }
  //2. create a target energy mesh
  //2.1 find the mesh corners from the existing energy mesh
  double min_energy = 1e20;
  double max_energy = -1e20;
  //2.1.1 loop over all existing mesh points and check whether the respective one is larger/smaller the max/min_energy
  std::set<std::vector<NemoMeshPoint> >::const_iterator all_cit = pointer_to_all_momenta->begin();
  for (; all_cit != pointer_to_all_momenta->end(); ++all_cit)
  {
    const Propagator* temp_propagator=*(known_Propagators.begin());
    NEMO_ASSERT(temp_propagator!=NULL,prefix+"have not found non-NULL propagator\n");
    double temp_energy = PropagationUtilities::read_energy_from_momentum(this,*all_cit, temp_propagator);
    if (temp_energy < min_energy)
      min_energy = temp_energy;
    if (temp_energy > max_energy)
      max_energy = temp_energy;
  }
  //2.2 read from inputdeck how many interpolation points to use
  output_data.clear();
  double interpolation_spacing = (max_energy - min_energy) / ((double) number_of_interpolation_points);
  for (unsigned int i = 0; i < number_of_interpolation_points; i++)
  {
    double temp_energy = min_energy + interpolation_spacing * ((double) i);
    output_data[temp_energy] = 0.0;
  }
  //3. interpolate for each k-point (or energy-free tupel) the data to the target E-mesh
  std::map<std::vector<NemoMeshPoint>, std::map<double, double> >::iterator it = sorted_data.begin();
  std::map<std::vector<NemoMeshPoint>, std::map<double, double> >::iterator it_next = it;

  /*  std::string filename = get_name() + "_raw_transmission_debug" + potential_diff + get_output_suffix() + ".dat";
    std::string filename_e = get_name() + "_raw_transmission_debug_e" + potential_diff + get_output_suffix() + ".dat";

    std::ofstream out_file;
    std::ofstream out_file_e;
    out_file.open(filename.c_str());
    out_file_e.open(filename_e.c_str());
    it = sorted_data.begin();
    std::map<double, double>::iterator it_e;
   for (; it != sorted_data.end(); it++)
    {
      it_e = it->second.begin();
      while (it_e != it->second.end())
      {
        out_file_e << (it_e->first) << "\t";
        it_e++;
      }
      out_file_e << "\n";
    }
    out_file_e.close();*/
  //3.1 loop over all non-energy points
  std::map<double, double>::iterator out_it = output_data.begin();
  std::map<double, double>::iterator it_bound;
  double temp_energy = 0;
  double close_energy1 = 0;
  double close_value1 = 0;
  double close_energy2 = 0;
  double close_value2 = 0;
  for (; out_it != output_data.end(); ++out_it)
  {
    it = sorted_data.begin();
    for (; it != sorted_data.end(); ++it)
    {
      //3.2 find the integration weight for this non-energy point (if not turned off)
      const Propagator* temp_Propagator = writeable_Propagator; //writeable_Propagators.begin()->second;
      //1.4 find the integration weight for this specific momentum (not energy)
      double integration_weight = 1.0;
      //if (!options.get_option("no_integration_for_transmission", false))
      {
        std::string momentum_name = std::string("");
        for (unsigned int i = 0; i < temp_Propagator->momentum_mesh_names.size(); i++)
        {
          //find the momentum name that does not contintegration_weightain "energy"
          if (temp_Propagator->momentum_mesh_names[i].find("energy") == std::string::npos)
          {
            momentum_name = temp_Propagator->momentum_mesh_names[i];
            //find the momentum mesh constructor and ask it for the weight
            std::map<std::string, Simulation*>::const_iterator c_it2 = Propagation::Mesh_Constructors.begin();
            c_it2 = Propagation::Mesh_Constructors.find(momentum_name);
            NEMO_ASSERT(c_it2!=Propagation::Mesh_Constructors.end(), prefix+"have not found constructor for momentum mesh \""+momentum_name+"\"\n");
            double temp;
            //check whether the momentum solver is providing a non rectangular mesh
            InputOptions& mesh_options = c_it2->second->get_reference_to_options();

            if (mesh_options.get_option(std::string("non_rectangular"), false))
              c_it2->second->get_data(std::string("integration_weight"), (it->first), (it->first)[i], temp);
            else
              c_it2->second->get_data("integration_weight", (it->first)[i], temp);

            integration_weight *= temp;

            if(momentum_name.find("momentum_1D")!=std::string::npos)
              integration_weight/=2.0*NemoMath::pi;
            if(momentum_name.find("momentum_2D")!=std::string::npos)
              integration_weight/=4.0*NemoMath::pi*NemoMath::pi;
            else if(momentum_name.find("momentum_3D")!=std::string::npos)
              integration_weight/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
          }
        }
      }
      bool found_upper, found_lower;
      temp_energy = out_it->first;
      it_bound = lower_bound(it->second, temp_energy, found_lower);
      close_energy1 = it_bound->first;
      close_value1 = it_bound->second;
      it_bound = upper_bound(it->second, temp_energy, found_upper);
      close_energy2 = it_bound->first;
      close_value2 = it_bound->second;
      double temp = 0;
      if ((found_upper == true) && (found_lower == true))
      {
        if (close_energy1 != close_energy2)
        {
          temp = (close_value1 + (close_value2 - close_value1) * std::abs(temp_energy - close_energy1) / std::abs(close_energy2 - close_energy1));
          out_it->second += (temp * integration_weight);
        }
        else
        {
          out_it->second += (close_value1 * integration_weight);
        }
      }

      //out_file << close_energy1 << "  ,  " << close_value1 << "   ,   " << close_energy2 << "  ,  " << close_value2 << "  ,  " << temp_energy << "  ,  "
      //    << temp << "   ,  " << integration_weight << "  ,  " << out_it->second << "\n";
      //  out_file << found_upper << "  ,  " << found_lower << "\n";
    }
    // out_file << "End of  k point   " << "\n\n\n\n\n";
  }
  //out_file.close();

}

std::map<double, double>::iterator lower_bound(std::map<double, double>& output_data, double value, bool& found)
{
  std::map<double, double>::iterator it = output_data.begin();
  std::map<double, double>::iterator it_min = output_data.begin();
  found = false;
  for (; it != output_data.end(); it++)
  {
    if ((it->first <= value) && ((found == false) || (it->first > it_min->first)))
    {
      it_min = it;
      found = true;
    }
  }
  return it_min;
}
std::map<double, double>::iterator upper_bound(std::map<double, double>& output_data, double value, bool& found)
{
  std::map<double, double>::iterator it = output_data.begin();
  std::map<double, double>::iterator it_max = output_data.begin();
  found = false;
  for (; it != output_data.end(); it++)
  {
    if ((it->first >= value) && ((found == false) || (it->first < it_max->first)))
    {
      it_max = it;
      found = true;
    }

  }
  return it_max;
}
void Propagation::get_data(const std::string& variable, const unsigned int input_index, double& result)
{
  double temp;
  Hamilton_Constructor->get_data(variable, input_index, temp);
  result = temp;
}

void Propagation::get_data(const std::string& variable, std::map<std::vector<double>,std::vector<double> >& data,vector<double>*momentum)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data refineable mesh ");
  NemoUtils::tic(tic_toc_prefix);
  if(variable.find("k_resolved_density")!=std::string::npos)
  {
    if(density_is_ready)
    {

      //if(k_resolved_density_for_output.empty())
      //{
      //  k_resolved_density = true;
      //  calculate_density();
      //  k_resolved_density = false;;
      //}
      if(k_resolved_1D_vector.empty())
      {

        //loop through k_resolved_density_for_output and store into flattened vector
        std::map<vector<double>, std::map<unsigned int, double> >::iterator it = k_resolved_density_for_output.begin();
        for(; it!=k_resolved_density_for_output.end(); ++it)
        {
          std::map<unsigned int, double>::iterator data_it = it->second.begin();
          vector<double> temp_vector(it->second.size());
          int counter = 0;
          for(; data_it != it->second.end(); ++data_it)
          {
            temp_vector[counter] = data_it->second;
            counter++;
          }
          k_resolved_1D_vector[it->first] = temp_vector;
        }
      }
      data = k_resolved_1D_vector;
    }
    else
      data.empty();
  }
  else if(variable.find("e_resolved_density")!=std::string::npos)
  {
    if(density_is_ready)
    {
      if(momentum == NULL) //assume nonrectangular
      {
        if(energy_resolved_density_for_output.empty())
        {
          calculate_density();
        }
        if(energy_resolved_1D_vector.empty())
        {
          //loop through k_resolved_density_for_output and store into flattened vector
          std::map<double, std::map<unsigned int, double> >::iterator it = energy_resolved_density_for_output.begin();
          for(; it!=energy_resolved_density_for_output.end(); ++it)
          {
            std::map<unsigned int, double>::iterator data_it = it->second.begin();
            vector<double> temp_vector(it->second.size());
            int counter = 0;
            for(; data_it != it->second.end(); ++data_it)
            {
              temp_vector[counter] = data_it->second;
              counter++;
            }
            std::vector<double> temp_key_vector(1,it->first);
            energy_resolved_1D_vector[temp_key_vector] = temp_vector;
          }
          data = energy_resolved_1D_vector;
        }
      }
      else
      {
        std::map<vector<double>, std::map<double, std::map<unsigned int, double> > > ::iterator k_it = energy_resolved_per_k_density_for_output.find(*momentum);
        NEMO_ASSERT(k_it!=energy_resolved_per_k_density_for_output.end(), tic_toc_prefix + " did not find momentum in energy resolved per k density");

        std::map<vector<double>, vector<double> > energy_resolved_per_k_1D_vector;
        std::map<double, std::map<unsigned int, double> >::iterator e_it = k_it->second.begin();

        for(; e_it!=k_it->second.end(); ++e_it)
        {
          std::map<unsigned int, double>::iterator data_it = e_it->second.begin();
          vector<double> temp_vector(e_it->second.size());
          int counter = 0;
          for(; data_it != e_it->second.end(); ++data_it)
          {
            temp_vector[counter] = data_it->second;
            counter++;
          }
          std::vector<double> temp_key_vector(1,e_it->first);
          energy_resolved_per_k_1D_vector[temp_key_vector] = temp_vector;
        }
        data = energy_resolved_per_k_1D_vector;
      }
    }
    else
      data.empty();

  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::get_data(const std::string& variable, std::map<std::vector<NemoMeshPoint>,std::vector<double> >& data, double prefactor, const DOFmapInterface* row_dof_map, const DOFmapInterface* col_dofmap)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data (diagonals of Propagator matrices) ");
  bool return_real=true;
  if(variable.find("imag")!=std::string::npos)
    return_real=false;
  data.clear();
  NEMO_ASSERT(writeable_Propagator!=NULL, tic_toc_prefix+"called with writeable_Propagator==NULL\n");
  std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>& temp_prop=writeable_Propagator->propagator_map;
  std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator cit=temp_prop.begin();
  for(;cit!=temp_prop.end();++cit)
  {
    PetscMatrixParallelComplex* Matrix = NULL;
    get_data(name_of_writeable_Propagator, &(cit->first), Matrix, row_dof_map, col_dofmap);
    std::vector<std::complex<double> > temp_vector;
    NEMO_ASSERT(Matrix!=NULL,tic_toc_prefix+"received NULL for Matrix pointer\n");
    Matrix->get_diagonal(&temp_vector);
    std::vector<double> temp_result(temp_vector.size(),0.0);
    if(return_real)
    {
      for(unsigned int i=0;i<temp_vector.size();i++)
        temp_result[i] = prefactor*temp_vector[i].real();
    }
    else
    {
      for(unsigned int i=0;i<temp_vector.size();i++)
        temp_result[i] = prefactor*temp_vector[i].imag();
    }
    data[cit->first]= temp_result;
  }
}

void Propagation::get_data(const std::string& , std::map<std::vector<NemoMeshPoint>,std::vector<std::complex<double> > >& data)
{
  std::string tic_toc_prefix = "Propagation(\""+tic_toc_name+"\")::get_data (diagonals of Propagator matrices) ";
  data.clear();
  NEMO_ASSERT(writeable_Propagator!=NULL, tic_toc_prefix+"called with writeable_Propagator==NULL\n");
  std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>& temp_prop=writeable_Propagator->propagator_map;
  std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator cit=temp_prop.begin();
  for(;cit!=temp_prop.end();++cit)
  {
    PetscMatrixParallelComplex* Matrix = NULL;
    get_data(name_of_writeable_Propagator, &(cit->first), Matrix);
    std::vector<std::complex<double> > temp_vector;
    NEMO_ASSERT(Matrix!=NULL,tic_toc_prefix+"received NULL for Matrix pointer\n");
    Matrix->get_diagonal(&temp_vector);
    data[cit->first]= temp_vector;
  }
}

void Propagation::get_data(const std::string& variable, std::set<vector<double> >& data)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::get_data ");
   NemoUtils::tic(tic_toc_prefix);
  if(variable == "all_kvectors")
  {
    if(pointer_to_all_momenta->size()==0)
    {
      //Propagator *temp_propagator = const_cast<Propagator*>(*(known_Propagators.begin()));
       Parallelizer->get_data("all_momenta",pointer_to_all_momenta);
    }
    NEMO_ASSERT(pointer_to_all_momenta->size()>0,tic_toc_prefix+"received empty all_momenta\n");
    std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
    //store unique k-vectors in set
    std::set<vector<double> > temp_kvectors;
    //loop through all momenta
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      std::vector<double> momentum_point=read_kvector_from_momentum(*c_it, writeable_Propagator);
      temp_kvectors.insert(momentum_point);
    }
    data = temp_kvectors;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::integrate_vector(const std::map<std::vector<NemoMeshPoint>,std::vector<double> >& input_data,std::vector<double>& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::integrate_scalar ");
  NemoUtils::tic(tic_toc_prefix);
  result.clear();
  result.resize(input_data.begin()->second.size(),0.0);

  if(Mesh_tree_topdown.size()==0)
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");
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
      double temp_double=find_integration_weight(momentum_mesh_name, c_it->first, writeable_Propagator);
      integration_weight *= temp_double;
    }
    for(unsigned int i=0;i<result.size();i++)
      result[i]+=integration_weight*c_it->second[i];
  }
  //3.communicate the result to get the global result
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  MPI_Barrier(topcomm);
  std::vector<double> temp_result=result;
  if(!solve_on_single_replica)
    MPI_Allreduce(&(temp_result[0]),&(result[0]),result.size(), MPI_DOUBLE, MPI_SUM ,topcomm);
  else
    MPI_Reduce(&(temp_result[0]),&(result[0]),result.size(), MPI_DOUBLE, MPI_SUM, 0, topcomm);

  NemoUtils::toc(tic_toc_prefix);
}

void Propagation::integrate_vector_for_energy_resolved(std::map<std::vector<NemoMeshPoint>,
                                                            std::vector<std::complex<double> >*>*& input_data,
                                                            std::map<double,std::vector<std::complex<double> > >*& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::integrate_vector_for_energy_resolved ");
  std::string prefix="Propagation::(\""+this->get_name()+"\")::integrate_vector_for_energy_resolved: ";

  NemoUtils::tic(tic_toc_prefix);


  std::vector<std::complex<double> > temp_result(input_data->size(),std::complex<double>(0.0,0.0));

  unsigned int data_size = input_data->begin()->second->size();

  if(Mesh_tree_topdown.size()==0)
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");


  NEMO_ASSERT(writeable_Propagator!=NULL,tic_toc_prefix+"called with writeable_Propagator NULL\n");

  //get unique energies
  //1.1 if energy resolved data is wanted, prepare the required container and the all_energies
  std::vector<double> all_energies;
  std::map<double, int> translation_map_energy_index;
  std::vector<vector<double> > all_kvectors;
  std::map<vector<double>, set<double> > all_energies_per_kvector;
  std::map<vector<double>, int > translation_map_kvector_index;
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
        double temp_energy = PropagationUtilities::read_energy_from_momentum(this,*c_it, writeable_Propagator);
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
  std::map<std::vector<NemoMeshPoint>,std::vector<std::complex<double> > *>::const_iterator c_it=input_data->begin();

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
        double temp_double=find_integration_weight(momentum_mesh_name, c_it->first, writeable_Propagator);

        energy_resolved_integration_weight*=temp_double;
      }
    }

    //do the integration and store in result
    {
      double energy = PropagationUtilities::read_energy_from_momentum(this,c_it->first,writeable_Propagator);
      //std::map<double,std::vector<std::complex<double> > >* energy_resolved_data=Propagator_pointer->get_energy_resolved_integrated_diagonal();
      //allocate the energy_resolved_data map
      if(result->size()!=all_energies.size())
      {
        result->clear();
        for(unsigned int i=0; i<all_energies.size(); i++)
          (*result)[all_energies[i]]=std::vector<std::complex<double> >(data_size,0.0);
      }
      std::map<double,std::vector<std::complex<double> > >::iterator edata_it=result->find(energy);
      if(edata_it==result->end())
      {
        (*result)[energy]=std::vector<std::complex<double> >(data_size,0.0);
        edata_it=result->find(energy);
      }
      NEMO_ASSERT(edata_it!=result->end(),prefix+"have not found energy entry in the energy resolved data map\n");
      for(unsigned int i=0; i<data_size; i++)
        (edata_it->second)[i]+=(*c_it->second)[i]*energy_resolved_integration_weight;
    }

  }

  //std::map<double,std::vector<std::complex<double> > >* energy_resolved_data=Propagator_pointer->get_energy_resolved_integrated_diagonal();
  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  std::map<double,std::vector<std::complex<double> > >::iterator edata_it=result->begin();
  //perform the sum over MPI data for a single energy
  for(; edata_it!=result->end(); ++edata_it)
  {
    std::vector<std::complex<double> > send_vector=edata_it->second;
    if(!solve_on_single_replica)
      MPI_Allreduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);
    else
      MPI_Reduce(&(send_vector[0]),&(edata_it->second[0]),send_vector.size(), MPI_DOUBLE_COMPLEX, MPI_SUM , 0, topcomm);
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

void Propagation::integrate_scalar(const std::map<std::vector<NemoMeshPoint>,double >& input_data, double& result)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("Propagation(\""+tic_toc_name+"\")::integrate_scalar ");
  NemoUtils::tic(tic_toc_prefix);
  result=0;
  if(Mesh_tree_topdown.size()==0)
  {
    //get the Mesh_tree_names:
    Parallelizer->get_data("mesh_tree_names",Mesh_tree_names);
    //get the Mesh_tree_downtop
    Parallelizer->get_data("mesh_tree_downtop",Mesh_tree_downtop);
    //get the Mesh_tree_topdown
    Parallelizer->get_data("mesh_tree_topdown",Mesh_tree_topdown);
  }
  NEMO_ASSERT(Mesh_tree_topdown.size()>0,tic_toc_prefix+"Mesh_tree_topdown is not ready for usage\n");
  //1.loop over incoming data
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
      double temp_double=find_integration_weight(momentum_mesh_name, c_it->first, writeable_Propagator);
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

void Propagation::get_Propagator(Propagator*& output_propagator, const NemoPhys::Propagator_type* )
{
  if(writeable_Propagator==NULL)
    initialize_Propagation();
  NEMO_ASSERT(writeable_Propagator!=NULL,"Propagation(\""+get_name()+"\")::get_Propagator writeable_Propagator=NULL\n");
  output_propagator=writeable_Propagator;
}

void Propagation::get_Propagator(const Propagator*& output_propagator, const NemoPhys::Propagator_type* type)
{
  Propagator* temp_propagator=NULL;
  get_Propagator(temp_propagator,type);
  output_propagator=temp_propagator;
}

void Propagation::symmetrize(PetscMatrixParallelComplex*& input_matrix,NemoMath::symmetry_type& symm_type)
{
  if(symm_type==NemoMath::antihermitian)
  {
    PetscMatrixParallelComplex temp_result=PetscMatrixParallelComplex(input_matrix->get_num_cols(),
      input_matrix->get_num_rows(),
      input_matrix->get_communicator());
    input_matrix->hermitian_transpose_matrix(temp_result,MAT_INITIAL_MATRIX);
    input_matrix->add_matrix(temp_result, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));
    *input_matrix*=std::complex<double>(0.5,0.0);
  }
  else
    throw std::invalid_argument("Propagation(\""+get_name()+"\")::symmetrize no implemented for this symmetry\n");
}

void Propagation::get_device_Hamiltonian(const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& Hamiltonian, EOMMatrixInterface*& eom_interface)
{
  NEMO_ASSERT(Hamilton_Constructor!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::get_device_Hamiltonian Hamilton_Constructor is NULL\n");
  eom_interface = dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
  NEMO_ASSERT(eom_interface!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::get_device_Hamiltonian "+Hamilton_Constructor->get_name()+" is not an EOMMatrixInterface\n");
  //std::vector<NemoMeshPoint> temp_vector;
  std::vector<NemoMeshPoint> sorted_momentum_point;
  QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum_point,options,momentum_mesh_types,Hamilton_Constructor);
  DOFmapInterface* temp_pointer=NULL;
  eom_interface->get_EOMMatrix(sorted_momentum_point, NULL, NULL, false, Hamiltonian,temp_pointer);
}

void Propagation::get_device_Overlap(const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& Overlap, EOMMatrixInterface*& eom_interface)
{
  NEMO_ASSERT(Hamilton_Constructor!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::get_device_Overlap  Hamilton_Constructor is NULL\n");
  eom_interface = dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
  NEMO_ASSERT(eom_interface!=NULL,"ForwardRGFSolver(\""+get_name()+"\")::get_device_Overlap "+Hamilton_Constructor->get_name()+" is not an EOMMatrixInterface\n");
  //std::vector<NemoMeshPoint> temp_vector;
  std::vector<NemoMeshPoint> sorted_momentum_point;
  QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum_point,options,momentum_mesh_types,Hamilton_Constructor);
  DOFmapInterface* temp_pointer=NULL;
  eom_interface->get_overlap_matrix(sorted_momentum_point, NULL, NULL, Overlap, temp_pointer);
}

void Propagation::get_contact_sigma(const std::vector<NemoMeshPoint>& momentum, EOMMatrixInterface * eom_interface, PetscMatrixParallelComplex*& result,
                    const std::string& target_domain_name, const Domain* lead_domain, PetscMatrixParallelComplex* Hamiltonian,
                    PetscMatrixParallelComplex* gR, PetscMatrixParallelComplex* coupling,
                    PetscMatrixParallelComplex* sigmaR_scattering, NemoPhys::Propagator_type type)
{
  //2.1 get the contact suface Green's function
  //PetscMatrixParallelComplex* gR=NULL;
  PetscMatrixParallelComplex* temp_result=NULL;
  if(gR==NULL)
  {
    if(forward_gR_solver==NULL)
    {
      std::string temp_name=options.get_option("forward_RGF_solver",std::string(""));
      forward_gR_solver=find_simulation(temp_name);
      NEMO_ASSERT(forward_gR_solver!=NULL,"Propagation(\""+get_name()+"\")::get_contact_sigma have not found \""+temp_name+"\"\n");
    }
    GreensfunctionInterface* temp_gR_interface=dynamic_cast<GreensfunctionInterface*>(forward_gR_solver);
    NEMO_ASSERT(temp_gR_interface!=NULL,"Propagation(\""+get_name()+"\")::get_contact_sigma \""+forward_gR_solver->get_name()+"\" is no GreensfunctionInterface\n");

    if(particle_type_is_Fermion)
    {
      if(lead_domain != NULL && forward_gR_solver->get_type().find("RGF")!=std::string::npos)
      {
        DOFmapInterface& row_dofmap = forward_gR_solver->get_dof_map(lead_domain);
        temp_gR_interface->get_Greensfunction(momentum, gR, &row_dofmap, NULL, type);
      }
      else
      {
        temp_gR_interface->get_Greensfunction(momentum,gR,NULL, NULL, type);
      }
    }
    else
    {
      //deal with default arguments for Bosons
      if(type==NemoPhys::Fermion_retarded_Green)
        type = NemoPhys::Boson_retarded_Green;
      temp_gR_interface->get_Greensfunction(momentum,gR,NULL, NULL, type);
    }
  }
  if(debug_output)
    gR->save_to_matlab_file("GR.m");

  if(coupling==NULL)
  {
    //2.2 get the contact/device Hamiltonian
    //2.2.1 get the supermatrix
    PetscMatrixParallelComplex* device_lead_H=NULL;
    PetscMatrixParallelComplex* device_lead_S=NULL;
    Domain* first_subdomain=Domain::get_domain(target_domain_name);
    if(lead_domain==NULL)
    {

      std::string temp_domain_name="";
      forward_gR_solver->get_data("result_domain_name",temp_domain_name);
      std::vector<std::string> start_leads(2,temp_domain_name);
      start_leads[1]=forward_gR_solver->get_const_simulation_domain()->get_name();
      lead_domain=Domain::get_domain(start_leads[0]);
      NEMO_ASSERT(lead_domain!=NULL,"Propagation(\""+get_name()+"\")::get_contact_sigma did not find domain \""+temp_domain_name+"\"\n");

    }

    std::vector<NemoMeshPoint> temp_vector;
    QuantumNumberUtils::sort_quantum_number(momentum,temp_vector,options,get_momentum_mesh_types(),this);

    DOFmapInterface& coupling_DOFmap = forward_gR_solver->get_dof_map(lead_domain);
    DOFmapInterface* temp_pointer=&coupling_DOFmap;
    eom_interface->get_EOMMatrix(temp_vector, first_subdomain,lead_domain,true,device_lead_H,temp_pointer);
    eom_interface->get_overlap_matrix(temp_vector, first_subdomain, lead_domain , device_lead_S,temp_pointer);

    bool include_sigma_scattering = false;
    if(sigmaR_scattering!=NULL)
      include_sigma_scattering = true;

    //2.2.2 get sub block
    unsigned int start_row = 0;
    unsigned int end_row = 0;
    unsigned int nonzero_index = 0;
    PetscMatrixParallelComplexContainer* temp_container=dynamic_cast<PetscMatrixParallelComplexContainer*>(Hamiltonian);
    NEMO_ASSERT(temp_container!=NULL, "Propagation(\""+get_name()+"\")::run_forward_RGF_for_momentum did not receive container for Hamiltonian\n");

    Simulation* temp_simulation=dynamic_cast<Simulation*>(eom_interface);

    unsigned int number_of_device_slab_rows = 0;
    unsigned int number_of_device_slab_cols = 0;
    unsigned int number_of_lead_cols = 0;
    unsigned int number_of_couple_rows = 0;
    std::vector<int> temp_rows;
    std::vector<int> temp_cols;

    // DL - If the matrix is received by LRAModule, it should contain only necessary data and be dense, as opposed to only having a section of nonzero rows and columns
    bool extract_coupling = device_lead_H->if_sparse();
    if (extract_coupling)  // Matrix is sparse and uneccesary rows and columns should be discarded
    {
      number_of_device_slab_rows = temp_simulation->get_const_dof_map(first_subdomain).get_number_of_dofs();  //H_blocks.begin()->second->get_num_rows();
      number_of_device_slab_cols = number_of_device_slab_rows;

      number_of_lead_cols = device_lead_H->get_num_cols()-number_of_device_slab_cols;

      for(unsigned int i=0; i<number_of_device_slab_rows; i++)
      {
        nonzero_index = device_lead_H->get_nz_diagonal(i);
        if(nonzero_index>0)
        {
          start_row = i;
          break;
        }
      }
      nonzero_index = 0;
      for(unsigned int i=number_of_device_slab_rows; i>0; i--)
      {
        nonzero_index = device_lead_H->get_nz_diagonal(i-1);
        if(nonzero_index>0)
        {
          end_row = i-1;
          break;
        }
      }
      number_of_couple_rows = end_row + 1 - start_row;

      temp_rows.resize(number_of_couple_rows);
      temp_cols.resize(number_of_lead_cols);
      for(unsigned int i=0; i<number_of_couple_rows; i++)
        temp_rows[i]=i+start_row;
      for(unsigned int i=0; i<number_of_lead_cols; i++)
        temp_cols[i]=i+number_of_device_slab_cols;
    }
    else  // The matrix is to be taken and used as-is
    {
      number_of_device_slab_rows = device_lead_H->get_num_rows();
      number_of_device_slab_cols = number_of_device_slab_rows;

      number_of_lead_cols = device_lead_H->get_num_cols();
      number_of_couple_rows = device_lead_H->get_num_rows();

      temp_rows.resize(number_of_couple_rows);
      temp_cols.resize(number_of_lead_cols);
      for(unsigned int i=0; i<number_of_couple_rows; i++)
        temp_rows[i]=i;
      for(unsigned int i=0; i<number_of_lead_cols; i++)
        temp_cols[i]=i;
    }

    PetscMatrixParallelComplex* coupling = NULL;
    PetscMatrixParallelComplex* S_matrix = NULL;

    coupling = new PetscMatrixParallelComplex(number_of_couple_rows, number_of_lead_cols, get_simulation_domain()->get_communicator());
    coupling->set_num_owned_rows(number_of_couple_rows);

    vector<int> rows_diagonal(number_of_couple_rows,0);
    vector<int> rows_offdiagonal(number_of_couple_rows,0);
    unsigned int number_of_nonzero_cols_local=0;
    unsigned int number_of_nonzero_cols_nonlocal=0;
    if(!include_sigma_scattering)
    {
      // PropagationUtilities::extract_coupling_Hamiltonian_RGF(this, number_of_device_slab_rows, number_of_device_slab_cols, device_lead_H,
      //   coupling);

      if (extract_coupling)
      {
        for (unsigned int i = 0; i<number_of_couple_rows; i++)
        {
          rows_diagonal[i] = device_lead_H->get_nz_diagonal(i + start_row);
          rows_offdiagonal[i] = device_lead_H->get_nz_offdiagonal(i + start_row);
          if (rows_diagonal[i]>0)
            number_of_nonzero_cols_local++;
          if (rows_offdiagonal[i] > 0)
            number_of_nonzero_cols_nonlocal++;
        }
        //setup sparsity pattern of the coupling Hamiltonian
        for(unsigned int i = 0; i < number_of_couple_rows; i++)
          coupling->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);

      }
      else
        coupling->consider_as_full();
      device_lead_H->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, coupling);

      coupling->assemble();
      //coupling->save_to_matlab_file("HcS.m");
      if(device_lead_S!=NULL)
      {
         S_matrix = new PetscMatrixParallelComplex(number_of_couple_rows,number_of_lead_cols, get_simulation_domain()->get_communicator());
         S_matrix->set_num_owned_rows(number_of_couple_rows);
         for(unsigned int i = 0; i < number_of_couple_rows; i++)
           S_matrix->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);
         device_lead_S->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, S_matrix);
         S_matrix->assemble();
         //S_matrix->save_to_matlab_file("ScS.m");
         double energy = PropagationUtilities::read_energy_from_momentum(this,momentum,writeable_Propagator);
         *S_matrix *= std::complex<double> (energy,0.0); //ES
         coupling->add_matrix(*S_matrix,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));//H-ES
         *coupling *= std::complex<double>(-1.0,0.0); //ES-H
         delete S_matrix;
         S_matrix=NULL;
       }

      if(debug_output)
      {
        coupling->save_to_matlab_file("small_DLH.m");
        device_lead_H->save_to_matlab_file("DLH.m");
        gR->save_to_matlab_file("GR.m");
      }
      //std::string temp_domain_name = target_domain_name;
      //device_lead_H->save_to_matlab_file("full_coupling_"+get_name()+"_"+temp_domain_name+".m");
      delete device_lead_H;
      device_lead_H=NULL;//NOTE: we might want to store the device_lead_H for other E,k

      //coupling->save_to_matlab_file("coupling_"+get_name()+"_"+temp_domain_name+".m");
      //gR->save_to_matlab_file("GR_"+get_name()+"_"+temp_domain_name+".m");
      //2.3 multiply 2.1 and 2.2
      PetscMatrixParallelComplex* temp_matrix=NULL;

      PetscMatrixParallelComplex::mult(*coupling,*gR,&temp_matrix);

      {
        PetscMatrixParallelComplex::mult_hermitian_transpose(*temp_matrix,*coupling,&temp_result);
        delete coupling;
        coupling=NULL;

        //PetscMatrixParallelComplex transpose_coupling=PetscMatrixParallelComplex(coupling->get_num_cols(),coupling->get_num_rows(),coupling->get_communicator());
        //coupling->hermitian_transpose_matrix(transpose_coupling,MAT_INITIAL_MATRIX);
        //PetscMatrixParallelComplex::mult(*temp_matrix,transpose_coupling,&temp_result);
        //delete coupling;
        //coupling=NULL;
      }
      if(debug_output)
        temp_result->save_to_matlab_file("temp_sigma.m");
      delete temp_matrix;
      temp_matrix=NULL;
    }
    else //include sigma scattering
    {
      std::vector<int> temp_rows(number_of_couple_rows);
      std::vector<int> temp_cols(number_of_lead_cols);
      for (unsigned int i = 0; i<number_of_couple_rows; i++)
        temp_rows[i] = i;
      if (extract_coupling)  // Need to shift columns
      {
        for (unsigned int i = 0; i<number_of_lead_cols; i++)
          temp_cols[i] = i + number_of_device_slab_cols;
      }
      else  // Don't need to shift columns
      {
        for (unsigned int i = 0; i<number_of_lead_cols; i++)
          temp_cols[i] = i;
      }
      delete coupling;
      //PetscMatrixParallelComplex* coupling = NULL;
      coupling = new PetscMatrixParallelComplex(number_of_couple_rows, number_of_lead_cols, get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);

      coupling->set_num_owned_rows(number_of_couple_rows);
      rows_diagonal.resize(number_of_couple_rows);
      rows_offdiagonal.resize(number_of_couple_rows);

      for (unsigned int i = 0; i<number_of_couple_rows; i++)
      {
        rows_diagonal[i] = std::max(device_lead_H->get_nz_diagonal(i),sigmaR_scattering->get_nz_diagonal(i));
        rows_offdiagonal[i] = std::max(device_lead_H->get_nz_offdiagonal(i),sigmaR_scattering->get_nz_offdiagonal(i));

        if (rows_diagonal[i]>0)
          number_of_nonzero_cols_local++;
        if (rows_offdiagonal[i] > 0)
          number_of_nonzero_cols_nonlocal++;
      }

      // Set sparsity pattern
      if (extract_coupling)
      {
        for (unsigned int i = 0; i<number_of_couple_rows; i++)
          coupling->set_num_nonzeros_for_local_row(i, rows_diagonal[i], rows_offdiagonal[i]);
      }
      else
        coupling->consider_as_full();

      device_lead_H->get_submatrix(temp_rows, temp_cols, MAT_INITIAL_MATRIX, coupling);

      coupling->assemble();
      start_row = 0;

      PetscMatrixParallelComplex transpose_coupling(coupling->get_num_cols(), coupling->get_num_rows(),
          coupling->get_communicator());

      if(type == NemoPhys::Fermion_lesser_Green || type == NemoPhys::Boson_lesser_Green)
      {
        coupling->add_matrix(*sigmaR_scattering,DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
        coupling->hermitian_transpose_matrix(transpose_coupling, MAT_INITIAL_MATRIX);
      }
      else
      {
        coupling->hermitian_transpose_matrix(transpose_coupling, MAT_INITIAL_MATRIX);
        coupling->add_matrix(*sigmaR_scattering,DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
        transpose_coupling.add_matrix(*sigmaR_scattering,DIFFERENT_NONZERO_PATTERN,std::complex<double>(1.0,0.0));
      }

      if(gR->if_container())
        gR->assemble();

      PetscMatrixParallelComplex* temp_matrix=NULL;
      PetscMatrixParallelComplex::mult(*coupling,*gR,&temp_matrix);

      PropagationUtilities::supress_noise(this,temp_matrix);
      //temp_matrix1->save_to_matlab_file("HxGR.m");
      delete coupling;
      coupling=NULL;
      PropagationUtilities::supress_noise(this,&transpose_coupling);
      PetscMatrixParallelComplex::mult(*temp_matrix,transpose_coupling,&temp_result);

      PropagationUtilities::supress_noise(this,temp_result);

      delete temp_matrix;
      temp_matrix=NULL;
    }

    if (extract_coupling)  // Resulting matrix is sparse
    {
      //3. copy result into a sparse matrix
      vector<int> result_diagonal(number_of_device_slab_rows,0);
      vector<int> result_offdiagonal(number_of_device_slab_rows,0);
      for(unsigned int i=start_row; i<=end_row; i++)
      {
        if(rows_diagonal[i-start_row]>0)
          result_diagonal[i]=number_of_nonzero_cols_local;
        if(rows_offdiagonal[i-start_row]>0)
          result_offdiagonal[i]=number_of_nonzero_cols_nonlocal;
      }
      PropagationUtilities::transfer_matrix_initialize_temp_matrix(this,number_of_device_slab_rows,number_of_device_slab_rows,result_diagonal,result_offdiagonal,result);
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
      delete temp_result;
      temp_result=NULL;
    }
    else  // Resulting matrix is dense
    {
      result = temp_result;
    }
  }
  else
  {
    if(debug_output)
      coupling->save_to_matlab_file("small_DLH.m");

    PetscMatrixParallelComplex* temp_matrix=NULL;
    PetscMatrixParallelComplex::mult(*gR,*coupling,&temp_matrix);
    {
      PetscMatrixParallelComplex transpose_coupling=PetscMatrixParallelComplex(coupling->get_num_cols(),coupling->get_num_rows(),coupling->get_communicator());
      coupling->hermitian_transpose_matrix(transpose_coupling,MAT_INITIAL_MATRIX);
      delete result;
      result=NULL;
      PetscMatrixParallelComplex::mult(transpose_coupling,*temp_matrix,&result);
      delete temp_matrix;
    }
  }
  if(debug_output)
    result->save_to_matlab_file("sigma.m");
}

void Propagation::get_contact_sigma_lesser_equilibrium(const std::vector<NemoMeshPoint>& momentum, EOMMatrixInterface * eom_interface, PetscMatrixParallelComplex*& result,
    const std::string& target_domain_name,const Domain* lead_domain, const std::vector<int>& ,
    PetscMatrixParallelComplex* Hamiltonian, PetscMatrixParallelComplex* gR,
    PetscMatrixParallelComplex* coupling, PetscMatrixParallelComplex* sigmaR_scattering, NemoPhys::Propagator_type type)
{
  std::string prefix = get_name() + "get_constact_sigma_lesser_equilibrium";
  //1. get equilibrium g< for the contact
  bool differentiate_over_energy = options.get_option("differentiate_over_energy",false);
  double energy = PropagationUtilities::read_energy_from_momentum(this,momentum,writeable_Propagator);
  bool lead_electron_like = true;
  if(energy<threshold_energy_for_lead1)
    lead_electron_like = false;

  //NEMO_ASSERT(!lead_electron_like,"why is this not an electron");
  //core_lesser_equilibrium(energy,chemical_potential, temperature1, temp_retarded, temp_advanced, result, particle_type_is_Fermion,differentiate_over_energy,
  //      lead_electron_like, electron_hole_model, averaged_hole_factor_pointer, momentum_type, electron_mass, hole_mass, periodic_space_volume);

  std::vector<double>* averaged_hole_factor_pointer = NULL;
  //std::vector<double> averaged_hole_factor;
  /*if(electron_hole_model)
  {
    Hamilton_Constructor->get_data(std::string("averaged_hole_factor"),energy,averaged_hole_factor);
    averaged_hole_factor_pointer = &averaged_hole_factor;
  }*/
  std::string momentum_type=options.get_option("analytical_momenta",std::string(""));
  double periodic_space_volume = get_const_simulation_domain()->return_periodic_space_volume();
  double analytical_momenta_me = options.get_option("analytical_momenta_me",-1.0);
  double analytical_momenta_mh = options.get_option("analytical_momenta_mh",-1.0);

  double electron_mass = -1.0;
  double hole_mass = -1.0;
  //this has to be done here not in the core
  if(momentum_type!="")
  {
    std::string temp_simulation_name=options.get_option("effective_mass_solver",Hamilton_Constructor->get_name());
    Simulation* temp_simulation=find_simulation(temp_simulation_name);
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

  PetscMatrixParallelComplex* gA=new PetscMatrixParallelComplex(gR->get_num_cols(),
      gR->get_num_rows(),
      gR->get_communicator());

  if(gR->if_container())
    gR->assemble();
  gR->hermitian_transpose_matrix(*gA,MAT_INITIAL_MATRIX);

  PetscMatrixParallelComplex* gL = NULL;
  PropagationUtilities::core_lesser_equilibrium(energy,chemical_potential,temperature1,gR,gA,gL,particle_type_is_Fermion,differentiate_over_energy,
      lead_electron_like,false, averaged_hole_factor_pointer, momentum_type, electron_mass, hole_mass, periodic_space_volume);

  //delete gR;
  //gR = NULL;
  delete gA;
  gA = NULL;

  //2. get equilibrium sigma< for the contact
  get_contact_sigma(momentum,eom_interface, result, target_domain_name, lead_domain,
      Hamiltonian, gL, coupling, sigmaR_scattering,type);
  delete gL;
  gL = NULL;

  NemoMath::symmetry_type symmetry = NemoMath::antihermitian;
  symmetrize(result,symmetry);

}


void Propagation::calculate_slab_resolved_current()
{
  std::string tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "PropagationUtilities(\""+tic_toc_name+"\")::calculate_slab_resolved_current ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "PropagationUtilities(\""+get_name()+"\")::calculate_slab_resolved_currents ";


  //1. get the Hamilton Constructor (not necessarily the Hamilton_Constructor for the rest of the calculation
  //assumed to give all of the Schroedingers for each subdomain.
  std::string EOM_constructor_name = options.get_option("Hamilton_constructor_for_current",Hamilton_Constructor->get_name());
  Simulation* EOM_Constructor = find_simulation(EOM_constructor_name);
  NEMO_ASSERT(EOM_Constructor!=NULL,prefix + "could not find EOM Constructor");


  //2. get the G solver
  std::string name_of_propagator;
  if(writeable_Propagator!=NULL)
    name_of_propagator=name_of_writeable_Propagator;
  if(options.check_option("density_of"))
    name_of_propagator=options.get_option("density_of",std::string(""));
  Simulation* source_of_G = find_source_of_data(name_of_propagator);
  NEMO_ASSERT(source_of_G!=NULL, prefix + "could not find source of Green's function");


  //3. do the actual work.
  vector<double> slab_resolved_current;
  vector< map<double, double> > slab_energy_resolved_current;
  PropagationUtilities::calculate_slab_resolved_current(this, name_of_propagator, EOM_Constructor,
      source_of_G, slab_resolved_current, slab_energy_resolved_current);

  //4. output
  if (holder.geometry_replica==0)
  {
    if (holder.geometry_rank==0)
    {
      std::ofstream output_file;
      output_file.precision(21);
      std:: string suffix = get_output_suffix();
      std::string output_file_name = get_name() + "_slab_resolved_current_" + suffix +  ".dat";

      output_file.open(output_file_name.c_str());
      //output_file_atom_resolved.open(output_file_name_atom_resolved.c_str());
      output_file << "% index  current(A) \n";

      for(unsigned int i = 0; i < slab_resolved_current.size(); i++)
      {
        output_file << i << " " << slab_resolved_current[i] << " \n";

      }
      output_file.close();

      bool energy_resolved_current = options.get_option("energy_resolved_current",false);
      if(energy_resolved_current)
      {
        std::ofstream output_file;
        std:: string suffix = get_output_suffix();
        std::string output_file_name = get_name() + "_slab_resolved_energy_resolved_current_" + suffix +  ".dat";
        output_file.open(output_file_name.c_str());
        //vector<vector<double> > slab_energy_resolved_current;
        output_file << "% index \t";

        map<double,double>::iterator it = slab_energy_resolved_current[0].begin();
        for(; it != slab_energy_resolved_current[0].end(); ++it)
          output_file << it->first << "[eV] \t";

        output_file << " \n";

        for(unsigned int i = 0; i < slab_energy_resolved_current.size(); i++)
        {
          output_file << i << " \t";
          map<double,double>::iterator it = slab_energy_resolved_current[i].begin();
          for(; it != slab_energy_resolved_current[i].end(); ++it)
            output_file << it->second << " \t";
          output_file << "\n";
        }
      }
    }
  }



  NemoUtils::toc(tic_toc_prefix);

}

void Propagation::extract_spin_diagonal(PetscMatrixParallelComplex*& input_matrix, std::vector<std::complex<double> >& result, NemoPhys::Spin_element_type spin_element)
{
  //1. allocate result
  result.clear();
  result.resize(input_matrix->get_num_rows()/2,std::complex<double>(0.0,0.0));
  //2. get the number of orbitals
  double number_of_orbitals=0;
  Hamilton_Constructor->get_data("number_of_orbitals",number_of_orbitals);
  number_of_orbitals/=2.0;
  //3. interpret spin_element
  NEMO_ASSERT(spin_element!=NemoPhys::Nospin, "Propagation(\""+get_name()+"\")::extract_spin_diagonal called with Nospin input setting\n");
  unsigned int row_shift=0;
  unsigned int column_shift=0;
  if(spin_element==NemoPhys::UpDown)
    column_shift=number_of_orbitals;
  else if(spin_element==NemoPhys::DownDown)
  {
    column_shift=number_of_orbitals;
    row_shift=number_of_orbitals;
  }
  else if(spin_element==NemoPhys::DownUp)
    row_shift=number_of_orbitals;
  //4. fill result
  for(unsigned int i=0;i<input_matrix->get_num_rows()/number_of_orbitals/2.0;i++) //loop over atoms
    for(unsigned int j=0;j<number_of_orbitals;j++) //loop over orbitals
      result[i*number_of_orbitals+j]=input_matrix->get(2*i*number_of_orbitals+j+row_shift,2*i*number_of_orbitals+j+column_shift);
}

void Propagation::solve_spin_polarization(Simulation* source_of_data, NemoPhys::Propagator_type input_type)
{
  std::string error_prefix="Propagation(\""+get_name()+"\")::solve_spin_polarization ";
  //1. get the integrated spin up, down, updown and downup densities from integrate_diagonal
  std::string name_of_propagator=options.get_option("density_of",std::string(""));
  NEMO_ASSERT(options.check_option("density_of"),error_prefix+"please define \"density_of\"");
  PropagatorInterface* temp_interface=dynamic_cast<PropagatorInterface*>(source_of_data);
  NEMO_ASSERT(temp_interface!=NULL,error_prefix+source_of_data->get_name()+" is not a PropagatorInterface\n");
  Propagator* lesser_Green=NULL;
  //1.1 get upup
  bool temp_bool=false;
  integrate_diagonal(source_of_data, input_type, name_of_propagator, temp_bool,temp_bool, temp_bool,temp_bool,NemoPhys::UpUp);
  source_of_data->get_data(name_of_propagator,lesser_Green);
  std::vector<std::complex<double> > temp_upup(*(lesser_Green->get_readable_integrated_diagonal()));
  //1.2 get updown
  temp_bool=false;
  integrate_diagonal(source_of_data, input_type, name_of_propagator, temp_bool,temp_bool, temp_bool, temp_bool,NemoPhys::UpDown);
  source_of_data->get_data(name_of_propagator,lesser_Green);
  std::vector<std::complex<double> > temp_updown(*(lesser_Green->get_readable_integrated_diagonal()));
  //1.3 get downup
  temp_bool=false;
  integrate_diagonal(source_of_data, input_type, name_of_propagator, temp_bool,temp_bool, temp_bool, temp_bool,NemoPhys::DownUp);
  source_of_data->get_data(name_of_propagator,lesser_Green);
  std::vector<std::complex<double> > temp_downup(*(lesser_Green->get_readable_integrated_diagonal()));
  //1.4 get downdown
  temp_bool=false;
  integrate_diagonal(source_of_data, input_type, name_of_propagator, temp_bool,temp_bool,temp_bool, temp_bool, NemoPhys::DownDown);
  source_of_data->get_data(name_of_propagator,lesser_Green);
  std::vector<std::complex<double> > temp_downdown(*(lesser_Green->get_readable_integrated_diagonal()));

  std::vector<double> total_density=std::vector<double>(temp_upup.size(),0.0);
  for(unsigned int i=0;i<total_density.size();i++)
    total_density[i]+=std::imag(temp_upup[i])+std::imag(temp_downdown[i]);

  //2. combine the spin densities and correlations to get spin polarization in many space directions
  std::vector<std::complex<double> > z_polarization=std::vector<std::complex<double> >(temp_upup.size(),std::complex<double> (0.0,0.0));
  std::vector<std::complex<double> > y_polarization=std::vector<std::complex<double> >(temp_upup.size(),std::complex<double> (0.0,0.0));
  std::vector<std::complex<double> > x_polarization=std::vector<std::complex<double> >(temp_upup.size(),std::complex<double> (0.0,0.0));
  for(unsigned int i=0;i<total_density.size();i++)
  {
    if(std::abs(total_density[i])>NemoMath::d_zero_tolerance)
    {
      //2.1 Spin polarization in z-direction
      z_polarization[i]+=std::complex<double> ((std::imag(temp_upup[i])-std::imag(temp_downdown[i]))/total_density[i],0.0);
      //2.2 Spin polarization in x-direction
      x_polarization[i]+=std::complex<double> ((std::imag(temp_updown[i])+std::imag(temp_downup[i]))/total_density[i]);
      //2.3 Spin polarization in y-direction
      y_polarization[i]+=std::complex<double> ((std::real(temp_updown[i])-std::real(temp_downup[i]))/total_density[i]);
    }
  }
  //3. transform data into atom-spin resolved representation
  /*Simulation* spin_free_HamiltonConstructor=NULL;
  std::string spin_free_HamiltonConstructor_name=options.get_option("spin_free_HamiltonConstructor",std::string(""));
  spin_free_HamiltonConstructor=find_simulation(spin_free_HamiltonConstructor_name);
  NEMO_ASSERT(spin_free_HamiltonConstructor!=NULL,error_prefix+"have not found simulation \""+spin_free_HamiltonConstructor_name+"\" for spin_free_HamiltonConstructor\n");*/
  NEMO_ASSERT(nospin_Hamilton_Constructor!=NULL, error_prefix + "nospin_Hamilton_Constructor is NULL\n");
  std::vector<std::complex<double> > small_z_polarization;
  std::vector<std::complex<double> > small_y_polarization;
  std::vector<std::complex<double> > small_x_polarization;
  TransformationUtilities::transform_vector_orbital_to_atom_resolved(nospin_Hamilton_Constructor, std::complex<double>(1.0,0.0), z_polarization,
                                                        small_z_polarization, false);
  TransformationUtilities::transform_vector_orbital_to_atom_resolved(nospin_Hamilton_Constructor, std::complex<double>(1.0,0.0), y_polarization,
                                                        small_y_polarization, false);
  TransformationUtilities::transform_vector_orbital_to_atom_resolved(nospin_Hamilton_Constructor, std::complex<double>(1.0,0.0), x_polarization,
                                                        small_x_polarization, false);

  //4. write the data to disk 
  std::map<unsigned int,double> z_pol_map;
  store_results_in_result_map(source_of_data, name_of_propagator,lesser_Green,std::complex<double>(1.0,0.0),1.0,small_z_polarization,&z_pol_map);
  std::map<unsigned int,double> y_pol_map;
  store_results_in_result_map(source_of_data, name_of_propagator,lesser_Green,std::complex<double>(1.0,0.0),1.0,small_y_polarization,&y_pol_map);
  std::map<unsigned int,double> x_pol_map;
  store_results_in_result_map(source_of_data, name_of_propagator,lesser_Green,std::complex<double>(1.0,0.0),1.0,small_x_polarization,&x_pol_map);

  const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  int rank;
  MPI_Comm_rank(topcomm,&rank);
  if(rank==0)
  {
    std::string file_describe="Spin polarization in z-direction (unitless)";
    std::string filename=get_name()+"_z_polarization";
    print_atomic_map(z_pol_map,filename, file_describe);
    file_describe="Spin polarization in y-direction (unitless)";
    filename=get_name()+"_y_polarization";
    print_atomic_map(y_pol_map,filename, file_describe);
    file_describe="Spin polarization in x-direction (unitless)";
    filename=get_name()+"_x_polarization";
    print_atomic_map(x_pol_map,filename, file_describe);
  }
}

void Propagation::store_results_in_result_map(Simulation* source_of_matrices,  const std::string& name_of_propagator, Propagator* lesser_Green,
                                              const std::complex<double>& complex_prefactor, const double prefactor, 
                                              std::vector<std::complex<double> >& input,std::map<unsigned int,double>* result)
{
  std::string error_prefix="Propagation(\""+get_name()+"\")::store_results_in_result_map ";
  //1. store the results in the result map
  //if the DOFmap size does not agree with the size of vector, we ask the matrix source how to translate the vector index into the atom_id
  //otherwise, we assume that there is a one to one correspondence
  translate_vector_into_map(input, complex_prefactor*prefactor, true,*result);

  //2. output the energy resolved density
  bool get_energy_resolved_data=options.get_option(name_of_propagator+"_energy_resolved_output",false) ||
                                  options.get_option(name_of_propagator + "_energy_resolved",bool(false)) || options.get_option("energy_resolved_density",false);
  bool get_energy_resolved_nonrectangular_data = false;
  if(options.get_option("energy_resolved_density",false))
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
  if(get_energy_resolved_data)
  {
    energy_resolved_density_for_output.clear();
    complex_energy_resolved_density_for_output.clear();
    //8.1 check whether this MPI rank is the one where the data had been reduced to
    NEMO_ASSERT(Mesh_tree_topdown.size()>0,error_prefix+"Mesh_tree_topdown is not ready for usage\n");
    const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
    int my_rank;
    MPI_Comm_rank(topcomm,&my_rank);
    {
      double real_prefactor=options.get_option(name_of_propagator+"_energy_resolved_output_real",1.0);
      double imag_prefactor=options.get_option(name_of_propagator+"_energy_resolved_output_imag",0.0);
      std::complex<double> total_input_prefactor(real_prefactor,imag_prefactor);
      //8.2 get access to the energy resolved data
      source_of_matrices->get_data(name_of_propagator,lesser_Green);

      if(complex_energy_used())
      {
        std::map<std::complex<double>,std::vector<std::complex<double> >, Compare_double_or_complex_number> temp_density(*(lesser_Green->get_complex_energy_resolved_integrated_diagonal()));
        std::map<std::complex<double>,std::vector<std::complex<double> >, Compare_double_or_complex_number>::iterator it=temp_density.begin();
        for(; it!=temp_density.end(); ++it)
        {
          std::map<unsigned int, double> temp_map;
          translate_vector_into_map(it->second, total_input_prefactor*prefactor*std::complex<double>(0.0,-1.0), true,temp_map);
          complex_energy_resolved_density_for_output[it->first]=temp_map;
        }
      }
      else
      {
        std::map<double,std::vector<std::complex<double> >, Compare_double_or_complex_number > temp_density(*(lesser_Green->get_energy_resolved_integrated_diagonal()));
        std::map<double,std::vector<std::complex<double> >, Compare_double_or_complex_number >::iterator it=temp_density.begin();
        for(; it!=temp_density.end(); ++it)
        {
          std::map<unsigned int, double> temp_map;
          translate_vector_into_map(it->second, total_input_prefactor*prefactor*std::complex<double>(0.0,-1.0), true,temp_map);
          energy_resolved_density_for_output[it->first]=temp_map;
        }
      }
      if(my_rank==0&&options.get_option(name_of_propagator+"_energy_resolved_output",false))
      {
        if(options.get_option(name_of_propagator+"_energy_resolved_output_one_per_file",false))
        {
          print_atomic_maps_per_file(energy_resolved_density_for_output,name_of_propagator+"_energy_resolved.E=");
        }
        else
        {
          std::string filename = "energy_resolved_"+name_of_propagator; //Yu:this should be enough, otherwise the name will be too long
          //if(options.check_option("output_suffix"))
          //  filename += options.get_option("output_suffix",std::string(""));
          //if(options.get_option(name_of_propagator + "_energy_resolved_output",bool(false)))
            if(!options.get_option("solve_eqneq",false))
            {
              if(complex_energy_used())
              {
                print_atomic_maps(complex_energy_resolved_density_for_output,filename);
              }
              else
              {
                print_atomic_maps(energy_resolved_density_for_output,filename);
              }
            }
        }
        reset_output_counter();
      }
    }
  }
  if(get_energy_resolved_nonrectangular_data)
  {
    energy_resolved_per_k_density_for_output.clear();
    //here I don't want to output that can be done elsewhere but just prepare the data for output.
    // I will also clear Propagator diagonal as we go for memory concerns.
    NEMO_ASSERT(Mesh_tree_topdown.size()>0,error_prefix+"Mesh_tree_topdown is not ready for usage\n");
    const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
    int my_rank;
    MPI_Comm_rank(topcomm,&my_rank);
    //if(my_rank==0)
    {
      double real_prefactor=options.get_option(name_of_propagator+"_energy_resolved_output_real",1.0);
      double imag_prefactor=options.get_option(name_of_propagator+"_energy_resolved_output_imag",0.0);
      std::complex<double> total_input_prefactor(real_prefactor,imag_prefactor);

      source_of_matrices->get_data(name_of_propagator,lesser_Green);
      std::map<vector<double> ,std::map<double,std::vector<std::complex<double> > > > temp_density(*(lesser_Green->get_energy_resolved_per_k_integrated_diagonal()));
      std::map<vector<double>,std::map<double,std::vector<std::complex<double> > > >::iterator k_it=temp_density.begin();
      for(; k_it != temp_density.end(); ++k_it)
      {
        std::map<double, std::map<unsigned int, double> > temp_map2;
        std::map<double,std::vector<std::complex<double> > >::iterator e_it = k_it->second.begin();
        for(; e_it!=k_it->second.end(); ++e_it)
        {
          std::map<unsigned int, double> temp_map;
          temp_map[0] = ((e_it->second)[0]*total_input_prefactor*prefactor).real();
          temp_map2[e_it->first] = temp_map;
        }
        energy_resolved_per_k_density_for_output[k_it->first] = temp_map2;
      }
    }
  }
  //9.0 output the k_resolved density if desired
  if(options.get_option(name_of_propagator+"_k_resolved_output",false) || k_resolved_density || options.get_option("k_resolved_density",false))
  {
    k_resolved_density_for_output.clear();
    //9.1 check whether this MPI rank is the one where the data had been reduced to
    NEMO_ASSERT(Mesh_tree_topdown.size()>0,error_prefix+"Mesh_tree_topdown is not ready for usage\n");
    const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
    int my_rank;
    MPI_Comm_rank(topcomm,&my_rank);
    {
      double real_prefactor=options.get_option(name_of_propagator+"_k_resolved_output_real",1.0);
      double imag_prefactor=options.get_option(name_of_propagator+"_k_resolved_output_imag",0.0);
      std::complex<double> total_input_prefactor(real_prefactor,imag_prefactor);
      //8.2 get access to the k resolved data
      source_of_matrices->get_data(name_of_propagator,lesser_Green);
      std::map<vector<double>,std::vector<std::complex<double> > > temp_density(*(lesser_Green->get_k_resolved_integrated_diagonal()));
      std::map<vector<double>,std::vector<std::complex<double> > >::iterator it=temp_density.begin();

      //loop through all density
      for(; it!=temp_density.end(); ++it)
      {
        std::map<unsigned int, double> temp_map;
        translate_vector_into_map(it->second, total_input_prefactor*prefactor*std::complex<double>(0.0,-1.0), true,
            temp_map);
        k_resolved_density_for_output[it->first]=temp_map;
      }

      //TODO:JC output to files in the same way as energy resolved data
      if(my_rank==0 && options.get_option(name_of_propagator+"_k_resolved_output",false))
      {
        if(options.get_option(name_of_propagator+"_k_resolved_output_one_per_file",false))
        {
          print_atomic_maps_per_file(k_resolved_density_for_output,name_of_propagator+"_k_resolved.k=");
        }
        else
        {
          std::string filename = "k_resolved_"+name_of_propagator; //Yu:this should be enough, otherwise the name will be too long
          //if(options.check_option("output_suffix"))
          //  filename += options.get_option("output_suffix",std::string(""));
          if(options.get_option(name_of_propagator + "_k_resolved_output",bool(false)))
            print_atomic_maps(k_resolved_density_for_output,filename);
        }
        reset_output_counter();
      }
    }
  }
}
