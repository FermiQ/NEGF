// NEMO5, The Nanoelectronics simulation package.
// This package is a free software.
// It is distributed under the NEMO5 Non-Commercial License (NNCL).
// Purdue Research Foundation, 1281 Win Hentschel Blvd., West Lafayette, IN 47906, USA
//$Id: ScatteringBackwardRGFSolver.cpp 21703 $

#include "ScatteringBackwardRGFSolver.h"
#include "EOMMatrixInterface.h"
#include "PetscMatrixParallelComplexContainer.h"
#include "QuantumNumberUtils.h"
#include "BlockSchroedingerModule.h"
#include "Propagator.h"
#include "PropagationUtilities.h"
#include "TransformationUtilities.h"
#include "ScatteringForwardRGFSolver.h"
#include "SelfenergyInterface.h"

ScatteringBackwardRGFSolver::ScatteringBackwardRGFSolver()
{
  repartition_solver = NULL;
  forward_solver = NULL;
  lead_domain = NULL;
  Hamiltonian = NULL;
  use_explicit_blocks = false;
  lesser_green_propagator = NULL;
  lesser_green_propagator_type = NemoPhys::Fermion_lesser_Green;
  greater_green_propagator = NULL;
  greater_green_propagator_type = NemoPhys::Fermion_advanced_Green;
  Hlesser_green_propagator = NULL;
  Hlesser_green_propagator_type = NemoPhys::Fermion_lesser_HGreen;
  do_transform_HG = true;
  temporary_HG_matrix = NULL;
  do_solve_HG = true;
  number_of_offdiagonal_blocks = 0;
  solve_greater_greens_function = false;
  energy_resolved_electron_density = false;
  k_resolved_electron_density = false;
  energy_resolved_hole_density = false;
  k_resolved_hole_density = false;
  energy_resolved_HG_density = false;
  k_resolved_HG_density = false;
  energy_resolved_DOS = false;
  k_resolved_DOS = false;
  use_density_for_resonance_mesh = true;
  explicitly_solve_greater_greens_function = false;
  use_diagonal_only = true;
  store_atom_blockdiagonal = false;
  energy_resolved_per_k = false;
  slab_resolved_local_current_energy_k_resolved.clear();
  calculate_transmission = false;
  landauer_current = false;
  number_offdiagonal_blocks = 0;
  local_Glesser_test = false;
  check_inversion_stability = false;
}

ScatteringBackwardRGFSolver::~ScatteringBackwardRGFSolver()
{
  //this is responsible for the extra propagator
  if (lesser_green_propagator != NULL)
  {
    delete_propagator_matrices(lesser_green_propagator);
    //delete propagator
    lesser_green_propagator->Propagator::~Propagator();
  }
  if (greater_green_propagator != NULL)
  {
    delete_propagator_matrices(greater_green_propagator);
    //delete propagator
    greater_green_propagator->Propagator::~Propagator();
  }
  if (Hlesser_green_propagator != NULL)
  {
    delete_propagator_matrices(Hlesser_green_propagator);
    //delete propagator
    Hlesser_green_propagator->Propagator::~Propagator();
  }
  delete temporary_HG_matrix;
  temporary_HG_matrix = NULL;

}

void ScatteringBackwardRGFSolver::do_init()
{
  read_NEGF_object_list = false;
  NEMO_ASSERT(options.check_option("external_Hamilton_Constructor"),
      "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::do_init please define \"external_Hamilton_Constructor\"\n");
  std::string temp_name = options.get_option("external_Hamilton_Constructor", std::string(""));
  options.set_option("Hamilton_constructor", temp_name);
  Greensolver::do_init();
  //additionally we have g< and HG propagators that must be calculated correctly
  options.get_option("Hamiltonian_indices", Hamiltonian_indices);

  Simulation* temp_simulation = Hamilton_Constructor;
  if (options.check_option("Repartition_solver"))
  {
    std::string name = options.get_option("Repartition_solver", std::string(""));
    Simulation* temp_simulation = find_simulation(name);
    NEMO_ASSERT(temp_simulation != NULL,
        "ForwardRGFSolver(\"" + get_name() + "\")::do_init have not found simulation \"" + name + "\"\n");
  }
  repartition_solver = dynamic_cast<RepartitionInterface*>(temp_simulation);
  NEMO_ASSERT(repartition_solver != NULL,
      "ForwardRGFSolver(\"" + get_name() + "\")::do_init \"" + temp_simulation->get_name() + "\" is not a RepartitionInterface\n");

  if (options.check_option("number_of_offdiagonal_blocks"))
  {
    number_of_offdiagonal_blocks = options.get_option("number_of_offdiagonal_blocks",0);
    use_explicit_blocks = true;
  }

  //set_writeable_propagator();
  if (options.check_option("scattering_self_energies"))
  {
    std::vector < std::string > scattering_self_energies;
    options.get_option("scattering_self_energies", scattering_self_energies);
    for (unsigned int i = 0; i < scattering_self_energies.size(); i++)
    {
      NEMO_ASSERT(options.check_option(scattering_self_energies[i] + "_solver"),
          get_name() + "have not found \"" + scattering_self_energies[i] + "+_solver\"\n");
      std::string solver_name = options.get_option(scattering_self_energies[i] + "_solver", std::string(""));
      //opt.set_option(scattering_self_energies[i]+"_solver",solver_name);
      if (options.check_option(scattering_self_energies[i] + "_constructor"))
      {
        std::string constructor_name = options.get_option(scattering_self_energies[i] + "_constructor", std::string(""));
        //opt.set_option(scattering_self_energies[i]+"_constructor",constructor_name);
        if (scattering_self_energies[i].find("retarded") != std::string::npos)
          scattering_sigmaR_solver = find_simulation(constructor_name);
        if (scattering_self_energies[i].find("lesser") != std::string::npos)
          scattering_sigmaL_solver = find_simulation(constructor_name);
      }
    }
  }
  if(electron_hole_model)
    solve_greater_greens_function = true;

  options.set_option("transform_density",true);

  if(options.get_option("energy_resolved_density",false))
  {
    energy_resolved_hole_density = true;
    energy_resolved_electron_density = true;

    //if energy resolved data for a UTB is needed then make it energy resolved per k -- spatially resolved by default.
    energy_resolved_per_k = options.get_option("energy_resolved_per_k_density",true);

  }

  if(options.get_option("k_resolved_density",false))
  {
    k_resolved_hole_density = true;
    k_resolved_electron_density = true;
  }

  use_density_for_resonance_mesh = true;

  explicitly_solve_greater_greens_function = options.get_option("explicitly_solve_greater_greens_function",bool(false));

  use_diagonal_only = options.get_option("diagonal_Greens_function", true);

  store_atom_blockdiagonal = options.get_option("store_atom_blockdiagonal",false);


  if(options.get_option("output_slab_resolved_energy_current",false))
  {
    energy_resolved_HG_density = true;

  }

  activate_regions();

  calculate_transmission = options.get_option("transmission",false);
  landauer_current = options.get_option("landauer_current",false);

  if(options.check_option("number_of_offdiagonal_blocks"))
  {
    use_explicit_blocks = true;
    number_offdiagonal_blocks = options.get_option("number_of_offdiagonal_blocks",0);
    //do_solve_HG = false;
  }

  //test local Glesser
  //local_Glesser_test = true;
  options.set_option("transpose_for_Glesser",true);

  check_inversion_stability = options.get_option("check_inversion_stability",false);
}

void ScatteringBackwardRGFSolver::do_solve()
{
  initialize_Propagators();
  Propagator::PropagatorMap& temp_map = writeable_Propagator->propagator_map;
  Propagator::PropagatorMap::iterator momentum_it = temp_map.begin();
  PetscMatrixParallelComplex* Matrix = NULL;
  //get_Greensfunction will trigger solving all relevant Propagations
  for (; momentum_it != temp_map.end(); ++momentum_it)
  {
    get_Greensfunction(momentum_it->first, Matrix, NULL, NULL, NemoPhys::Fermion_retarded_Green);
  }

}

void ScatteringBackwardRGFSolver::do_output()
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::do_output: ";
  NemoUtils::tic(tic_toc_prefix);

  if(options.get_option("energy_resolved_density_output",false))
  {
    options.set_option(lesser_green_propagator_name + "_energy_resolved_output", true);
    options.set_option(lesser_green_propagator_name + "_energy_resolved_output_real", options.get_option("energy_resolved_density_real",0.0));
    options.set_option(lesser_green_propagator_name + "_energy_resolved_output_imag", options.get_option("energy_resolved_density_imag",1.0));
    energy_resolved_electron_density = true;
    electron_density.clear();
    hole_density.clear();
    options.set_option(lesser_green_propagator_name + "_energy_resolved_output_one_per_file",options.get_option("energy_resolved_density_output_one_per_file",false));
  }
  if(options.get_option("k_resolved_density_output",false))
   {
     options.set_option(lesser_green_propagator_name + "_k_resolved_output", true);
     options.set_option(lesser_green_propagator_name + "_k_resolved_output_real", options.get_option("k_resolved_density_real",0.0));
     options.set_option(lesser_green_propagator_name + "_k_resolved_output_imag", options.get_option("k_resolved_density_imag",1.0));
     k_resolved_electron_density = true;
     electron_density.clear();
     hole_density.clear();
     options.set_option(lesser_green_propagator_name + "_k_resolved_output_one_per_file",options.get_option("k_resolved_density_output_one_per_file",false));
   }
  std::string writeable_Propagator_name = writeable_Propagator->get_name();
  if(options.get_option("energy_resolved_density_of_states_output",false))
  {
    options.set_option(writeable_Propagator_name + "_energy_resolved_output", true);
    options.set_option(writeable_Propagator_name + "_energy_resolved_output_real", options.get_option("energy_resolved_density_real",-2.0));
    options.set_option(writeable_Propagator_name + "_energy_resolved_output_imag", options.get_option("energy_resolved_density_imag",0.0));
    energy_resolved_DOS = true;
    options.set_option(writeable_Propagator_name + "_energy_resolved_output_one_per_file",options.get_option("energy_resolved_density_output_one_per_file",false));
  }
  if(options.get_option("k_resolved_density_of_states_output",false))
  {
    options.set_option(writeable_Propagator_name + "_k_resolved_output", true);
    options.set_option(writeable_Propagator_name + "_k_resolved_output_real", options.get_option("k_resolved_density_of_states_real",-1.0));
    options.set_option(writeable_Propagator_name + "_k_resolved_output_imag", options.get_option("k_resolved_density_of_states_imag",0.0));
    k_resolved_DOS = true;
    options.set_option(writeable_Propagator_name + "_k_resolved_output_one_per_file",options.get_option("k_resolved_density_of_states_output_one_per_file",false));
  }

  if (options.get_option("density_output", false))
  {
    //calculate density
    if (electron_density.empty())
    {
      calculate_density(lesser_green_propagator, &electron_density, electron_hole_model);
    }
      int myrank;
      MPI_Comm_rank(Mesh_tree_topdown.begin()->first->get_global_comm(), &myrank);

      if(electron_hole_model)
      {
        std::map<unsigned int,double> temp_density = electron_density;

        if(hole_density.empty())
        {
          calculate_density(greater_green_propagator, &hole_density, electron_hole_model);
        }

        for (std::map<unsigned int, double>::const_iterator it = hole_density.begin();
            it != hole_density.end(); it++)
        {
          if(!explicitly_solve_greater_greens_function)
            temp_density[it->first] +=  it->second;
          else
            temp_density[it->first] -=  it->second;
        }
        if(myrank==0)
        {
          print_atomic_map(temp_density, get_name() + "_EHdensity");
          print_atomic_map(electron_density, get_name() + "_electron_density");
          print_atomic_map(hole_density, get_name() + "_hole_density");
        }
      }
      else
      {
        if (myrank == 0)
        {
          print_atomic_map(electron_density, get_name() + "_density");
        }
      }
    //}
  }

  if (options.get_option("density_of_states_output", false))
    {
      //calculate density
      if (DOS.empty())
      {
        calculate_density(writeable_Propagator, &DOS, false);
      }
        int myrank;
        MPI_Comm_rank(Mesh_tree_topdown.begin()->first->get_global_comm(), &myrank);


        if (myrank == 0)
        {
          print_atomic_map(DOS, get_name() + "DOS");
        }
    }


  if (options.get_option("output_slab_resolved_current", false) && !landauer_current)
  {
    //if (slab_resolved_local_current.empty())
      output_slab_resolved_current();
  }
  else
  {
    calculate_current();
  }

  NemoUtils::toc(tic_toc_prefix);

  //Propagation::do_output();

}

void ScatteringBackwardRGFSolver::get_Greensfunction(const std::vector<NemoMeshPoint>& momentum,
    PetscMatrixParallelComplex*& result, const DOFmapInterface* row_dofmap, const DOFmapInterface* column_dofmap,
    const NemoPhys::Propagator_type& type)
{
  if(!momentum.empty())
  {
    bool data_is_ready = false;
    if (type == NemoPhys::Fermion_retarded_Green || type == NemoPhys::Boson_retarded_Green)
    {
      std::map<std::vector<NemoMeshPoint>, bool>::const_iterator momentum_it = job_done_momentum_map.find(momentum);
      if (momentum_it != job_done_momentum_map.end())
        data_is_ready = momentum_it->second;
      else
        job_done_momentum_map[momentum] = false;
      if (!data_is_ready)
        run_backward_RGF_for_momentum(momentum);

      std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name()
            + "\")::get Greensfunction after run_backward GR: ";
      NemoUtils::tic(tic_toc_prefix);
      Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      //extract the requested submatrix
      if (row_dofmap != NULL || column_dofmap != NULL)
      {
        std::vector<int> row_indices;
        if (row_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *row_dofmap, row_indices);
        std::vector<int> column_indices = row_indices;
        if (column_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *column_dofmap, column_indices);
        if (row_dofmap == NULL)
          row_indices = column_indices;
        PetscMatrixParallelComplex* temp_matrix_container = prop_it->second;
        NEMO_ASSERT(temp_matrix_container != NULL,
            "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_Greensfunction gR is still NULL\n");

        if (row_indices.size() != Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()).get_number_of_dofs() && use_explicit_blocks)
        {
          delete result;
          result = NULL;
          temp_matrix_container->get_submatrix(row_indices, column_indices, MAT_REUSE_MATRIX, result);
        }
        else
          result = prop_it->second;

      }
      else
        result = prop_it->second;

      NemoUtils::toc(tic_toc_prefix);

    }
    else if (type == NemoPhys::Fermion_lesser_Green || type == NemoPhys::Boson_lesser_Green)
    {

      //throw std::runtime_error("ScatteringBackwardRGFSolver(\""+get_name()+"\")::get_Greensfunction: lesser Green NYI\n");
      std::map<std::vector<NemoMeshPoint>, bool>::const_iterator momentum_it = lesser_propagator_job_done_momentum_map.find(
          momentum);
      if (momentum_it != lesser_propagator_job_done_momentum_map.end())
        data_is_ready = momentum_it->second;
      else
        lesser_propagator_job_done_momentum_map[momentum] = false;
      if (!data_is_ready)
        run_backward_RGF_for_momentum(momentum);
      std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name()
            + "\")::get Greensfunction after run_backward GL: ";
      NemoUtils::tic(tic_toc_prefix);
      Propagator::PropagatorMap& result_prop_map = lesser_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      //extract the requested submatrix
      if (row_dofmap != NULL || column_dofmap != NULL)
      {
        std::vector<int> row_indices;
        if (row_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *row_dofmap, row_indices);
        std::vector<int> column_indices = row_indices;
        if (column_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *column_dofmap, column_indices);
        if (row_dofmap == NULL)
          row_indices = column_indices;
        PetscMatrixParallelComplex* temp_matrix_container = prop_it->second;
        NEMO_ASSERT(temp_matrix_container != NULL,
            "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_Greensfunction gL is still NULL\n");

        if (row_indices.size() != Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()).get_number_of_dofs() && use_explicit_blocks)
        {
          delete result;
          result = NULL;
          temp_matrix_container->get_submatrix(row_indices, column_indices, MAT_REUSE_MATRIX, result);
        }
        else
          result = prop_it->second;
      }
      else
        result = prop_it->second;
      //result->assemble();
      NemoUtils::toc(tic_toc_prefix);

    }
    else if (type == NemoPhys::Fermion_greater_Green || type == NemoPhys::Boson_greater_Green)
    {
      std::map<std::vector<NemoMeshPoint>, bool>::const_iterator momentum_it = greater_propagator_job_done_momentum_map.find(
          momentum);
      if (momentum_it != greater_propagator_job_done_momentum_map.end())
        data_is_ready = momentum_it->second;
      else
        greater_propagator_job_done_momentum_map[momentum] = false;
      if (!data_is_ready)
        run_backward_RGF_for_momentum(momentum);
      std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name()
                    + "\")::get Greensfunction after run_backward GL: ";
      NemoUtils::tic(tic_toc_prefix);
      Propagator::PropagatorMap& result_prop_map = greater_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      //extract the requested submatrix
      if (row_dofmap != NULL || column_dofmap != NULL)
      {
        std::vector<int> row_indices;
        if (row_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *row_dofmap, row_indices);
        std::vector<int> column_indices = row_indices;
        if (column_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *column_dofmap, column_indices);
        if (row_dofmap == NULL)
          row_indices = column_indices;
        PetscMatrixParallelComplex* temp_matrix_container = prop_it->second;
        NEMO_ASSERT(temp_matrix_container != NULL,
            "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_Greensfunction gL is still NULL\n");

        if (row_indices.size() != Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()).get_number_of_dofs() && use_explicit_blocks)
        {
          delete result;
          result = NULL;
          temp_matrix_container->get_submatrix(row_indices, column_indices, MAT_REUSE_MATRIX, result);
        }
        else
          result = prop_it->second;
      }
      else
        result = prop_it->second;
      //result->assemble();
      NemoUtils::toc(tic_toc_prefix);
    }
    else if (type == NemoPhys::Fermion_lesser_HGreen || type == NemoPhys::Boson_lesser_HGreen)
    {

      //throw std::runtime_error("ScatteringBackwardRGFSolver(\""+get_name()+"\")::get_Greensfunction: lesser HGreen NYI\n");
      std::map<std::vector<NemoMeshPoint>, bool>::const_iterator momentum_it = Hlesser_propagator_job_done_momentum_map.find(
          momentum);
      if (momentum_it != Hlesser_propagator_job_done_momentum_map.end())
        data_is_ready = momentum_it->second;
      else
        Hlesser_propagator_job_done_momentum_map[momentum] = false;
      if (!data_is_ready)
        run_backward_RGF_for_momentum(momentum);

      std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name()
            + "\")::get Greensfunction after run_backward HG: ";
      NemoUtils::tic(tic_toc_prefix);

      Propagator::PropagatorMap& result_prop_map = Hlesser_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      //extract the requested submatrix
      if (row_dofmap != NULL || column_dofmap != NULL)
      {
        std::vector<int> row_indices;
        if (row_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *row_dofmap, row_indices);
        std::vector<int> column_indices = row_indices;
        if (column_dofmap != NULL)
          Hamilton_Constructor->translate_subDOFmap_into_int(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain()),
              *column_dofmap, column_indices);
        if (row_dofmap == NULL)
          row_indices = column_indices;
        PetscMatrixParallelComplex* temp_matrix_container = prop_it->second;
        NEMO_ASSERT(temp_matrix_container != NULL,
            "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_Greensfunction gL is still NULL\n");

        unsigned int global_size;
        Hamilton_Constructor->get_data("global_size", global_size);
        if (row_indices.size() != global_size)
        {
          delete result;
          result = NULL;
          temp_matrix_container->get_submatrix(row_indices, column_indices, MAT_REUSE_MATRIX, result);
        }
        else
        {
          result = prop_it->second;
        }
      }
      NemoUtils::toc(tic_toc_prefix);

    }
    else
    {
      throw std::runtime_error(
          "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_Greensfunction: called with unknown Propagator type\n");
    }

  }
  else
  {
    //since momentum is empty integrate
    //call function to do the integration. The result will be an unassembled container
    calculate_integrated_density_matrix(type,result);

    if(debug_output)
    {
     result->save_to_matlab_file("integrated_density_matrix.m");
    }
  }



}

void ScatteringBackwardRGFSolver::get_data(const std::string& variable, const std::vector<NemoMeshPoint>* momentum,
    PetscMatrixParallelComplex*& Matrix, const DOFmapInterface* row_dof_map, const DOFmapInterface* col_dofmap)
{
  NemoPhys::Propagator_type type;
  if(momentum!=NULL)
  {
    if (variable == writeable_Propagator->get_name())
      type = writeable_Propagator->get_propagator_type();
    else if (variable == lesser_green_propagator_name)
      type = lesser_green_propagator_type;
    else if (variable == greater_green_propagator_name)
      type = greater_green_propagator_type;
    else if (variable == Hlesser_green_propagator_name)
      type = Hlesser_green_propagator_type;
    else
      throw std::runtime_error("ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_data variable, momentum, matrix, "
          "row dof, col dof: called with unknown Propagator type\n");

    get_Greensfunction(*momentum, Matrix, row_dof_map, col_dofmap, type);
  }
  else
  {
   //assume that the caller wants integrated matrix so far the use for subatomic density
    if (variable.find("lesser"))
      type = lesser_green_propagator_type;
      else
        throw std::runtime_error("ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_data variable, momentum, matrix, "
            "row dof, col dof for integrated density matrix: called with unknown Propagator type\n");

    //call function to do the integration. The result will be an unassembled container
    calculate_integrated_density_matrix(type,Matrix);


  }


}

void ScatteringBackwardRGFSolver::calculate_integrated_density_matrix(NemoPhys::Propagator_type prop_type, PetscMatrixParallelComplex*& Matrix)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::calculate_integrated_density_matrix: ";
  NemoUtils::tic(tic_toc_prefix);

  //1. get row col indices set
  //
  std::set<std::pair<int,int> > set_row_col_indices;
  //1.1 loop through diagonal_indices and put into the set of row col indices

  //make sure the type was solved
  do_solve();

  Propagator::PropagatorMap& temp_map = writeable_Propagator->propagator_map;
  Propagator::PropagatorMap::iterator momentum_it = temp_map.begin();

  EOMMatrixInterface* eom_interface = NULL;
  get_device_Hamiltonian(momentum_it->first, Hamiltonian, eom_interface);

  PetscMatrixParallelComplexContainer* Hamiltonian_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(Hamiltonian);
  NEMO_ASSERT(Hamiltonian_container!=NULL, tic_toc_prefix + " cast to PetscMatrixParallelComplexContainer failed");
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& H_blocks =
      Hamiltonian_container->get_const_container_blocks();
  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator block_cit =
      H_blocks.begin();
  std::vector < std::pair<std::vector<int>, std::vector<int> > > diagonal_indices;
  std::vector < std::pair<std::vector<int>, std::vector<int> > > offdiagonal_indices;
  EOMMatrixInterface* temp_module = dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
  NEMO_ASSERT(temp_module != NULL,
      "ScatteringBackwardRGFSolver(\"" + get_name()
      + "\")::calculate_integrated_density_matrix Hamilton_Constructor does not inherit from class EOMMatrixInterface\n");
  temp_module->get_ordered_Block_keys(diagonal_indices, offdiagonal_indices);
  int num_slabs = diagonal_indices.size();
  vector<int> slab_start_indices(num_slabs);
  vector<int> slab_end_indices(num_slabs);
  int counter = 0;
  for (int slab_idx =0; slab_idx < num_slabs; ++slab_idx)
  {
    std::vector<int> slab_indices = diagonal_indices[slab_idx].first;
    int slab_size = slab_indices.size();
    slab_start_indices[slab_idx] = counter;
    for(int idx = 0; idx < slab_size; ++idx)
    {
      set_row_col_indices.insert(std::make_pair<int,int>(slab_indices[idx],slab_indices[idx]));
      counter++;
    }
    slab_end_indices[slab_idx] = counter;
  }

  //get all momenta
  std::set<std::vector<NemoMeshPoint> > integration_range ;
  std::set<std::vector<NemoMeshPoint> >* pointer_to_integration_range = &integration_range;

  Parallelizer->get_data("all_momenta",pointer_to_integration_range);


  int root = 0 ;//rank = 0


  std::vector<int> group; //empty not used by default MPI reduce
  const std::set<std::string> exclude_integration; //empty not needed for this integration

  std::vector<cplx> result_vector;
  std::vector<cplx>* pointer_to_result_vector = &result_vector;

  PropagationUtilities::integrate_submatrix(this, set_row_col_indices,pointer_to_integration_range, this, prop_type, Hamiltonian->get_communicator(),
      root, group, pointer_to_result_vector, exclude_integration);


  //create result container
  if (holder.geometry_replica == 0)
    if (holder.geometry_rank == 0)
    {

      PetscMatrixParallelComplexContainer* temp_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
          Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
          //dynamic_cast<PetscMatrixParallelComplexContainer*>(Matrix);

      double prefactor = 1;
      const std::vector<std::string>& temp_mesh_names = lesser_green_propagator->momentum_mesh_names;
      for (unsigned int i = 0; i < temp_mesh_names.size(); i++)
      {
        if (temp_mesh_names[i].find("energy") != std::string::npos || temp_mesh_names[i].find("momentum_1D") != std::string::npos)
          prefactor /= 2.0 * NemoMath::pi;
        else if (temp_mesh_names[i].find("momentum_2D") != std::string::npos)
          prefactor /= 4.0 * NemoMath::pi * NemoMath::pi;
        else if (temp_mesh_names[i].find("momentum_3D") != std::string::npos)
          prefactor /= 8.0 * NemoMath::pi * NemoMath::pi * NemoMath::pi;
      }


      //3. put result vector into container
      int counter = 0;
      for (int slab_idx = 0; slab_idx < num_slabs; ++slab_idx)
      {
        PetscMatrixParallelComplex* temp_submatrix = new PetscMatrixParallelComplex(diagonal_indices[slab_idx].first.size(),
            diagonal_indices[slab_idx].second.size(),Hamiltonian->get_communicator());

        temp_submatrix->set_num_owned_rows(diagonal_indices[slab_idx].first.size());
        temp_submatrix->consider_as_full();
        for(unsigned int idx=0; idx<diagonal_indices[slab_idx].first.size(); idx++)
          temp_submatrix->set_num_nonzeros_for_local_row(idx,diagonal_indices[slab_idx].second.size(),0);
        temp_submatrix->allocate_memory();

        //get values from result_vector
        vector<cplx> values(diagonal_indices[slab_idx].first.size());
        vector<int> subindices(diagonal_indices[slab_idx].first.size());
        for(int idx = 0; idx < values.size(); ++idx)
        {
          values[idx] = result_vector[counter]*cplx(prefactor,0.0);
          subindices[idx] = idx;
          ++counter;

        }
        temp_submatrix->set(subindices, subindices ,values);
        temp_submatrix->assemble();
        //temp_submatrix->save_to_matlab_file("G"+subdomain_names[slab_idx]+".m");
        temp_container->set_block_from_matrix1(*temp_submatrix, diagonal_indices[slab_idx].first,diagonal_indices[slab_idx].second);

        delete temp_submatrix;
        temp_submatrix = NULL;

      }
      //delete temp_container ;
      //temp_container = NULL;
      Matrix = temp_container;
    }
    else
      Matrix = NULL;

  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::run_backward_RGF_for_momentum(const std::vector<NemoMeshPoint>& momentum)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::run_backward_RGF_for_momentum: ";
  NemoUtils::tic(tic_toc_prefix);

  initialize_Propagators();

  std::map<NemoPhys::Propagator_type, bool>::const_iterator temp_cit = propagator_initialized_map.find(
       writeable_Propagator->get_propagator_type());
   //NEMO_ASSERT(temp_cit!=Propagator_is_initialized.end(),get_name() + " variable, Propagator_pointer could not find writeable_Propagator");
   bool propagator_initialized = false;
   if (temp_cit->second)
     propagator_initialized = true;
   if (!propagator_initialized)
     initialize_Propagator(writeable_Propagator);
   propagator_initialized_map[ writeable_Propagator->get_propagator_type()] = true;

  temp_cit = propagator_initialized_map.find(
      lesser_green_propagator_type);
  //NEMO_ASSERT(temp_cit!=Propagator_is_initialized.end(),get_name() + " variable, Propagator_pointer could not find writeable_Propagator");
  propagator_initialized = false;
  if (temp_cit->second)
    propagator_initialized = true;
  if (!propagator_initialized)
    initialize_Propagator(lesser_green_propagator);
  propagator_initialized_map[lesser_green_propagator_type] = true;

  temp_cit = propagator_initialized_map.find(Hlesser_green_propagator_type);
  //NEMO_ASSERT(temp_cit!=Propagator_is_initialized.end(),get_name() + " variable, Propagator_pointer could not find writeable_Propagator");
  propagator_initialized = false;
  if (temp_cit->second)
    propagator_initialized = true;
  if (!propagator_initialized)
    initialize_Propagator(Hlesser_green_propagator);
  propagator_initialized_map[Hlesser_green_propagator_type] = true;

  if(solve_greater_greens_function)
  {
    temp_cit = propagator_initialized_map.find(greater_green_propagator_type);
    //NEMO_ASSERT(temp_cit!=Propagator_is_initialized.end(),get_name() + " variable, Propagator_pointer could not find writeable_Propagator");
    propagator_initialized = false;
    if (temp_cit->second)
      propagator_initialized = true;
    if (!propagator_initialized)
      initialize_Propagator(greater_green_propagator);
    propagator_initialized_map[greater_green_propagator_type] = true;
  }
  //0. get some general information
  //0.1 get the list of subdomain names
  if (subdomain_names.size() < 1)
    repartition_solver->get_subdomain_names(get_const_simulation_domain(), true, subdomain_names);

  //1. get the Hamiltonian block matrices
  EOMMatrixInterface* eom_interface = NULL;
  get_device_Hamiltonian(momentum, Hamiltonian, eom_interface);
  if (debug_output)
    Hamiltonian->save_to_matlab_file("total_H.m");

  //2.1 get the contact self-energy
  PetscMatrixParallelComplex* gR = NULL;
  GreensfunctionInterface* temp_gR_interface = dynamic_cast<GreensfunctionInterface*>(forward_gR_solver);
  NEMO_ASSERT(temp_gR_interface != NULL,
      "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_contact_sigma \"" + forward_gR_solver->get_name()
          + "\" is no GreensfunctionInterface\n");
  if (particle_type_is_Fermion)
    temp_gR_interface->get_Greensfunction(momentum, gR, NULL, NULL, NemoPhys::Fermion_retarded_Green);
  else
    temp_gR_interface->get_Greensfunction(momentum, gR, NULL, NULL, NemoPhys::Boson_retarded_Green);

  //PetscMatrixParallelComplex* Sigma_contact = NULL;
  //get_contact_sigma(momentum, eom_interface, Sigma_contact, subdomain_names[subdomain_names.size() - 1], lead_domain, Hamiltonian, gR);


  //Sigma_contact->save_to_matlab_file("sigmaL_drain_lead.m");

  //3. loop over the blocks of 1. and solve backward RGF
  PetscMatrixParallelComplexContainer* Hamiltonian_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(Hamiltonian);
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& H_blocks =
      Hamiltonian_container->get_const_container_blocks();
  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator block_cit =
      H_blocks.begin();
  std::vector < std::pair<std::vector<int>, std::vector<int> > > diagonal_indices;
  std::vector < std::pair<std::vector<int>, std::vector<int> > > offdiagonal_indices;
  EOMMatrixInterface* temp_module = dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
  NEMO_ASSERT(temp_module != NULL,
      "ScatteringBackwardRGFSolver(\"" + get_name()
          + "\")::run_backward_RGF_for_momentum Hamilton_Constructor does not inherit from class EOMMatrixInterface\n");
  temp_module->get_ordered_Block_keys(diagonal_indices, offdiagonal_indices);

  //get pre-last subdomain gR and calculate sigma
  if (forward_solver == NULL)
  {
    std::string forward_solver_name = options.get_option("forward_RGF_solver", std::string(""));
    Simulation* temp_solver = find_simulation(forward_solver_name);
    NEMO_ASSERT(temp_solver != NULL,
        "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found gr-solver \""
            + forward_solver_name + "\"\n");
    forward_solver = dynamic_cast<GreensfunctionInterface*>(temp_solver);
    NEMO_ASSERT(forward_solver != NULL,
        "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum \"" + forward_solver_name
            + "\" is not a GreensfunctionInterface\n");

  }
  //calculate last diagonal block
  //gR = Gr

  PetscMatrixParallelComplex* temp_container = NULL;
  if (particle_type_is_Fermion)
      forward_solver->get_Greensfunction(momentum, temp_container, NULL, NULL, NemoPhys::Fermion_retarded_Green);
    else
      forward_solver->get_Greensfunction(momentum, temp_container, NULL, NULL, NemoPhys::Boson_retarded_Green);
    PetscMatrixParallelComplexContainer* gR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
    NEMO_ASSERT(gR_container != NULL,
        "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum gR is not received in the container format\n");
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks =
        gR_container->get_const_container_blocks();

    std::string forward_solver_name = options.get_option("forward_RGF_solver", std::string(""));
    Simulation* temp_solver = find_simulation(forward_solver_name);
    temp_solver->get_data("Ek_stable_inversion_map",Ek_stable_inversion);

    PetscMatrixParallelComplex *exact_GR = NULL;
    solve_last_diagonal_block_GR(momentum, eom_interface, H_blocks, diagonal_indices, offdiagonal_indices, gR_blocks, gR, exact_GR);


   /*
   if(use_explicit_blocks)
   {
     //need Gr gR and L blocks

     //get Gr blocks
     Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
     Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
     PetscMatrixParallelComplex* temp_container = prop_it->second;
     PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
     NEMO_ASSERT(GR_container != NULL,
         "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum GR is not received in the container format\n");
     const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
         GR_container->get_const_container_blocks();

     //get L blocks
     PetscMatrixParallelComplex* temp_matrix;
     Simulation* temp_simulation=dynamic_cast<Simulation*>(forward_solver);
     temp_simulation->get_data(std::string("L_container"), &momentum, temp_matrix);
     PetscMatrixParallelComplexContainer* L_block_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix);
     const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
         L_block_container->get_const_container_blocks();

     int col_index = subdomain_names.size()-1;

     //loop through j which is nonlocality index
     for(int j = 1; j <= col_index-1; ++j)
     {
       if(j < number_of_offdiagonal_blocks)
       {
         cerr << "j " << j << "\n";
         solve_offdiagonal_GR_nonlocal_blocks(momentum, diagonal_indices, number_of_offdiagonal_blocks, col_index-j, col_index,
             L_blocks, GR_blocks);
       }
     }
   }
   */


   //calculate transmission if requested
   if(calculate_transmission)
   {
     PetscMatrixParallelComplex* exact_GR_block_for_transmission = exact_GR;
     PetscMatrixParallelComplex* Sigma_contact_end = NULL;
     PetscMatrixParallelComplex* Sigma_contact_left = NULL;

     //gR_container->get_submatrix(diagonal_indices[diagonal_indices.size() - 1].first, diagonal_indices[diagonal_indices.size() - 1].second, MAT_REUSE_MATRIX, exact_GR_block_for_transmission);


     std::string forward_solver_name = options.get_option("forward_RGF_solver", std::string(""));
     Simulation *temp_solver = find_simulation(forward_solver_name);
     ScatteringForwardRGFSolver* forward_gR_solver = dynamic_cast<ScatteringForwardRGFSolver*>(temp_solver);
     NEMO_ASSERT(forward_gR_solver != NULL, "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum " + forward_solver_name + " is not a ScatteringForwardRGFSolver\n");

     //get left sigma
     //forward_gR_solver->get_start_lead_sigma(momentum, Sigma_contact_end);
     //get_contact_sigma(momentum,eom_interface, Sigma_contact_left, subdomain_names[subdomain_names.size() - 2],
     //get the right coupling Hamiltonian
     std::map< std::pair < std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex* >::const_iterator off_block_cit=H_blocks.find(offdiagonal_indices[subdomain_names.size() - 2]);
     NEMO_ASSERT(off_block_cit!=H_blocks.end(),"ForwardRGFSolver(\""+get_name()+"\")::run_forward_RGF_for_momentum have not found offdiagonal block Hamiltonian\n");
     PetscMatrixParallelComplex* coupling_Hamiltonian=off_block_cit->second;

     if(!use_explicit_blocks)
     {
       //forward_gR_solver->get_Greensfunction(momentum,block_gR,)
       PetscMatrixParallelComplex* block_gR = NULL;
       gR_container->get_submatrix(diagonal_indices[diagonal_indices.size() - 2].first, diagonal_indices[diagonal_indices.size() - 2].second, MAT_REUSE_MATRIX, block_gR);
       get_contact_sigma(momentum,eom_interface, Sigma_contact_left, subdomain_names[subdomain_names.size() - 1], lead_domain, Hamiltonian, block_gR,coupling_Hamiltonian);
     }
     else
     {

       PetscMatrixParallelComplex* temp_matrix;
       Simulation* temp_simulation=dynamic_cast<Simulation*>(forward_solver);
       temp_simulation->get_data(std::string("L_container"), &momentum, temp_matrix);
       PetscMatrixParallelComplexContainer* L_block_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix);
       const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
           L_block_container->get_const_container_blocks();

       PropagationUtilities::get_contact_sigmaR_nonlocal(this, momentum, eom_interface, subdomain_names.size() - 1, number_offdiagonal_blocks-1, subdomain_names,
           diagonal_indices, L_block_container, gR_container, Sigma_contact_left, scattering_sigmaR_solver);


     }
     //NemoMath::symmetry_type type = NemoMath::symmetric;
     //PropagationUtilities::symmetrize(this, Sigma_contact_left,type);
     //get end sigma
     get_contact_sigma(momentum, eom_interface, Sigma_contact_end, subdomain_names[subdomain_names.size() - 1], lead_domain, Hamiltonian);
     //NemoMath::symmetry_type type_symm = NemoMath::symmetric;
     //PropagationUtilities::symmetrize(this,Sigma_contact_end,type_symm);
     //calculate transmission
     double transmission;
     PropagationUtilities::core_calculate_transmission(this, Sigma_contact_left, Sigma_contact_end, exact_GR_block_for_transmission, transmission);

     delete Sigma_contact_end;
     Sigma_contact_end = NULL;
     delete Sigma_contact_left;
     Sigma_contact_left= NULL;
     exact_GR_block_for_transmission = NULL;
     //save transmission in map
     if (momentum_transmission.find(momentum) != momentum_transmission.end())
       momentum_transmission[momentum] += transmission;
     else
       momentum_transmission[momentum] = transmission;

     //cerr << " transmission " << transmission << " \n";
   }

  if(debug_output)
  {
   exact_GR->save_to_matlab_file("exact_GR_last_subdomain.m");
  }
  //gL = Gl
  temp_container = NULL;
   forward_solver->get_Greensfunction(momentum, temp_container, NULL, NULL, lesser_green_propagator_type);
   PetscMatrixParallelComplexContainer* gL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
   NEMO_ASSERT(gL_container != NULL,
       "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum gL is not a container \n");
   const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gL_blocks =
       gL_container->get_const_container_blocks();
  if(!use_explicit_blocks)
    solve_last_diagonal_block_GL(momentum, eom_interface,H_blocks, diagonal_indices, offdiagonal_indices, gL_blocks, gR, exact_GR);
  else
  {

    //in the nonlocal case off diagonal g< != G<

    //off diagonal g<
    int i = subdomain_names.size()-1;

    //solve off diagonal gL
    //Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
    //Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);

    //PetscMatrixParallelComplexContainer* gL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);

    Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

    PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
        GR_container->get_const_container_blocks();

    std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[i].first,diagonal_indices[i].second);
    std::map< std::pair < std::vector<int>, std::vector<int> >,
    PetscMatrixParallelComplex* >::const_iterator GRblock_cit=GR_blocks.find(temp_pair2);
    PetscMatrixParallelComplex* GR_row_row = GRblock_cit->second;

    PetscMatrixParallelComplex* temp_matrix;
    Simulation* temp_simulation=dynamic_cast<Simulation*>(forward_solver);
    temp_simulation->get_data(std::string("L_container"), &momentum, temp_matrix);
    PetscMatrixParallelComplexContainer* L_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix);

    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
        L_container->get_const_container_blocks();


    int start_index = i-number_offdiagonal_blocks;
    if(start_index < 0 )// subdomain_names.size()-1)
      start_index = 0; //subdomain_names.size()-1;
    int end_index = i-1;//start_index+number_offdiagonal_blocks;

    for(int j = start_index; j <= end_index; ++j)
    {

      PetscMatrixParallelComplex* result_gL = NULL;
      PropagationUtilities::solve_offdiagonal_gL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
          number_offdiagonal_blocks, i, j, start_index, subdomain_names, L_blocks, gR_blocks, GR_row_row, scattering_sigmaL_solver, gL_container,
          result_gL);
      delete result_gL;
      result_gL = NULL;
    }
    int start_index_diag = i-number_offdiagonal_blocks;
    if(start_index_diag < 0)
      start_index_diag = 0;

    PetscMatrixParallelComplex* Sigma_lesser_contact = NULL;

    get_contact_sigma_lesser_equilibrium(momentum, eom_interface, Sigma_lesser_contact, subdomain_names[subdomain_names.size() - 1],
        lead_domain, Hamiltonian_indices, Hamiltonian, gR /*last_coupling*/);
    NemoMath::symmetry_type type = NemoMath::antihermitian;

    PropagationUtilities::symmetrize(this,Sigma_lesser_contact, type);
    if(debug_output)
    {
      Sigma_lesser_contact->save_to_matlab_file("Sigmalesser_last.m");
      GR_row_row->save_to_matlab_file("gR_last.m");
    }

    std::vector<cplx> diagonal;
    Sigma_lesser_contact->get_diagonal(&diagonal);

    double tmp_sum_real = 0.0;
    for (unsigned int idx =0; idx < diagonal.size(); idx++)
      tmp_sum_real += diagonal[idx].real();
    std::string name_of_propagator;
    Propagator* writeable_Propagator=NULL;
    //PropagatorInterface* PropInterface=get_PropagatorInterface(this_simulation);
    //PropInterface->get_Propagator(writeable_Propagator);
    //double energy = PropagationUtilities::read_energy_from_momentum(this_simulation,momentum,writeable_Propagator);
    //cerr << "Sigma_lesser_contact row " << " " <<  tmp_sum_real << " \n";

    PetscMatrixParallelComplex* result_gL = NULL;
    PropagationUtilities::solve_diagonal_gL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
        number_offdiagonal_blocks, i, start_index_diag, subdomain_names, L_blocks, gR_blocks, Sigma_lesser_contact, GR_row_row,
        scattering_sigmaL_solver, gL_container, result_gL);

    //delete Sigma_lesser_contact;
    //Sigma_lesser_contact = NULL;


    Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);
    if (prop_it == result_prop_map_lesser.end())
    {
      result_prop_map[momentum] = NULL;
      prop_it_lesser = result_prop_map_lesser.find(momentum);
    }
    if (prop_it_lesser->second == NULL)
    {
      PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
          Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
      lesser_green_propagator->allocated_momentum_Propagator_map[momentum] = true;
      prop_it_lesser->second = new_container;
    }
    NemoMath::symmetry_type type2 = NemoMath::antihermitian;
    symmetrize(result_gL, type2);
    PetscMatrixParallelComplexContainer* GL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);
    GL_container->set_block_from_matrix1(*result_gL,
        diagonal_indices[diagonal_indices.size() - 1].first, diagonal_indices[diagonal_indices.size() - 1].second);

    delete result_gL;
    result_gL = NULL;

    /*
    //solve off diagonal Gl
    for(int j = start_index; j <= end_index; ++j)
    {
      PetscMatrixParallelComplex* result_gL = NULL;

      PropagationUtilities::solve_offdiagonal_GL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
                       number_offdiagonal_blocks, i, j, start_index, subdomain_names,
                       L_blocks,gR_blocks,gL_blocks,GR_blocks,
                       scattering_sigmaL_solver, GL_container,result_gL);

      delete result_gL;
      result_gL = NULL;
    }*/
  }
  if (do_solve_HG)
  {
    //solve last HG and put zeros for first
    PetscMatrixParallelComplex fake_0_matrix(diagonal_indices[0].first.size(), diagonal_indices[0].second.size(),
        exact_GR->get_communicator());
    int start_row = 0;
    int end_row = diagonal_indices[0].first.size(); //fix this
    fake_0_matrix.set_num_owned_rows(end_row - start_row);
    for (int i = start_row; i < end_row; i++)
      fake_0_matrix.set_num_nonzeros_for_local_row(i, 1, 0);

    fake_0_matrix.allocate_memory();
    fake_0_matrix.set_to_zero();
    fake_0_matrix.assemble();

    Propagator::PropagatorMap& result_prop_map = Hlesser_green_propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
    if (prop_it == result_prop_map.end())
    {
      result_prop_map[momentum] = NULL;
      prop_it = result_prop_map.find(momentum);
    }
    if (prop_it->second == NULL)
    {
      PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
          Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
      Hlesser_green_propagator->allocated_momentum_Propagator_map[momentum] = true;
      prop_it->second = new_container;
    }
    PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
    temp_container->set_block_from_matrix1(fake_0_matrix, diagonal_indices[0].first, diagonal_indices[0].second);

    if (!use_explicit_blocks)
    {
      //prepare_blocks_HG
      int i = subdomain_names.size() - 1;
      PetscMatrixParallelComplex* left_coupling;
      PetscMatrixParallelComplex* gR_block;
      PetscMatrixParallelComplex* GL_block;
      PetscMatrixParallelComplex* gL_block;
      PetscMatrixParallelComplex* GR_block;
      prepare_HG_matrices(momentum, i, H_blocks, diagonal_indices, offdiagonal_indices,
          gR_blocks, gL_blocks, left_coupling, gR_block, GL_block, gL_block, GR_block);

      PetscMatrixParallelComplex* HG = NULL;
      solve_HG(diagonal_indices[i], momentum, left_coupling, gR_block, GL_block, gL_block, GR_block, HG);
      delete left_coupling;
      left_coupling = NULL;
      delete HG;
      HG = NULL;
    }
    else
    {

      PetscMatrixParallelComplex fake_last_matrix(diagonal_indices[subdomain_names.size()-1].first.size(), diagonal_indices[subdomain_names.size()-1].second.size(),
          exact_GR->get_communicator());
      int start_row = 0;
      int end_row = diagonal_indices[subdomain_names.size()-1].first.size(); //fix this
      fake_last_matrix.set_num_owned_rows(end_row - start_row);
      for (int i = start_row; i < end_row; i++)
        fake_last_matrix.set_num_nonzeros_for_local_row(i, 1, 0);

      fake_last_matrix.allocate_memory();
      fake_last_matrix.set_to_zero();
      fake_last_matrix.assemble();
      PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
         temp_container->set_block_from_matrix1(fake_last_matrix,
             diagonal_indices[subdomain_names.size()-1].first, diagonal_indices[subdomain_names.size()-1].second);
    }
  }
  if(solve_greater_greens_function)
  {
    int i = subdomain_names.size() - 1;
    PetscMatrixParallelComplex* GL_block = NULL;
    {
      Propagator::PropagatorMap& result_prop_map = lesser_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      PetscMatrixParallelComplexContainer* GL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GL_blocks =
          GL_container->get_const_container_blocks();
      std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GL_block_cit =
          GL_blocks.find(diagonal_indices[i]);
      NEMO_ASSERT(GL_block_cit != GL_blocks.end(),
          "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of GL\n");
      GL_block = GL_block_cit->second;
    }
    //PetscMatrixParallelComplex* GR_block = NULL;
    Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
    Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

    PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
        GR_container->get_const_container_blocks();
    std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GR_block_cit =
        GR_blocks.find(diagonal_indices[i]);
    NEMO_ASSERT(GR_block_cit != GR_blocks.end(),
        "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of GR\n");
    //GR_block = GR_block_cit->second;


    PetscMatrixParallelComplex *this_GG_block = NULL;
    solve_diagonal_GG(diagonal_indices[i], momentum, exact_GR, GL_block, this_GG_block);
    delete this_GG_block;
    this_GG_block = NULL;
  }

  delete exact_GR;
  exact_GR = NULL;

  //loop through rest of the subdomains and solve Gr and G< and HG
  for (int i = diagonal_indices.size() - 2; i >= 0; i--)
  {
    msg.threshold(2) << get_name() << " solving domain " << i << " \n";
    //solve diagonal Gr

    //get the right coupling Hamiltonian
    PetscMatrixParallelComplex* coupling = NULL;
    if (!use_explicit_blocks)
    {
      std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator off_block_cit =
          H_blocks.find(offdiagonal_indices[i]);
      NEMO_ASSERT(off_block_cit != H_blocks.end(),
          "ScatteringBackwardRGFSolver(\"" + get_name()
              + "\")::run_backward_RGF_for_momentum have not found offdiagonal block Hamiltonian\n");
      coupling = off_block_cit->second;
    }
    else
    {
      //NYI
      //TODO:JC
      std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator off_block_cit =
                H_blocks.find(offdiagonal_indices[i]);
            NEMO_ASSERT(off_block_cit != H_blocks.end(),
                "ScatteringBackwardRGFSolver(\"" + get_name()
                    + "\")::run_backward_RGF_for_momentum have not found offdiagonal block Hamiltonian\n");
            coupling = off_block_cit->second;

    }

    //get gR
    PetscMatrixParallelComplex* gR_block = NULL;
    std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator gR_block_cit
                                                                                          = gR_blocks.find(diagonal_indices[i]);
    NEMO_ASSERT(gR_block_cit != gR_blocks.end(),
        "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of gR\n");
    gR_block = gR_block_cit->second;

    PetscMatrixParallelComplex* this_GR_block = NULL;
    if (!use_explicit_blocks)
    {
      //get GR blocks
      Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
      NEMO_ASSERT(prop_it != result_prop_map.end(),
          "ScatteringBackwardRGFSolver(\"" + get_name()
              + "\")::run_backward_RGF_for_momentum could not find momentum in writeable Propagator GR")
      PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
          GR_container->get_const_container_blocks();
      std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GR_block_cit =
          GR_blocks.find(diagonal_indices[i + 1]);
      exact_GR = GR_block_cit->second;

      solve_diagonal_GR(diagonal_indices[i], momentum, exact_GR, gR_block, coupling, this_GR_block);
    }
    else
    {
      //NYI
      //TODO:JC
      //get GR blocks
      Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
      NEMO_ASSERT(prop_it != result_prop_map.end(),
          "ScatteringBackwardRGFSolver(\"" + get_name()
          + "\")::run_backward_RGF_for_momentum could not find momentum in writeable Propagator GR")
      PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
          GR_container->get_const_container_blocks();
      std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GR_block_cit =
          GR_blocks.find(diagonal_indices[i + 1]);
      exact_GR = GR_block_cit->second;

      //get L blocks
      PetscMatrixParallelComplex* temp_matrix;
      Simulation* temp_simulation=dynamic_cast<Simulation*>(forward_solver);
      temp_simulation->get_data(std::string("L_container"), &momentum, temp_matrix);
      PetscMatrixParallelComplexContainer* L_block_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
          L_block_container->get_const_container_blocks();

      //solve_diagonal_GR(diagonal_indices[i], momentum, exact_GR, gR_block, coupling, this_GR_block);

      int col_index = i;

      unsigned int actual_requested = number_of_offdiagonal_blocks;
      //check bounds
      int difference = subdomain_names.size()-i-1;
      actual_requested = std::min(int(number_of_offdiagonal_blocks),difference);

      //loop through j which is nonlocality index
      for(unsigned int j = 1; j <= actual_requested; ++j)
      {
        //if(j < number_of_offdiagonal_blocks)
        {
          //cerr << "j " << j << "\n";
          int start_index = i+1;
          PetscMatrixParallelComplex* result_Gr = NULL;
          //PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
          //NEMO_ASSERT(GR_container != NULL,
          //    "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum GR is not received in the container format\n");
          PropagationUtilities::solve_offdiagonal_GR_nonlocal_blocks(this, momentum, diagonal_indices, number_of_offdiagonal_blocks-1, col_index+j, col_index, start_index,
              subdomain_names, L_blocks, gR_blocks, GR_container, result_Gr);
          delete result_Gr;
          result_Gr = NULL;
        }
      }
      //}

      solve_diagonal_GR_nonlocal(diagonal_indices, momentum, number_of_offdiagonal_blocks, col_index,
          L_blocks, gR_block, gR_blocks, this_GR_block);

      if(debug_output)
      {
        Domain* temp_domain = Domain::get_domain(subdomain_names[i]);
        this_GR_block->save_to_matlab_file("GRresult_"+temp_domain->get_name()+".m");
      }

    }
    //extra memory needs to be deleted (stored in container)
    delete this_GR_block;
    this_GR_block = NULL;

    //solve diagonal Gl
    PetscMatrixParallelComplex* this_GL_block = NULL;

    if (!use_explicit_blocks)
    {
      PetscMatrixParallelComplex* Sigma_lesser_contact_GL = NULL;
      PetscMatrixParallelComplex* Sigma_retarded_contact_GL = NULL;
      PetscMatrixParallelComplex* gL_block = NULL;
      prepare_diagonal_GL_matrices(momentum, i, eom_interface, H_blocks, diagonal_indices, offdiagonal_indices,
          gL_blocks, exact_GR, Sigma_lesser_contact_GL, Sigma_retarded_contact_GL, gL_block);
      solve_diagonal_GL(diagonal_indices[i], momentum, Sigma_lesser_contact_GL, Sigma_retarded_contact_GL, gR_block, gL_block,
                    this_GL_block);

      delete Sigma_lesser_contact_GL;
      Sigma_lesser_contact_GL = NULL;
      delete Sigma_retarded_contact_GL;
      Sigma_retarded_contact_GL = NULL;

      //solve greater if needed
      if(solve_greater_greens_function)
      {
        PetscMatrixParallelComplex* GR_block = NULL;
        {
          Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
          Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

          PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
          const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
              GR_container->get_const_container_blocks();
          std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GR_block_cit =
              GR_blocks.find(diagonal_indices[i]);
          NEMO_ASSERT(GR_block_cit != GR_blocks.end(),
              "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of GR\n");
          GR_block = GR_block_cit->second;
        }

        PetscMatrixParallelComplex *this_GG_block = NULL;
        solve_diagonal_GG(diagonal_indices[i], momentum, GR_block, this_GL_block, this_GG_block);
        //Domain* temp_domain = Domain::get_domain(subdomain_names[i]);
        /*if(i==0)
        {
          std::string temp_string;
          const std::vector<NemoMeshPoint>* temp_pointer=&momentum;
          PropagationUtilities::translate_momentum_vector(this,temp_pointer, temp_string);
          temp_string+=temp_domain->get_name();
          PetscMatrixParallelComplex* temp_result = NULL;
          this_GG_block->get_diagonal_matrix(temp_result);
          temp_result->save_to_matlab_file("GG_"+temp_string+".m");
          PetscMatrixParallelComplex* temp_result2 = NULL;
          this_GL_block->get_diagonal_matrix(temp_result2);
          temp_result2->save_to_matlab_file("GL_"+temp_string+".m");
          PetscMatrixParallelComplex* temp_result3 = NULL;
          GR_block->get_diagonal_matrix(temp_result3);
          temp_result3->save_to_matlab_file("GR_"+temp_string+".m");


        }*/
        //if(my_rank==0)
        {
          //this_GG_block->save_to_matlab_file("GG_"+temp_domain->get_name()+".m");
          //this_GL_block->save_to_matlab_file("GL_"+temp_domain->get_name()+".m");

        }

        delete this_GG_block;
        this_GG_block = NULL;
      }
      delete this_GL_block;
      this_GL_block = NULL;
    }
    else
    {

      //solve off diagonal Gl
//      int start_index = 1;//i-number_offdiagonal_blocks;//i+1;//-number_offdiagonal_blocks;
//      //if(start_index < 0 )// subdomain_names.size()-1)
      //  start_index = 0; //subdomain_names.size()-1;
//      int end_index = number_offdiagonal_blocks; //i;//start_index+number_offdiagonal_blocks;
//      if(end_index>(number_offdiagonal_blocks-(i+1))
//        end_index = subdomain_names.size()-1;
//      for(int j = start_index; j <= end_index; ++j)
      //int col_index = i;

      unsigned int actual_requested = number_of_offdiagonal_blocks;
      //check bounds
      int difference = i+1;//subdomain_names.size()-i-1;
      actual_requested = std::min(int(number_of_offdiagonal_blocks),difference);

      PetscMatrixParallelComplex* result_GL = NULL;
      Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
          GR_container->get_const_container_blocks();

      PetscMatrixParallelComplex* temp_matrix;
      Simulation* temp_simulation=dynamic_cast<Simulation*>(forward_solver);
      temp_simulation->get_data(std::string("L_container"), &momentum, temp_matrix);
      PetscMatrixParallelComplexContainer* L_block_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
          L_block_container->get_const_container_blocks();

      Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);

      PetscMatrixParallelComplexContainer* GL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GL_blocks =
          GL_container->get_const_container_blocks();

      //loop through j which is nonlocality index
      for(unsigned int j = 1; j <= actual_requested; ++j)
      {


        int start_index = (i+1-j)+1; //one more than the row
        PropagationUtilities::solve_offdiagonal_GL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
            number_offdiagonal_blocks, i+1-j, i+1/*j*/, start_index/*start_index*/, subdomain_names,
            L_blocks,gR_blocks,gL_blocks,GR_blocks,
            scattering_sigmaL_solver, GL_container,result_GL);

        delete result_GL;
        result_GL = NULL;
      }

      int start_index_diag = i + 1;
      PropagationUtilities::solve_offdiagonal_GL_nonlocal_blocks(this, momentum, eom_interface, diagonal_indices,
          number_offdiagonal_blocks, i, i/*j*/, start_index_diag/*start_index*/, subdomain_names,
          L_blocks,gR_blocks,gL_blocks,GR_blocks,
          scattering_sigmaL_solver, GL_container,this_GL_block);

      /*
      //NYI
      //TODO:JC
      PetscMatrixParallelComplex* Sigma_lesser_contact_GL = NULL;
      PetscMatrixParallelComplex* Sigma_retarded_contact_GL = NULL;
      PetscMatrixParallelComplex* gL_block = NULL;
      prepare_diagonal_GL_matrices(momentum, i, eom_interface, H_blocks, diagonal_indices, offdiagonal_indices,
          gL_blocks, exact_GR, Sigma_lesser_contact_GL, Sigma_retarded_contact_GL, gL_block);
      solve_diagonal_GL(diagonal_indices[i], momentum, Sigma_lesser_contact_GL, Sigma_retarded_contact_GL, gR_block, gL_block,
          this_GL_block);

      delete Sigma_lesser_contact_GL;
      Sigma_lesser_contact_GL = NULL;
      delete Sigma_retarded_contact_GL;
      Sigma_retarded_contact_GL = NULL;
*/

      delete this_GL_block;
      this_GL_block = NULL;
    }

    //initialize_HG_correlation(subdomain_names[i],subdomain_names[i-1]);
    if (do_solve_HG)
    {
      if (!use_explicit_blocks && i >0)
      {
        PetscMatrixParallelComplex* left_coupling;
        PetscMatrixParallelComplex* gR_block;
        PetscMatrixParallelComplex* GL_block;
        PetscMatrixParallelComplex* gL_block;
        PetscMatrixParallelComplex* GR_block;
        prepare_HG_matrices(momentum, i, H_blocks, diagonal_indices, offdiagonal_indices,
            gR_blocks, gL_blocks, left_coupling, gR_block, GL_block, gL_block, GR_block);

        PetscMatrixParallelComplex* HG = NULL;
        solve_HG(diagonal_indices[i], momentum, left_coupling, gR_block, GL_block, gL_block, GR_block, HG);
        delete left_coupling;
        left_coupling = NULL;
        delete HG;
        HG = NULL;

      }
      else if(use_explicit_blocks && i < subdomain_names.size()-2)
      {
        Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
        Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);
        PetscMatrixParallelComplexContainer* GL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);
        const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GL_blocks =
            GL_container->get_const_container_blocks();
        PetscMatrixParallelComplex* HG = NULL;
        solve_nonlocal_HG(diagonal_indices,momentum,i,H_blocks,GL_blocks,HG);
        delete HG;
        HG = NULL;
        /*//NYI
        //TODO:JC
        PetscMatrixParallelComplex* left_coupling;
        PetscMatrixParallelComplex* gR_block;
        PetscMatrixParallelComplex* GL_block;
        PetscMatrixParallelComplex* gL_block;
        PetscMatrixParallelComplex* GR_block;
        prepare_HG_matrices(momentum, i, H_blocks, diagonal_indices, offdiagonal_indices,
            gR_blocks, gL_blocks, left_coupling, gR_block, GL_block, gL_block, GR_block);

        PetscMatrixParallelComplex* HG = NULL;
        solve_HG(diagonal_indices[i], momentum, left_coupling, gR_block, GL_block, gL_block, GR_block, HG);
        delete left_coupling;
        left_coupling = NULL;
        delete HG;
        HG = NULL;
*/
      }

    }
    //after done with gR delete it
    gR_container->delete_submatrix(diagonal_indices[i].first, diagonal_indices[i].second);
    gL_container->delete_submatrix(diagonal_indices[i].first, diagonal_indices[i].second);
    Hamiltonian_container->delete_submatrix(diagonal_indices[i].first,diagonal_indices[i].second);
    Hamiltonian_container->delete_submatrix(offdiagonal_indices[i].first,offdiagonal_indices[i].second);

    //delete for the nonlocal case
    //gR gL diagonal and off diagonal

    if(use_explicit_blocks)
    {
      //L container
      int start_idx = i + number_of_offdiagonal_blocks;
      int end_idx = i + 2*number_of_offdiagonal_blocks;
      for (int k = start_idx;k <  end_idx; ++k)
      {
        PetscMatrixParallelComplex* temp_matrix;
        Simulation* temp_simulation=dynamic_cast<Simulation*>(forward_solver);
        temp_simulation->get_data(std::string("L_container"), &momentum, temp_matrix);
        PetscMatrixParallelComplexContainer* L_block_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix);
        NEMO_ASSERT(L_block_container!=NULL,
            "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum L_container is not a container to delete\n");
        if (end_idx < subdomain_names.size() -1 )
        {
          L_block_container->delete_submatrix(diagonal_indices[end_idx].first, diagonal_indices[k].second);
        }

      }

      //gR, gL
      start_idx = i + 1;
      end_idx = i + number_of_offdiagonal_blocks;
      for(int k = start_idx; k <= end_idx; k++)
      {
        if(i < subdomain_names.size()-2 && k < subdomain_names.size()-2)
        {
          gR_container->delete_submatrix(diagonal_indices[i].first, diagonal_indices[k].second);
          gL_container->delete_submatrix(diagonal_indices[k].first, diagonal_indices[i].second);
        }
      }
    }
  }

  //TODO JC: refactor these conditionals
  set_job_done_momentum_map(&name_of_writeable_Propagator, &momentum, true);
  if(!use_explicit_blocks&& use_diagonal_only)
  {
   //get diagonal and only store the diagonal
    Propagator::PropagatorMap::iterator prop_it=writeable_Propagator->propagator_map.find(momentum);
    NEMO_ASSERT(prop_it!=writeable_Propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
    PetscMatrixParallelComplex* Matrix = NULL;
    prop_it->second->get_diagonal_matrix(Matrix);
    delete prop_it->second;
    Matrix->assemble();
    if(check_inversion_stability)
    {
      std::map<vector<NemoMeshPoint>, int>::iterator stable_it = Ek_stable_inversion.find(momentum);
      if(stable_it!=Ek_stable_inversion.end())
      {
        //cerr << " deleting for Gr\n";
        Matrix->set_to_zero();
        Matrix->assemble();
      }
    }
    prop_it->second=Matrix;


  }
  else if(store_atom_blockdiagonal)
  {
    Propagator::PropagatorMap::iterator prop_it=writeable_Propagator->propagator_map.find(momentum);
    NEMO_ASSERT(prop_it!=writeable_Propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
    PetscMatrixParallelComplex* Matrix = NULL;
    get_atom_blockdiagonal_matrix(prop_it->second,Matrix);
    delete prop_it->second;
    Matrix->assemble();
    prop_it->second = Matrix;
    //Matrix->save_to_matlab_file("Gr_atomblock.m");

  }
  else if(use_explicit_blocks)
  {
    Propagator::PropagatorMap::iterator prop_it=writeable_Propagator->propagator_map.find(momentum);
    NEMO_ASSERT(prop_it!=writeable_Propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
    if(check_inversion_stability)
    {
      std::map<vector<NemoMeshPoint>, int>::iterator stable_it = Ek_stable_inversion.find(momentum);
      if(stable_it!=Ek_stable_inversion.end())
      {
        //cerr << " deleting for Gr\n";
        prop_it->second->set_to_zero();
      }
    }
  }
  set_job_done_momentum_map(&lesser_green_propagator_name, &momentum, true);
  if(!use_explicit_blocks && use_diagonal_only)//TODO:JC
  {
    //get diagonal and only store the diagonal
    Propagator::PropagatorMap::iterator prop_it=lesser_green_propagator->propagator_map.find(momentum);
    NEMO_ASSERT(prop_it!=lesser_green_propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
    PetscMatrixParallelComplex* Matrix = NULL;
    prop_it->second->get_diagonal_matrix(Matrix);
    delete prop_it->second;
    Matrix->assemble();
    if(check_inversion_stability)
    {
      std::map<vector<NemoMeshPoint>, int>::iterator stable_it = Ek_stable_inversion.find(momentum);
      if(stable_it!=Ek_stable_inversion.end())
      {
        Matrix->set_to_zero();
        Matrix->assemble();
      }
    }
    prop_it->second=Matrix;


  }
  else if(store_atom_blockdiagonal)
  {
    Propagator::PropagatorMap::iterator prop_it=lesser_green_propagator->propagator_map.find(momentum);
    NEMO_ASSERT(prop_it!=lesser_green_propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
    PetscMatrixParallelComplex* Matrix = NULL;
    get_atom_blockdiagonal_matrix(prop_it->second,Matrix);
    delete prop_it->second;
    Matrix->assemble();
    prop_it->second = Matrix;
    //Matrix->save_to_matlab_file("Gl_atomblock.m");

  }
  else if(use_explicit_blocks)
  {
    Propagator::PropagatorMap::iterator prop_it=lesser_green_propagator->propagator_map.find(momentum);
    NEMO_ASSERT(prop_it!=lesser_green_propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
    if(check_inversion_stability)
    {
      std::map<vector<NemoMeshPoint>, int>::iterator stable_it = Ek_stable_inversion.find(momentum);
      if(stable_it!=Ek_stable_inversion.end())
      {
        //cerr << " deleting for Gr\n";
        prop_it->second->set_to_zero();
      }
    }
  }

  if(solve_greater_greens_function)
  {
   set_job_done_momentum_map(&greater_green_propagator_name, &momentum, true);
  }

  //if (do_transform_HG && do_solve_HG)
  if(do_solve_HG)
  {
    //transform_data_to_atoms(Hlesser_green_propagator,momentum);
    //
    //std::vector < std::pair<std::vector<int>, std::vector<int> > > diagonal_indices;

    if(check_inversion_stability)
    {
      Propagator::PropagatorMap::iterator prop_it=Hlesser_green_propagator->propagator_map.find(momentum);
      NEMO_ASSERT(prop_it!=Hlesser_green_propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");

      std::map<vector<NemoMeshPoint>, int>::iterator stable_it = Ek_stable_inversion.find(momentum);
      if(stable_it!=Ek_stable_inversion.end())
      {
        //prop_it->second->set_to_zero();
        //prop_it->second->assemble();
      }
    }
    transform_current_to_slabs(Hlesser_green_propagator,momentum, diagonal_indices);
  }
  set_job_done_momentum_map(&Hlesser_green_propagator_name, &momentum, true);
  delete Hamiltonian;
  Hamiltonian = NULL;

  if(debug_output)
  {
    Propagator::PropagatorMap::iterator prop_it=writeable_Propagator->propagator_map.find(momentum);
    NEMO_ASSERT(prop_it!=writeable_Propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
    prop_it->second->save_to_matlab_file("Gr_full_preassemble.m");
    prop_it->second->assemble();
    prop_it->second->save_to_matlab_file("Gr_full_afterassemble.m");

    prop_it=lesser_green_propagator->propagator_map.find(momentum);
     NEMO_ASSERT(prop_it!=lesser_green_propagator->propagator_map.end(),tic_toc_prefix+"have not found the matrix for this momentum\n");
     prop_it->second->save_to_matlab_file("Gl_full_preassemble.m");
     prop_it->second->assemble();
     prop_it->second->save_to_matlab_file("Gl_full_afterassemble.m");

  }

  //delete Sigma_contact;
  //Sigma_contact = NULL;
  //delete Sigma_lesser_contact;
  //Sigma_lesser_contact = NULL;

  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::do_solve_retarded(Propagator*& /*output_Propagator*/,
    const std::vector<NemoMeshPoint>& momentum_point, PetscMatrixParallelComplex*& result)
{
  //Propagator* output_Propagator_temp = output_Propagator;
  //output_Propagator_temp = NULL;
  get_Greensfunction(momentum_point, result, NULL, NULL, NemoPhys::Fermion_retarded_Green);
}

void ScatteringBackwardRGFSolver::solve_last_diagonal_block_GR(const std::vector<NemoMeshPoint>& momentum,
    EOMMatrixInterface* eom_interface,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& H_blocks,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& offdiagonal_indices,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
    PetscMatrixParallelComplex* gR_lead, PetscMatrixParallelComplex*& exact_GR)
{
  std::string prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_last_diagonal_block_GR: ";

  PetscMatrixParallelComplex* Sigma_contact = NULL;
  get_contact_sigma(momentum, eom_interface, Sigma_contact, subdomain_names[subdomain_names.size() - 1], lead_domain, Hamiltonian, gR_lead);
  Sigma_contact->matrix_convert_dense();
  //NemoMath::symmetry_type type = NemoMath::symmetric;
  //PropagationUtilities::symmetrize(this, Sigma_contact,type);
  PetscMatrixParallelComplex* prelast_gR = NULL;

  //const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks =
  //    gR_container->get_const_container_blocks();

  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator gR_block_cit =
      gR_blocks.find(diagonal_indices[diagonal_indices.size() - 2]);
  NEMO_ASSERT(gR_block_cit != gR_blocks.end(),
      "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of gR\n");
  prelast_gR = gR_block_cit->second;

  Domain* temp_domain = Domain::get_domain(subdomain_names[subdomain_names.size() - 2]);
  PetscMatrixParallelComplex* Sigma_contact_prelast = NULL;

  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator H_block_cit =
      H_blocks.find(offdiagonal_indices[offdiagonal_indices.size() - 1]);
  NEMO_ASSERT(H_block_cit != H_blocks.end(),
      "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found Hamiltonian block\n");
  PetscMatrixParallelComplex* coupling = H_block_cit->second;
  //should direction be flipped here?
  if (!use_explicit_blocks)
    get_contact_sigma(momentum, eom_interface, Sigma_contact_prelast, subdomain_names[subdomain_names.size() - 1], temp_domain,
        Hamiltonian, prelast_gR, coupling);
  else
  {

    //get_contact_sigma(momentum,eom_interface, Sigma_contact_prelast, subdomain_names[subdomain_names.size()-2], lead_domain,
    //    Hamiltonian_indices, Hamiltonian, prelast_gR, coupling_Hamiltonian, sigmaR_off);


    PetscMatrixParallelComplex* temp_container = NULL;
    if (particle_type_is_Fermion)
      forward_solver->get_Greensfunction(momentum, temp_container, NULL, NULL, NemoPhys::Fermion_retarded_Green);
    else
      forward_solver->get_Greensfunction(momentum, temp_container, NULL, NULL, NemoPhys::Boson_retarded_Green);
    PetscMatrixParallelComplexContainer* gR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
    NEMO_ASSERT(gR_container != NULL,
        "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum gR is not received in the container format\n");


    PetscMatrixParallelComplex* temp_matrix;

    Simulation* temp_simulation=dynamic_cast<Simulation*>(forward_solver);
    //TOOD:JC replace get_data
    temp_simulation->get_data(std::string("L_container"), &momentum, temp_matrix);
    NEMO_ASSERT(temp_matrix!=NULL,
        "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum L_container is NULL from forward\n");
    PetscMatrixParallelComplexContainer* L_block_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix);
    NEMO_ASSERT(L_block_container!=NULL,
         "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum L_container is not a container \n");



    //get L_block_container from forward solver
    PropagationUtilities::get_contact_sigmaR_nonlocal(this,momentum,eom_interface,subdomain_names.size()-1,number_of_offdiagonal_blocks-1,subdomain_names,
        diagonal_indices, L_block_container, gR_container, Sigma_contact_prelast, scattering_sigmaR_solver);
    if(debug_output)
    {
      Sigma_contact_prelast->assemble();
      Sigma_contact_prelast->save_to_matlab_file("sigma_contact_prelast_nonlocal.m");

      //Sigma_contact->assemble();
      //Sigma_contact->save_to_matlab_file("sigma_contact_lead.m");
    }
    //NYI
  }
  //NemoMath::symmetry_type type = NemoMath::symmetric;
  //PropagationUtilities::symmetrize(this, Sigma_contact_prelast, type);
  Sigma_contact->add_matrix(*Sigma_contact_prelast, SAME_NONZERO_PATTERN, std::complex<double>(1.0, 0.0));
  delete Sigma_contact_prelast;
  Sigma_contact_prelast = NULL;

  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator
  block_cit = H_blocks.find(diagonal_indices[diagonal_indices.size() - 1]);
  NEMO_ASSERT(block_cit != H_blocks.end(),
      "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::solve last diagonal block gR have not found block Hamiltonian\n");
  PetscMatrixParallelComplex* diagonal_H = block_cit->second;//new PetscMatrixParallelComplex(*(block_cit->second));

  PetscMatrixParallelComplex* scattering_sigmaR_matrix = NULL;
  if (scattering_sigmaR_solver != NULL)
  {
    //PetscMatrixParallelComplex* temp_container = NULL;
    const DOFmapInterface& dof_map = Hamilton_Constructor->get_dof_map(
        Domain::get_domain(subdomain_names[subdomain_names.size() - 1]));
    SelfenergyInterface* selfenergy_interface = dynamic_cast<SelfenergyInterface*>(scattering_sigmaR_solver);
    NEMO_ASSERT(selfenergy_interface, prefix + "scattering sigmaR solver cannot be cast into class type SelfenergyInterface");
    NemoPhys::Propagator_type propagator_type;
    if (particle_type_is_Fermion)
      propagator_type = NemoPhys::Fermion_retarded_self;
    else
      propagator_type = NemoPhys::Boson_retarded_self;
    selfenergy_interface->get_Selfenergy(momentum, scattering_sigmaR_matrix, &dof_map, &dof_map, propagator_type);
    //PetscMatrixParallelComplexContainer* sigmaR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_container);
    //NEMO_ASSERT(sigmaR_container != NULL,
    //    "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_forward_RGF sigmaR is not a container \n");
    //const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& sigmaR_blocks =
    //    sigmaR_container->get_const_container_blocks();
    //std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator sigmaR_block_cit =
    //    sigmaR_blocks.find(diagonal_indices[diagonal_indices.size() - 1]);
  }

  //solve last subdomain gR which is last domain Gr
  //PetscMatrixParallelComplex *exact_GR = NULL;
  //if (!use_explicit_blocks)
  solve_diagonal_gR(diagonal_indices[diagonal_indices.size() - 1], momentum, diagonal_H, Sigma_contact, exact_GR,
                    scattering_sigmaR_matrix);

  delete Sigma_contact;
  Sigma_contact = NULL;
}

void ScatteringBackwardRGFSolver::solve_last_diagonal_block_GL(const std::vector<NemoMeshPoint>& momentum,
     EOMMatrixInterface* eom_interface,
     const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& H_blocks,
     std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
     std::vector < std::pair<std::vector<int>, std::vector<int> > >& offdiagonal_indices,
     const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gL_blocks,
     PetscMatrixParallelComplex *gR, PetscMatrixParallelComplex* exact_GR)
{
  std::string prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_last_diagonal_block_GL: ";

  //get prelast subdomain gL
    Domain* temp_domain = Domain::get_domain(subdomain_names[subdomain_names.size() - 2]);
    PetscMatrixParallelComplex *prelast_gL = NULL;

    std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator gL_block_cit =
        gL_blocks.find(diagonal_indices[diagonal_indices.size() - 2]);
    NEMO_ASSERT(gL_block_cit != gL_blocks.end(),
        "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of gR\n");
    prelast_gL = gL_block_cit->second;

    //get prelast subdomain sigmaL
    PetscMatrixParallelComplex* Sigma_lesser_contact_prelast = NULL;

    std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator H_block_cit =
        H_blocks.find(offdiagonal_indices[offdiagonal_indices.size() - 1]);
    NEMO_ASSERT(H_block_cit != H_blocks.end(),
        "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found Hamiltonian block\n");
    PetscMatrixParallelComplex* coupling = H_block_cit->second;

    if (!use_explicit_blocks)
    {
      get_contact_sigma(momentum, eom_interface, Sigma_lesser_contact_prelast, subdomain_names[subdomain_names.size() - 1],
          temp_domain, Hamiltonian, prelast_gL, coupling, NULL, lesser_green_propagator_type);

      NemoMath::symmetry_type type = NemoMath::antihermitian;
      symmetrize(Sigma_lesser_contact_prelast, type);
    }
    else
    {
      //NYI
      //TODO:JC
      //get_contact_sigma(momentum, eom_interface, Sigma_lesser_contact_prelast, subdomain_names[subdomain_names.size() - 1],
      //    temp_domain, Hamiltonian, prelast_gL, coupling, NULL, lesser_green_propagator_type);

      //NemoMath::symmetry_type type = NemoMath::antihermitian;
      //symmetrize(Sigma_lesser_contact_prelast, type);

    }

    //2.2 get the lesser contact_self_energy
     PetscMatrixParallelComplex* Sigma_lesser_contact = NULL;

     get_contact_sigma_lesser_equilibrium(momentum, eom_interface, Sigma_lesser_contact, subdomain_names[subdomain_names.size() - 1],
         lead_domain, Hamiltonian_indices, Hamiltonian, gR /*last_coupling*/);

     Sigma_lesser_contact->matrix_convert_dense();
     if(!use_explicit_blocks)
       Sigma_lesser_contact->add_matrix(*Sigma_lesser_contact_prelast, DIFFERENT_NONZERO_PATTERN, std::complex<double>(1.0, 0.0));
     delete Sigma_lesser_contact_prelast;
     Sigma_lesser_contact_prelast = NULL;
     //solve last subdomain g<
     PetscMatrixParallelComplex* scattering_sigmaL_matrix = NULL;
    if (scattering_sigmaL_solver != NULL)
    {
      PetscMatrixParallelComplex* temp_matrix = NULL;
      SelfenergyInterface* selfenergy_interface = dynamic_cast<SelfenergyInterface*>(scattering_sigmaL_solver);
      NEMO_ASSERT(selfenergy_interface, prefix + "scattering sigmaL solver cannot be cast into class type SelfenergyInterface");
      const DOFmapInterface& dof_map = Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain());
      NemoPhys::Propagator_type propagator_type;
      if (particle_type_is_Fermion)
        propagator_type = NemoPhys::Fermion_lesser_self;
      else
        propagator_type = NemoPhys::Boson_lesser_self;
      selfenergy_interface->get_Selfenergy(momentum, temp_matrix, &dof_map, &dof_map, propagator_type);
      scattering_sigmaL_matrix = new PetscMatrixParallelComplex(diagonal_indices[diagonal_indices.size() - 1].first.size(),
          diagonal_indices[diagonal_indices.size() - 1].second.size(), temp_matrix->get_communicator());

      temp_matrix->get_submatrix(diagonal_indices[diagonal_indices.size() - 1].first,
          diagonal_indices[diagonal_indices.size() - 1].second, MAT_INITIAL_MATRIX, scattering_sigmaL_matrix);
    }

    PetscMatrixParallelComplex* exact_GL = NULL;
    if (!use_explicit_blocks)
    {
      bool real_lead = false;//true;
      solve_diagonal_gL(diagonal_indices[diagonal_indices.size() - 1], momentum, Sigma_lesser_contact, exact_GR, exact_GL, real_lead,
          Sigma_lesser_contact_prelast, scattering_sigmaL_matrix);

      //exact_GL->save_to_matlab_file("forward_gL_"+subdomain_names[diagonal_indices.size() - 1]+".m");
      delete Sigma_lesser_contact;
      Sigma_lesser_contact = NULL;
      delete Sigma_lesser_contact_prelast;
      Sigma_lesser_contact_prelast = NULL;
      delete scattering_sigmaL_matrix;
      scattering_sigmaL_matrix = NULL;
    }
    else
    {
      //NYI
      //TODO:JC
/*
      //off diagonal g<
      int i = subdomain_names.size()-1;

      //solve off diagonal gL
      Propagator::PropagatorMap& result_prop_map_lesser = lesser_green_propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it_lesser = result_prop_map_lesser.find(momentum);

      PetscMatrixParallelComplexContainer* gL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it_lesser->second);

      Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
      Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

      PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
          GR_container->get_const_container_blocks();

      std::pair<std::vector<int>, std::vector<int> > temp_pair2(diagonal_indices[i].first,diagonal_indices[i].second);
      std::map< std::pair < std::vector<int>, std::vector<int> >,
      PetscMatrixParallelComplex* >::const_iterator GRblock_cit=GR_blocks.find(temp_pair2);
      PetscMatrixParallelComplex* GR_row_row = GRblock_cit->second;

      const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks =
          L_container->get_const_container_blocks();




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



    solve_diagonal_gL(diagonal_indices[diagonal_indices.size() - 1], momentum, Sigma_lesser_contact, exact_GR, exact_GL,
        scattering_sigmaL_matrix);
    delete Sigma_lesser_contact;
    Sigma_lesser_contact = NULL;
    delete scattering_sigmaL_matrix;
    scattering_sigmaL_matrix = NULL;
*/
    }
    delete exact_GL;
    exact_GL = NULL;


}

void ScatteringBackwardRGFSolver::solve_diagonal_gR(std::pair<std::vector<int>, std::vector<int> >& diagonal_index,
    const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& diagonal_H,
    PetscMatrixParallelComplex*& Sigma_contact, PetscMatrixParallelComplex*& block_gR,
    PetscMatrixParallelComplex* scattering_sigmaR_matrix)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_gR: ";
  NemoUtils::tic(tic_toc_prefix);

  //sigma is dense
  diagonal_H->matrix_convert_dense();
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

  //test
  //PetscMatrixParallelComplex* inverse_Green = new PetscMatrixParallelComplex(* Sigma_contact);
  //inverse_Green->assemble();
  //std::string invert_solver = "petsc";//lapack";
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
      identity.matrix_diagonal_shift(cplx(1.0,0.0));//identity
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

  //4. store results in PMPCC
  //if needed create a PetscMatrixParallelComplexContainer

  Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    writeable_Propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*block_gR, diagonal_index.first, diagonal_index.second);
  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::prepare_diagonal_GL_matrices(const std::vector<NemoMeshPoint>& momentum, unsigned int i,
    EOMMatrixInterface* eom_interface,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& H_blocks,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& offdiagonal_indices,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gL_blocks,
    PetscMatrixParallelComplex* exact_GR,
    PetscMatrixParallelComplex*& Sigma_lesser_contact, PetscMatrixParallelComplex*& Sigma_retarded_contact,
    PetscMatrixParallelComplex*& gL_block)
{

  Domain* previous_lead_domain = Domain::get_domain(subdomain_names[i+1]);

  //get sigmaL = t'*G<(i+1,i+1)*t
  //PetscMatrixParallelComplex* Sigma_lesser_contact = NULL;
  Propagator::PropagatorMap& result_prop_map = lesser_green_propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

  PetscMatrixParallelComplexContainer* GL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GL_blocks =
      GL_container->get_const_container_blocks();
  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GL_block_cit =
      GL_blocks.find(diagonal_indices[i + 1]);
  PetscMatrixParallelComplex* exact_GL = GL_block_cit->second;

  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator off_block_cit =
      H_blocks.find(offdiagonal_indices[i]);
  PetscMatrixParallelComplex left_coupling(off_block_cit->second->get_num_cols(), off_block_cit->second->get_num_rows(),
      off_block_cit->second->get_communicator());
  off_block_cit->second->hermitian_transpose_matrix(left_coupling, MAT_INITIAL_MATRIX);
  //left_coupling.save_to_matlab_file("coupling_"+subdomain_names[i]+".m");
  get_contact_sigma(momentum, eom_interface, Sigma_lesser_contact, subdomain_names[i+1], previous_lead_domain ,
      Hamiltonian, exact_GL, &left_coupling, NULL, lesser_green_propagator_type);
  NemoMath::symmetry_type type = NemoMath::antihermitian;
  symmetrize(Sigma_lesser_contact, type);

  //get sigmaR = t'*Gr(i+1,i+1)*t
      //PetscMatrixParallelComplex* Sigma_retarded_contact = NULL;
  get_contact_sigma(momentum, eom_interface, Sigma_retarded_contact, subdomain_names[i+1], previous_lead_domain ,
      Hamiltonian, exact_GR, &left_coupling, NULL);
  //get gL
  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator gL_block_cit = gL_blocks.find(diagonal_indices[i]);
  NEMO_ASSERT(gL_block_cit != gL_blocks.end(),
      "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of gL\n");
  gL_block = gL_block_cit->second;

}

void ScatteringBackwardRGFSolver::solve_diagonal_gL(std::pair<std::vector<int>, std::vector<int> >& diagonal_index,
    const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex*& Sigma_lesser_contact,
    PetscMatrixParallelComplex*& diagonal_block_gR, PetscMatrixParallelComplex*& diagonal_block_gL, bool real_lead,
    PetscMatrixParallelComplex*& Sigma_contact_lesser_prelast, PetscMatrixParallelComplex* scattering_sigmaL_matrix)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_gL: ";
  NemoUtils::tic(tic_toc_prefix);

  //1. call core routine for the g<

  if (scattering_sigmaL_matrix != NULL)// && !real_lead)
  {
    Sigma_lesser_contact->matrix_convert_dense();
    scattering_sigmaL_matrix->matrix_convert_dense();
    Sigma_lesser_contact->add_matrix(*scattering_sigmaL_matrix, DIFFERENT_NONZERO_PATTERN, std::complex<double>(1.0, 0.0));
  }

  PropagationUtilities::core_correlation_Green_exact(Sigma_lesser_contact, diagonal_block_gL, diagonal_block_gR);

  //diagonal_block_gL->save_to_matlab_file("gL.m");

  //2. store in container for g<
  Propagator::PropagatorMap& result_prop_map = lesser_green_propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    lesser_green_propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  NemoMath::symmetry_type type = NemoMath::antihermitian;
  symmetrize(diagonal_block_gL, type);
  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*diagonal_block_gL, diagonal_index.first, diagonal_index.second);

  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::solve_diagonal_GR(std::pair<std::vector<int>, std::vector<int> >& diagonal_index,
    const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex* exact_GR, PetscMatrixParallelComplex* gR,
    PetscMatrixParallelComplex* coupling, PetscMatrixParallelComplex*& this_GR)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_GR: ";
  NemoUtils::tic(tic_toc_prefix);

  PropagationUtilities::core_retarded_Green_back_RGF(this, momentum, exact_GR, gR, coupling, this_GR);
  //NemoMath::symmetry_type type = NemoMath::symmetric;
  //PropagationUtilities::symmetrize(this, exact_GR, type);
  //store in container
  Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    writeable_Propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*this_GR, diagonal_index.first, diagonal_index.second);
  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::solve_diagonal_GL(std::pair<std::vector<int>, std::vector<int> >& diagonal_index,
    const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex* Sigma_lesser_contact,
    PetscMatrixParallelComplex* Sigma_retarded_contact, PetscMatrixParallelComplex* gR, PetscMatrixParallelComplex* gL,
    PetscMatrixParallelComplex*& this_GL_block)
{

  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_GL: ";
  NemoUtils::tic(tic_toc_prefix);

  //call core function
  NemoUtils::tic(tic_toc_prefix + " core");
  Sigma_lesser_contact->matrix_convert_dense();
  Sigma_retarded_contact->matrix_convert_dense();

  NemoMatrixInterface *NMI_this_GL_block = NULL;
  NemoMatrixInterface *NMI_Sigma_lesser_contact =
      Sigma_lesser_contact->convert_to_NMI(DEFAULT_MAT_TYPE);
  NemoMatrixInterface *NMI_Sigma_retarded_contact =
      Sigma_retarded_contact->convert_to_NMI(DEFAULT_MAT_TYPE);
  NemoMatrixInterface *NMI_gL = gL->convert_to_NMI(DEFAULT_MAT_TYPE);
  NemoMatrixInterface *NMI_gR = gR->convert_to_NMI(DEFAULT_MAT_TYPE);
  PropagationUtilities::core_correlation_Green_RGF2(this,
      NMI_Sigma_lesser_contact, NMI_Sigma_retarded_contact, NMI_gL, NMI_gR,
      NMI_this_GL_block, true);
  this_GL_block =
      dynamic_cast<PetscMatrixParallelComplex *>(NMI_this_GL_block->convert_to_PMPC());
  NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE, NMI_this_GL_block);
  NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE,
      NMI_Sigma_lesser_contact);
  NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE,
      NMI_Sigma_retarded_contact);
  NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE, NMI_gL);
  NemoMatrixInterface::clean_temp_mat(DEFAULT_MAT_TYPE, NMI_gR);


  NemoUtils::toc(tic_toc_prefix + " core");

  //store in container
  NemoUtils::tic(tic_toc_prefix + " store container");

  Propagator::PropagatorMap& result_prop_map = lesser_green_propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    lesser_green_propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  NemoMath::symmetry_type type = NemoMath::antihermitian;
  symmetrize(this_GL_block, type);

  

  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*this_GL_block, diagonal_index.first, diagonal_index.second);
  NemoUtils::toc(tic_toc_prefix + " store container");

  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::solve_diagonal_GG(std::pair<std::vector<int>,std::vector<int> >& diagonal_index, const std::vector<NemoMeshPoint>& momentum,
    PetscMatrixParallelComplex* exact_GR, PetscMatrixParallelComplex* this_GL_block, PetscMatrixParallelComplex*& this_GG_block)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_GG: ";
  NemoUtils::tic(tic_toc_prefix);

  //Gr-Ga = G> - G<
  //G> = Gr-Ga + G<
  PetscMatrixParallelComplex *temp_result = new PetscMatrixParallelComplex(*exact_GR);
  //this_GG_block->imaginary_part();
//*this_GG_block*=cplx(0.0,1.0);
  //PetscMatrixParallelComplex exact_GA(temp_result->get_num_cols(), temp_result->get_num_rows(),
  //    temp_result->get_communicator());
  //exact_GR->hermitian_transpose_matrix(exact_GA, MAT_INITIAL_MATRIX);

  //temp_result->add_matrix(exact_GA,SAME_NONZERO_PATTERN,cplx(-1.0,0.0));
  temp_result->imaginary_part();
  *temp_result *= cplx(0.0,2.0);

  //Gr-Ga+G<
  if(!explicitly_solve_greater_greens_function)
    temp_result->add_matrix(*this_GL_block,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
  else
  {
    //-G<= Gr-Ga-G>
    temp_result->add_matrix(*this_GL_block,SAME_NONZERO_PATTERN,cplx(-1.0,0.0));
    *temp_result *= cplx(-1.0,0.0);
  }

  if(use_diagonal_only)
  {
    temp_result->get_diagonal_matrix(this_GG_block);
    delete temp_result;
  }
  else
    this_GG_block = temp_result;
  temp_result = NULL;
  Propagator::PropagatorMap& result_prop_map = greater_green_propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    greater_green_propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  //NemoMath::symmetry_type type = NemoMath::antihermitian;
  //symmetrize(this_GG_block, type);
  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*this_GG_block, diagonal_index.first, diagonal_index.second);

  NemoUtils::toc(tic_toc_prefix);

}

//
//void ScatteringBackwardRGFSolver::do_reinit()
//{
//  // TODO : Clean variables, reinitialize external libraries, etc ...
//}

void ScatteringBackwardRGFSolver::set_writeable_propagator()
{
  std::string this_domain = get_const_simulation_domain()->get_name();
  NemoPhys::Propagator_type temp_type;
  if (particle_type_is_Fermion)
  {
    name_of_writeable_Propagator = get_name() + "_combined_retarded_Green_Fermion_rgf_"+this_domain;
    Propagator_types[name_of_writeable_Propagator] = NemoPhys::Fermion_retarded_Green;
    temp_type = NemoPhys::Fermion_retarded_Green;
  }
  else
  {
    name_of_writeable_Propagator = get_name() + "_combined_retarded_Green_Boson_rgf_"+this_domain;
    Propagator_types[name_of_writeable_Propagator] = NemoPhys::Boson_retarded_Green;
    temp_type = NemoPhys::Boson_retarded_Green;
  }
  delete writeable_Propagator;
  Propagator* temp_propagator = new Propagator(name_of_writeable_Propagator, temp_type);
  Propagators[name_of_writeable_Propagator] = temp_propagator;
  writeable_Propagator = temp_propagator;

  this_is_combine_Propagation = false;

  ready_Propagator_map[name_of_writeable_Propagator] = false;
  Propagator_Constructors[name_of_writeable_Propagator] = this;

  type_of_writeable_Propagator = NemoPhys::Inverse_Green;

  //additionally need to create and store propagator for g<
  if (particle_type_is_Fermion)
  {
    lesser_green_propagator_name = get_name() + "_combined_lesser_Green_Fermion_rgf_"+this_domain;
    lesser_green_propagator_type = NemoPhys::Fermion_lesser_Green;
  }
  else
  {
    lesser_green_propagator_name = get_name() + "_combined_lesser_Green_Boson_rgf_"+this_domain;
    lesser_green_propagator_type = NemoPhys::Boson_lesser_Green;
  }
  delete lesser_green_propagator;
  lesser_green_propagator = new Propagator(lesser_green_propagator_name, lesser_green_propagator_type);
  lesser_propagator_ready_map[lesser_green_propagator_type] = false;
  //Propagator_Constructors[lesser_green_propagator_name]=this;

  if(electron_hole_model)
  {
    if (particle_type_is_Fermion)
    {
      greater_green_propagator_name = get_name() + "_combined_greater_Green_Fermion_rgf_"+this_domain;
      greater_green_propagator_type = NemoPhys::Fermion_greater_Green;
    }
    else
    {
      greater_green_propagator_name = get_name() + "_combined_greater_Green_Boson_rgf_"+this_domain;
      greater_green_propagator_type = NemoPhys::Boson_greater_Green;
    }
    delete greater_green_propagator;
    greater_green_propagator = new Propagator(greater_green_propagator_name, greater_green_propagator_type);
    greater_propagator_ready_map[greater_green_propagator_type] = false;
  }

  //additionally need to create and store propagator for HG<
  //combined_product_lesser_Green
  if (particle_type_is_Fermion)
  {
    Hlesser_green_propagator_name = get_name() + "_combined_product_lesser_Green_Fermion_rgf_"+this_domain;
    Hlesser_green_propagator_type = NemoPhys::Fermion_lesser_HGreen;
  }
  else
  {
    Hlesser_green_propagator_name = get_name() + "_product_lesser_Green_Boson_rgf_"+this_domain;
    Hlesser_green_propagator_type = NemoPhys::Boson_lesser_HGreen;
  }
  delete Hlesser_green_propagator;
  Hlesser_green_propagator = new Propagator(Hlesser_green_propagator_name, Hlesser_green_propagator_type);
  Hlesser_propagator_ready_map[Hlesser_green_propagator_type] = false;
  //Propagator_Constructors[Hlesser_green_propagator_name]=this;

}

void ScatteringBackwardRGFSolver::delete_propagator_matrices(Propagator* input_Propagator,
    const std::vector<NemoMeshPoint>* momentum, const DOFmapInterface* row_DOFmap, const DOFmapInterface* col_DOFmap)
{

  if (input_Propagator == NULL || input_Propagator->get_propagator_type() == writeable_Propagator->get_propagator_type())
  {
    Propagation::delete_propagator_matrices(input_Propagator, momentum, row_DOFmap, col_DOFmap);
    delete_propagator_matrices(lesser_green_propagator, momentum, row_DOFmap, col_DOFmap);
    delete_propagator_matrices(Hlesser_green_propagator, momentum, row_DOFmap, col_DOFmap);
  }
  //delete_propagator_matrices can be called twice if this is NULL so only really delete the lesser propagator on the 2nd call
  else if (input_Propagator->get_propagator_type() == lesser_green_propagator_type
  || input_Propagator->get_propagator_type()==Hlesser_green_propagator_type ||
  (solve_greater_greens_function && input_Propagator->get_propagator_type() == greater_green_propagator_type) )
  {
    Propagator *temp_propagator;
    if (input_Propagator->get_propagator_type() == lesser_green_propagator_type)
      temp_propagator = lesser_green_propagator;
    else if(input_Propagator->get_propagator_type() == Hlesser_green_propagator_type)
      temp_propagator = Hlesser_green_propagator;
    else
      temp_propagator = greater_green_propagator;

    if (momentum != NULL)
    {
      std::map<const std::vector<NemoMeshPoint>, bool>::iterator it = temp_propagator->allocated_momentum_Propagator_map.find(
          *momentum);
      NEMO_ASSERT(it != temp_propagator->allocated_momentum_Propagator_map.end(),
          "Propagation(\"" + get_name() + "\")::delete_propagator_matrices have not found momentum in \""
              + temp_propagator->get_name() + "\"\n");
      if (it->second)
      {
        Propagator::PropagatorMap::iterator it2 = temp_propagator->propagator_map.find(*momentum);
        NEMO_ASSERT(it2 != temp_propagator->propagator_map.end(),
            "Propagation(\"" + get_name() + "\")::delete_propagator_matrices have not found momentum in propagator_map of \""
                + temp_propagator->get_name() + "\"\n");
        if (it2->second != NULL)
        {
          msg << "ScatteringBackwardRGFSolver(\"" << get_name() << "\")::delete_propagator_matrices: " << it2->second << "\n";
          if (/*input_Propagator->get_name().find("combine")!=std::string::npos &&*/row_DOFmap != NULL)
          {

            const DOFmapInterface& large_DOFmap = Hamilton_Constructor->get_dof_map(get_const_simulation_domain());
            std::vector<int> rows_to_delete;

            Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap, *row_DOFmap, rows_to_delete,
                get_const_simulation_domain());

            std::vector<int> cols_to_delete;
            if (col_DOFmap != NULL)
            {
              Hamilton_Constructor->translate_subDOFmap_into_int(large_DOFmap, *col_DOFmap, cols_to_delete,
                  get_const_simulation_domain());
            }
            else
              cols_to_delete = rows_to_delete;

            PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(it2->second);
            if (temp_container != NULL)
              temp_container->delete_submatrix(rows_to_delete, cols_to_delete);
            else
            {
              delete it2->second;
              it2->second = NULL;
              it->second = false;
            }
            //since this is combine solver, need to quit the function inorder not to mess up with job_done_map etc
            return void();
          }
          else
          {
            delete it2->second;
            it2->second = NULL;
          }
          temp_propagator->propagator_map[it2->first] = NULL;
        }
        it->second = false;

      }
      else
      {
      }
      set_job_done_momentum_map(&(temp_propagator->get_name()), momentum, false);
    }
    else
    {
      msg << "ScatteringBackwardRGFSolver(\"" << get_name() << "\")::delete_propagator_matrices: " << "deleting Propagator \""
          << temp_propagator->get_name() << "\"\n";
      Propagator::PropagatorMap::iterator it2 = temp_propagator->propagator_map.begin();
      for (; it2 != temp_propagator->propagator_map.end(); ++it2)
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

void ScatteringBackwardRGFSolver::set_job_done_momentum_map(const std::string* Propagator_name,
    const std::vector<NemoMeshPoint>* momentum_point, const bool input_status)
{
  if (Propagator_name == NULL || *Propagator_name == writeable_Propagator->get_name())
    Propagation::set_job_done_momentum_map(Propagator_name, momentum_point, input_status);
  else if (*Propagator_name == lesser_green_propagator_name)
  {
    //bool always_ready=options.get_option("always_ready",false);
    bool input_status2 = always_ready || input_status; //DM:boolean operation is faster..
    {
      if (momentum_point != NULL)
      {
        //it->second[*momentum_point]=input_status2; //Normally this is faster
        lesser_propagator_job_done_momentum_map[*momentum_point] = input_status2;
      }
      else
      {
        std::map<std::vector<NemoMeshPoint>, bool>::iterator it2 = lesser_propagator_job_done_momentum_map.begin();
        for (; it2 != lesser_propagator_job_done_momentum_map.end(); it2++)
          it2->second = input_status2;
      }

    }
  }
  else if (*Propagator_name == Hlesser_green_propagator_name)
  {
    //bool always_ready=options.get_option("always_ready",false);
    bool input_status2 = always_ready || input_status; //DM:boolean operation is faster..
    {
      if (momentum_point != NULL)
      {
        //it->second[*momentum_point]=input_status2; //Normally this is faster
        Hlesser_propagator_job_done_momentum_map[*momentum_point] = input_status2;
      }
      else
      {
        std::map<std::vector<NemoMeshPoint>, bool>::iterator it2 = Hlesser_propagator_job_done_momentum_map.begin();
        for (; it2 != Hlesser_propagator_job_done_momentum_map.end(); it2++)
          it2->second = input_status2;
      }

    }
  }
  else if (solve_greater_greens_function && *Propagator_name == greater_green_propagator_name)
  {
    bool input_status2 = always_ready || input_status; //DM:boolean operation is faster..
    {
      if (momentum_point != NULL)
      {
        //it->second[*momentum_point]=input_status2; //Normally this is faster
        greater_propagator_job_done_momentum_map[*momentum_point] = input_status2;
      }
      else
      {
        std::map<std::vector<NemoMeshPoint>, bool>::iterator it2 = greater_propagator_job_done_momentum_map.begin();
        for (; it2 != greater_propagator_job_done_momentum_map.end(); it2++)
          it2->second = input_status2;
      }

    }
  }

}

void ScatteringBackwardRGFSolver::get_data(const std::string& variable, Propagator*& Propagator_pointer)
{
  /*std::map<std::string, bool>::const_iterator temp_cit=Propagator_is_initialized.find(writeable_Propagator->get_name());
   //NEMO_ASSERT(temp_cit!=Propagator_is_initialized.end(),get_name() + " variable, Propagator_pointer could not find writeable_Propagator");
   bool propagator_initialized = false;
   if(temp_cit->second)
   propagator_initialized = true;
   if(!propagator_initialized)
   {
   Propagation::get_data(variable, Propagator_pointer);
   Propagator_pointer = writeable_Propagator;
   */
  //}
  //else
  /*std::map<NemoPhys::Propagator_type, bool>::const_iterator temp_cit=propagator_initialized_map.find(Propagator_pointer->get_propagator_type());
   bool propagator_initialized = false;
   if(temp_cit->second)
   propagator_initialized = true;
   if(!propagator_initialized)
   initialize_Propagator(Propagator_pointer);
   */

  {
    //get_data(const std::string& variable, Propagator*& Propagator_pointer)
    if (variable.find("retarded") != std::string::npos)
    {
      Propagator_pointer = writeable_Propagator;
      std::map<NemoPhys::Propagator_type, bool>::const_iterator temp_cit = propagator_initialized_map.find(
          Propagator_pointer->get_propagator_type());
      bool propagator_initialized = false;
      if (temp_cit->second)
        propagator_initialized = true;
      if (!propagator_initialized)
        initialize_Propagator(Propagator_pointer);

      Propagation::get_data(variable, Propagator_pointer);
      propagator_initialized_map[writeable_Propagator->get_propagator_type()] = true;
    }
    else if (variable.find("product") != std::string::npos)
    {
      Propagator_pointer = Hlesser_green_propagator;
      std::map<NemoPhys::Propagator_type, bool>::const_iterator temp_cit = propagator_initialized_map.find(
          Propagator_pointer->get_propagator_type());
      bool propagator_initialized = false;
      if (temp_cit->second)
        propagator_initialized = true;
      if (!propagator_initialized)
        initialize_Propagator(Propagator_pointer);

      propagator_initialized_map[Hlesser_green_propagator_type] = true;
    }
    else if (variable.find("lesser") != std::string::npos)
    {
      Propagator_pointer = lesser_green_propagator;
      std::map<NemoPhys::Propagator_type, bool>::const_iterator temp_cit = propagator_initialized_map.find(
          Propagator_pointer->get_propagator_type());
      bool propagator_initialized = false;
      if (temp_cit->second)
        propagator_initialized = true;
      if (!propagator_initialized)
        initialize_Propagator(Propagator_pointer);

      propagator_initialized_map[lesser_green_propagator_type] = true;

    }
    else if (variable.find("greater") != std::string::npos)
    {
      Propagator_pointer = greater_green_propagator;
      std::map<NemoPhys::Propagator_type, bool>::const_iterator temp_cit = propagator_initialized_map.find(
          Propagator_pointer->get_propagator_type());
      bool propagator_initialized = false;
      if (temp_cit->second)
        propagator_initialized = true;
      if (!propagator_initialized)
        initialize_Propagator(Propagator_pointer);

      propagator_initialized_map[greater_green_propagator_type] = true;
    }
    else
      throw std::runtime_error("ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_data propagator name unknown variable ");
  }

}

void ScatteringBackwardRGFSolver::get_Propagator(Propagator*& output_propagator, const NemoPhys::Propagator_type* type)
{
  if(writeable_Propagator==NULL)
      initialize_Propagation();

 if(type==NULL || *type==writeable_Propagator->get_propagator_type())
   output_propagator=writeable_Propagator;
 else if(*type==lesser_green_propagator_type)
   output_propagator=lesser_green_propagator;
 else if(*type==Hlesser_green_propagator_type)
   output_propagator=Hlesser_green_propagator;
 else if(solve_greater_greens_function && *type == greater_green_propagator_type)
   output_propagator = greater_green_propagator;
 else
   throw std::runtime_error("ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_Propagator called with unknown propagator type ");


}
void ScatteringBackwardRGFSolver::get_data(const std::string& variable, std::map<unsigned int, double>& data)
{
  std::string prefix = get_name() + " get_data std map unsigned int double ";
  //force init subsolvers of BSM
  std::vector<string> temp_domains;
  Hamilton_Constructor->get_data("subdomains", temp_domains);

  if (variable == "electron_density" || variable == "free_charge")
  {
    if (electron_density.empty())
    {
      calculate_density(lesser_green_propagator, &electron_density, electron_hole_model);
    }
    for (std::map<unsigned int, double>::const_iterator it = electron_density.begin();
        it != electron_density.end(); it++)
    {
      if(!explicitly_solve_greater_greens_function)
        data[it->first] = - it->second;
      else
        data[it->first] = + it->second;
    }
    if(electron_hole_model)//&&hole_density.empty())
    {
      if(hole_density.empty())
        calculate_density(greater_green_propagator, &hole_density, electron_hole_model);

      for (std::map<unsigned int, double>::const_iterator it = hole_density.begin();
          it != hole_density.end(); it++)
      {
        if(!explicitly_solve_greater_greens_function)
          data[it->first] +=it->second;
        else
          data[it->first] -= it->second;

      }
    }
  }
  else if (variable=="derivative_electron_density_over_potential"||variable=="derivative_total_charge_density_over_potential")
  {
    if (electron_density_Jacobian.empty())
    {
      calculate_density(lesser_green_propagator, &electron_density_Jacobian, electron_hole_model);
    }
    for (std::map<unsigned int, double>::const_iterator it = electron_density_Jacobian.begin();
        it != electron_density_Jacobian.end(); it++)
    {
      if(!explicitly_solve_greater_greens_function)
        data[it->first] = - it->second;
      else
        data[it->first] = + it->second;
    }
    if(electron_hole_model)//&&hole_density.empty())
    {
      if(hole_density_Jacobian.empty())
        calculate_density(greater_green_propagator, &hole_density_Jacobian, electron_hole_model);

      for (std::map<unsigned int, double>::const_iterator it = hole_density_Jacobian.begin();
          it != hole_density_Jacobian.end(); it++)
      {
        if(!explicitly_solve_greater_greens_function)
          data[it->first] +=it->second;
        else
          data[it->first] -= it->second;

      }
    }
  }
  else if (variable == "density_of_states")
  {
    //throw std::runtime_error("ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_data density of states NYI ");
    DOS.clear();
    std::string writeable_Propagator_name = writeable_Propagator->get_name();
    if(options.get_option("energy_resolved_density_of_states_output",false))
    {
      options.set_option(writeable_Propagator_name + "_energy_resolved_output", true);
      options.set_option(writeable_Propagator_name + "_energy_resolved_output_real", options.get_option("energy_resolved_density_real",-1.0));
      options.set_option(writeable_Propagator_name + "_energy_resolved_output_imag", options.get_option("energy_resolved_density_imag",0.0));
      energy_resolved_DOS = true;
      options.set_option(writeable_Propagator_name + "_energy_resolved_output_one_per_file",options.get_option("energy_resolved_density_output_one_per_file",false));
    }
    if(options.get_option("k_resolved_density_of_states_output",false))
    {
      options.set_option(writeable_Propagator_name + "_k_resolved_output", true);
      options.set_option(writeable_Propagator_name + "_k_resolved_output_real", options.get_option("k_resolved_density_of_states_real",-1.0));
      options.set_option(writeable_Propagator_name + "_k_resolved_output_imag", options.get_option("k_resolved_density_of_states_imag",0.0));
      k_resolved_DOS = true;
      options.set_option(writeable_Propagator_name + "_k_resolved_output_one_per_file",options.get_option("k_resolved_density_of_states_output_one_per_file",false));
    }
    //if (options.get_option("density_of_states_output", false))
     {
       //calculate density
       if (DOS.empty())
       {
         calculate_density(writeable_Propagator, &DOS, false);
       }
       int myrank;
       MPI_Comm_rank(Mesh_tree_topdown.begin()->first->get_global_comm(), &myrank);

       {
         if (myrank == 0)
         {
           print_atomic_map(DOS, get_name() + "_density_of_states");
         }
       }
       //}
     }
  }
  else if(variable == "local_current")
  {
    //if (HG_density.empty() && do_solve_HG)
    //  calculate_density(Hlesser_green_propagator, &HG_density, false);
    NEMO_ASSERT(false,"called?");
  }

  else
    throw std::runtime_error(prefix + "called with unknown variable \"" + variable + "\"\n");

  //data = electron_density;
}

void ScatteringBackwardRGFSolver::calculate_density(Propagator* propagator, std::map<unsigned int, double>* result, bool density_by_hole_factor)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::calculate_density: ";
  NemoUtils::tic(tic_toc_prefix);

  std::string error_prefix = get_name() + "calculate density ";
  std::string name_of_propagator = propagator->get_name();
  NemoPhys::Propagator_type input_type = NemoPhys::Fermion_lesser_Green;
  std::map<double, std::map<unsigned int, double>, Compare_double_or_complex_number>* energy_resolved_density_for_output = NULL;
  std::map<std::complex<double>, std::map<unsigned int, double>, Compare_double_or_complex_number>
      * complex_energy_resolved_density_for_output;
  std::map<vector<double>, std::map<double, std::map<unsigned int, double> > > * energy_resolved_per_k_density_for_output;
  std::map<vector<double>, std::map<unsigned int, double> > * k_resolved_density_for_output;

  Simulation* source_of_data = this;
  bool k_resolved_data = false;
  bool energy_resolved_data = false;
  bool energy_resolved_per_k_data = false;

  if (name_of_propagator == lesser_green_propagator_name)
  {
    if(explicitly_solve_greater_greens_function)
      input_type = greater_green_propagator_type;
    else
      input_type = lesser_green_propagator_type;
    energy_resolved_density_for_output = &energy_resolved_electron_density_for_output;
    complex_energy_resolved_density_for_output = &complex_energy_resolved_electron_density_for_output;
    energy_resolved_per_k_density_for_output = &energy_resolved_per_k_electron_density_for_output;
    k_resolved_density_for_output = &k_resolved_electron_density_for_output;
    k_resolved_data = k_resolved_electron_density;
    energy_resolved_data = energy_resolved_electron_density;

  }
  else if (name_of_propagator == Hlesser_green_propagator_name)
  {
    input_type = Hlesser_green_propagator_type;
    energy_resolved_density_for_output = &energy_resolved_HG_density_for_output;
    complex_energy_resolved_density_for_output = &complex_energy_resolved_HG_density_for_output;
    energy_resolved_per_k_density_for_output = &energy_resolved_per_k_HG_density_for_output;
    k_resolved_density_for_output = &k_resolved_HG_density_for_output;
    k_resolved_data = k_resolved_HG_density;
    energy_resolved_data = energy_resolved_HG_density;
  }
  else if (name_of_propagator == greater_green_propagator_name)
  {
    if(explicitly_solve_greater_greens_function)
      input_type = lesser_green_propagator_type;
    else
      input_type = greater_green_propagator_type;
    energy_resolved_density_for_output = &energy_resolved_hole_density_for_output;
    complex_energy_resolved_density_for_output = &complex_energy_resolved_hole_density_for_output;
    energy_resolved_per_k_density_for_output = &energy_resolved_per_k_hole_density_for_output;
    k_resolved_density_for_output = &k_resolved_hole_density_for_output;
    k_resolved_data = k_resolved_hole_density;
    energy_resolved_data = energy_resolved_hole_density;
  }
  else if (name_of_propagator == writeable_Propagator->get_name())
  {
    input_type = writeable_Propagator->get_propagator_type();

    energy_resolved_density_for_output = &energy_resolved_DOS_for_output;
    complex_energy_resolved_density_for_output = &complex_energy_resolved_DOS_for_output;
    energy_resolved_per_k_density_for_output = &energy_resolved_per_k_DOS_for_output;
    k_resolved_density_for_output = &k_resolved_DOS_for_output;
    k_resolved_data = k_resolved_DOS;
    energy_resolved_data = energy_resolved_DOS;
  }
  else
    throw std::runtime_error("ScatteringBackwardRGFSolver(\"" + get_name() + "\")::get_data calculate density NYI ");

  if(energy_resolved_data)
  {
    //loop over all mesh_constructors and check whether one of them has the option ("non_rectangular = true")

    std::map<std::string, Simulation*>::const_iterator mesh_cit=Mesh_Constructors.begin();
    for(; mesh_cit!=Mesh_Constructors.end() && !energy_resolved_per_k_data; ++mesh_cit)
    {
      InputOptions& mesh_options = mesh_cit->second->get_reference_to_options();
      if(mesh_options.get_option(std::string("non_rectangular"),false) && energy_resolved_per_k)
      {
        energy_resolved_per_k_data = true;
        //don't want to store both energy resolved all k and energy resolved per k. This won't fit into memory.
        energy_resolved_data = false;
      }
    }
  }

  integrate_diagonal(source_of_data, input_type, name_of_propagator, energy_resolved_data, k_resolved_data, energy_resolved_per_k_data, density_by_hole_factor);

  //6. multiply with prefactors (depending on particle type)
  //get the prefactor according to the cell volume (depending on the momentum dimensionality)
  //for this read the momentum names (-1D,2D...)
  double prefactor = 1;
  const std::vector<std::string>& temp_mesh_names = propagator->momentum_mesh_names;
  for (unsigned int i = 0; i < temp_mesh_names.size(); i++)
  {
    if (temp_mesh_names[i].find("energy") != std::string::npos || temp_mesh_names[i].find("momentum_1D") != std::string::npos)
      prefactor /= 2.0 * NemoMath::pi;
    else if (temp_mesh_names[i].find("momentum_2D") != std::string::npos)
      prefactor /= 4.0 * NemoMath::pi * NemoMath::pi;
    else if (temp_mesh_names[i].find("momentum_3D") != std::string::npos)
      prefactor /= 8.0 * NemoMath::pi * NemoMath::pi * NemoMath::pi;
  }
  if (result != NULL)
    result->clear();

  double real_part_of_prefactor2(options.get_option("real_part_of_density_prefactor2", 0.0));
  double imaginary_part_of_prefactor2(options.get_option("imaginary_part_of_density_prefactor2", -1.0));
  std::complex<double> prefactor2(real_part_of_prefactor2, imaginary_part_of_prefactor2);

  //7. store the results in the result map
  //if the DOFmap size does not agree with the size of vector, we ask the matrix source how to translate the vector index into the atom_id
  //otherwise, we assume that there is a one to one correspondence
  std::vector < std::complex<double> > temp_density(*(propagator->get_readable_integrated_diagonal()));
  translate_vector_into_map(temp_density, cplx(prefactor,0.0) * prefactor2, true, *result);

  //8. output the energy resolved density
  if (energy_resolved_data)
  {
    //8.1 check whether this MPI rank is the one where the data had been reduced to
    NEMO_ASSERT(Mesh_tree_topdown.size() > 0, error_prefix + "Mesh_tree_topdown is not ready for usage\n");
    const MPI_Comm& topcomm = Mesh_tree_topdown.begin()->first->get_global_comm();
    int my_rank;
    MPI_Comm_rank(topcomm, &my_rank);
    {
      double real_prefactor = options.get_option(name_of_propagator + "_energy_resolved_output_real", 1.0);
      double imag_prefactor = options.get_option(name_of_propagator + "_energy_resolved_output_imag", 0.0);
      std::complex<double> total_input_prefactor(real_prefactor, imag_prefactor);
      //8.2 get access to the energy resolved data
      //source_of_matrices->get_data(name_of_propagator,lesser_Green);

      if (complex_energy_used())
      {
        std::map<std::complex<double>, std::vector<std::complex<double> >, Compare_double_or_complex_number> temp_density(
            *(propagator->get_complex_energy_resolved_integrated_diagonal()));
        std::map<std::complex<double>, std::vector<std::complex<double> >, Compare_double_or_complex_number>::iterator it =
            temp_density.begin();
        for (; it != temp_density.end(); ++it)
        {
          std::map<unsigned int, double> temp_map;
          translate_vector_into_map(it->second, total_input_prefactor * prefactor * std::complex<double>(0.0, -1.0), true,
              temp_map);
          (*complex_energy_resolved_density_for_output)[it->first] = temp_map;
        }
      }
      else
      {
        std::map<double, std::vector<std::complex<double> >, Compare_double_or_complex_number> temp_density(
            *(propagator->get_energy_resolved_integrated_diagonal()));
        std::map<double, std::vector<std::complex<double> >, Compare_double_or_complex_number>::iterator it =
            temp_density.begin();
        for (; it != temp_density.end(); ++it)
        {
          std::map<unsigned int, double> temp_map;
          translate_vector_into_map(it->second, total_input_prefactor * prefactor * std::complex<double>(0.0, -1.0), true,
              temp_map);
          (*energy_resolved_density_for_output)[it->first] = temp_map;
        }
      }
      if (my_rank == 0 && options.get_option(name_of_propagator + "_energy_resolved_output", false))
      {
        if (options.get_option(name_of_propagator + "_energy_resolved_output_one_per_file", false))
        {
          print_atomic_maps_per_file(*energy_resolved_density_for_output, name_of_propagator + "_energy_resolved.E=");
        }
        else
        {
          std::string filename = "energy_resolved_" + name_of_propagator;
          if (options.check_option("output_suffix"))
            filename=filename+"_"+options.get_option("output_suffix",std::string(""));
          if (!options.get_option("solve_eqneq", false))
          {
            if (complex_energy_used())
            {
              print_atomic_maps(*complex_energy_resolved_density_for_output, filename);
            }
            else
            {
              print_atomic_maps(*energy_resolved_density_for_output, filename);
            }
          }
        }
        reset_output_counter();
      }
    }
  }
  if (energy_resolved_per_k_data)
  {
    //here I don't want to output that can be done elsewhere but just prepare the data for output.
    // I will also clear Propagator diagonal as we go for memory concerns.
    NEMO_ASSERT(Mesh_tree_topdown.size() > 0, error_prefix + "Mesh_tree_topdown is not ready for usage\n");
    const MPI_Comm& topcomm = Mesh_tree_topdown.begin()->first->get_global_comm();
    int my_rank;
    MPI_Comm_rank(topcomm, &my_rank);
    //if(my_rank==0)
    {
      double real_prefactor = options.get_option(name_of_propagator + "_energy_resolved_output_real", 1.0);
      double imag_prefactor = options.get_option(name_of_propagator + "_energy_resolved_output_imag", 0.0);
      std::complex<double> total_input_prefactor(real_prefactor, imag_prefactor);

      //source_of_matrices->get_data(name_of_propagator,lesser_Green);
      std::map<vector<double>, std::map<double, std::vector<std::complex<double> > > > temp_density(
          *(propagator->get_energy_resolved_per_k_integrated_diagonal()));
      std::map<vector<double>, std::map<double, std::vector<std::complex<double> > > >::iterator k_it = temp_density.begin();
      for (; k_it != temp_density.end(); ++k_it)
      {
        std::map<double, std::map<unsigned int, double> > temp_map2;
        std::map<double, std::vector<std::complex<double> > >::iterator e_it = k_it->second.begin();
        for (; e_it != k_it->second.end(); ++e_it)
        {
          std::map<unsigned int, double> temp_map;
          temp_map[0] = ((e_it->second)[0] * total_input_prefactor * prefactor).real();
          temp_map2[e_it->first] = temp_map;
        }
        (*energy_resolved_per_k_density_for_output)[k_it->first] = temp_map2;
      }
    }
  }
  //9.0 output the k_resolved density if desired
  if (options.get_option(name_of_propagator + "_k_resolved_output", false) || options.get_option("k_resolved_density", false))
  {
    //9.1 check whether this MPI rank is the one where the data had been reduced to
    NEMO_ASSERT(Mesh_tree_topdown.size() > 0, error_prefix + "Mesh_tree_topdown is not ready for usage\n");
    const MPI_Comm& topcomm = Mesh_tree_topdown.begin()->first->get_global_comm();
    int my_rank;
    MPI_Comm_rank(topcomm, &my_rank);
    {
      double real_prefactor = options.get_option(name_of_propagator + "_k_resolved_output_real", 1.0);
      double imag_prefactor = options.get_option(name_of_propagator + "_k_resolved_output_imag", 0.0);
      std::complex<double> total_input_prefactor(real_prefactor, imag_prefactor);
      //8.2 get access to the k resolved data
      //source_of_matrices->get_data(name_of_propagator,lesser_Green);
      std::map<vector<double>, std::vector<std::complex<double> > > temp_density(
          *(propagator->get_k_resolved_integrated_diagonal()));
      std::map<vector<double>, std::vector<std::complex<double> > >::iterator it = temp_density.begin();

      //loop through all density
      for (; it != temp_density.end(); ++it)
      {
        std::map<unsigned int, double> temp_map;
        translate_vector_into_map(it->second, total_input_prefactor * prefactor * std::complex<double>(0.0, -1.0), true,
            temp_map);
        (*k_resolved_density_for_output)[it->first] = temp_map;
      }
      if (my_rank == 0 && options.get_option(name_of_propagator + "_k_resolved_output", false))
      {
        if (options.get_option(name_of_propagator + "_k_resolved_output_one_per_file", false))
        {
          print_atomic_maps_per_file(*k_resolved_density_for_output, name_of_propagator + "_k_resolved.k=");
        }
        else
        {
          std::string filename = "k_resolved_" + name_of_propagator; //Yu:this should be enough, otherwise the name will be too long
          if (options.check_option("output_suffix"))
            filename=filename+"_"+options.get_option("output_suffix",std::string(""));
          //if(options.check_option("output_suffix"))
          //  filename += options.get_option("output_suffix",std::string(""));
          if (options.get_option(name_of_propagator + "_k_resolved_output", bool(false)))
            print_atomic_maps(*k_resolved_density_for_output, filename);
        }
        reset_output_counter();
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void ScatteringBackwardRGFSolver::initialize_Propagator(Propagator* Propagator_pointer)
{
  std::string error_prefix = get_name() + " initialize_Propagator Propagator_pointer";
  //update some mesh information if needed, i.e. only for the interaction with resonance finder
  {
    if (Mesh_tree_topdown.size() == 0)
    {
      //get the Mesh_tree_names:
      Parallelizer->get_data("mesh_tree_names", Mesh_tree_names);
      //get the Mesh_tree_downtop
      Parallelizer->get_data("mesh_tree_downtop", Mesh_tree_downtop);
      //get the Mesh_tree_topdown
      Parallelizer->get_data("mesh_tree_topdown", Mesh_tree_topdown);
    }

    //shape up the Propagator (setting its momenta, etc.) if this==constructor of it
    {

      Propagator_pointer->momentum_mesh_names.resize(number_of_momenta);
      bool do_initialize_propagator = true;
      //if(options.get_option("skip_book_keeping",false))
      if (get_skip_book_keeping())
      {
        do_initialize_propagator = false;
      }
      if (!options.get_option("one_energy_only", false))
      {

        //find the topmost Mesh (all to be replaced by a call to Propagation Parallelizer)
        NemoMesh* topmost_mesh = NULL;
        std::map<NemoMesh*, std::vector<NemoMesh*> >::const_iterator Mesh_it = Mesh_tree_topdown.begin();
        for (; Mesh_it != Mesh_tree_topdown.end(); ++Mesh_it)
        {
          if (Mesh_it->first->get_name() == Mesh_tree_names[0])
          {
            NEMO_ASSERT(topmost_mesh == NULL, error_prefix + "found more than one topmost mesh\n");
            topmost_mesh = Mesh_it->first;
          }
        }
        for (unsigned int i = 0; i < number_of_momenta; i++)
        {
          NEMO_ASSERT(Propagator_pointer->momentum_mesh_names.size() > i,
              error_prefix + "momentum_mesh_names of Propagator \"" + name_of_writeable_Propagator + "\" is too small\n");
          Propagator_pointer->momentum_mesh_names[i] = Mesh_tree_names[i];
        }
      }
      else
      {
        std::vector < std::string > ordered_momentum_list;
        create_ordered_mesh_list (ordered_momentum_list);
        NEMO_ASSERT(ordered_momentum_list.size() == number_of_momenta, error_prefix + "inconsistent number of momenta\n");
        for (unsigned int i = 0; i < number_of_momenta; i++)
          Propagator_pointer->momentum_mesh_names[i] = ordered_momentum_list[i];
      }

      std::set < std::vector<NemoMeshPoint> > *pointer_to_resulting_points;
      Parallelizer->get_data("local_momenta", pointer_to_resulting_points);
      std::set < std::vector<NemoMeshPoint> > &resulting_points = *pointer_to_resulting_points;
      if (do_initialize_propagator)
      {
        msg << error_prefix << "initializing " << resulting_points.size() << " matrices" << " for the Propagator \""
            << name_of_writeable_Propagator << "\n";
        std::set<std::vector<NemoMeshPoint> >::const_iterator momentum_c_it = resulting_points.begin();
        for (; momentum_c_it != resulting_points.end(); ++momentum_c_it)
        {
          Propagator_pointer->propagator_map[*momentum_c_it] = NULL;
          //set the boolean for the matrix at this momentum being allocated
          Propagator_pointer->allocated_momentum_Propagator_map[*momentum_c_it] = false;

          //set_job_done_momentum_map(&(name_of_writeable_Propagator),&*momentum_c_it,false);
          if (Propagator_pointer == lesser_green_propagator)
            lesser_propagator_job_done_momentum_map[*momentum_c_it] = false;
          else if(solve_greater_greens_function && Propagator_pointer == greater_green_propagator)
            greater_propagator_job_done_momentum_map[*momentum_c_it] = false;
          else if (Propagator_pointer == Hlesser_green_propagator)
            Hlesser_propagator_job_done_momentum_map[*momentum_c_it] = false;

          //for debug output:
          for (unsigned int j = 0; j < (*momentum_c_it).size(); j++)
            (*momentum_c_it)[j].print();
        }
      }

      // determine whether to offload this *momentum_c_it and where
      if (offload_solver_initialized)
      {
        std::set<std::vector<NemoMeshPoint> >::const_iterator momentum_c_it = resulting_points.begin();
        for (; momentum_c_it != resulting_points.end(); ++momentum_c_it)
        {
          NEMO_ASSERT(offload_solver != NULL, "Offload solver " + offload_solver->get_name() + " has not been defined\n");
          fill_offloading_momentum_map(*momentum_c_it, offload_solver->offloading_momentum_map);
        }
      }
    }
  }
}

void ScatteringBackwardRGFSolver::output_slab_resolved_current()
{

  PropagationUtilities::integrate_vector(this, slab_resolved_local_current_energy_k_resolved, slab_resolved_local_current);

  if (holder.geometry_replica == 0)
  {
    if (holder.geometry_rank == 0)
    {

      std::ofstream output_file;
      output_file.precision(21);
      std::string suffix = options.get_option("output_suffix", std::string(""));
      std::string output_file_name = get_name() + "_slab_resolved_current_" + suffix + ".dat";
      output_file.open(output_file_name.c_str());
      output_file << "% index  current(A) \n";
      for (unsigned int i = 0; i < subdomain_names.size(); i++)
      {
        //if (energy_resolved && i != 0)
        //  energy_resolved_local_current_file << i << " \t";
        //calculate_slab_resolved_local_current(i, energy_resolved);

        output_file << i << " " << slab_resolved_local_current[i] << " \n";

      }
      output_file.close();
    }
  }

  if( energy_resolved_HG_density )
  {
    //void PropagationUtilities::integrate_vector_for_energy_resolved(Simulation* this_simulation, std::map<std::vector<NemoMeshPoint>,
    //                                                            std::vector<double > >*& input_data,
    //                                                            std::map<double,std::vector<double> >*& result,
    //                                                            bool multiply_by_energy)
    std::map<double, std::vector<double> > tmp;
    std::map<double, std::vector<double> > *slab_resolved_local_current_energy_resolved = &tmp;
    std::map<std::vector<NemoMeshPoint>, std::vector<double> >* input_data = &slab_resolved_local_current_energy_k_resolved;
    PropagationUtilities::integrate_vector_for_energy_resolved(this, input_data,
        slab_resolved_local_current_energy_resolved, false);
    if (holder.geometry_replica == 0)
    {
      if (holder.geometry_rank == 0)
      {
        std::string suffix = options.get_option("output_suffix", std::string(""));
        std::string output_file_name_e_resolved = get_name() + "_slab_resolved_energy_current_" + suffix + ".dat";
        energy_resolved_local_current_file.open(output_file_name_e_resolved.c_str());

        for (unsigned int i = 0; i < subdomain_names.size(); i++)
        {
          //loop through and set header
          if (i == 0)
          {
            std::string energy_resolved_file_name = get_name() + "_energy.dat";
            std::ofstream energy_resolved_file;
            energy_resolved_file.open(energy_resolved_file_name.c_str());

            energy_resolved_local_current_file << "%\t";
            //std::map<double, std::map<unsigned int, double> >::iterator temp_it = energy_resolved_HG_density_for_output.begin();
            std::map<double, std::vector<double> >::iterator temp_it = slab_resolved_local_current_energy_resolved->begin();
            //for (; temp_it != energy_resolved_HG_density_for_output.end(); ++temp_it)
            for(; temp_it != slab_resolved_local_current_energy_resolved->end(); ++temp_it)
            {
              energy_resolved_local_current_file << temp_it->first << "[eV] \t";
              energy_resolved_file << temp_it->first << " \n";
            }
            energy_resolved_local_current_file << "\n";
            energy_resolved_local_current_file << i << " \t";
            energy_resolved_file.close();
          }
          else
            energy_resolved_local_current_file << i << " \t";

          vector<double> temp_energy_resolved(slab_resolved_local_current_energy_resolved->size(), 0.0);
          int energy_count = 0;
          std::map<double, vector<double> >::iterator energy_it = slab_resolved_local_current_energy_resolved->begin();
          for (; energy_it != slab_resolved_local_current_energy_resolved->end(); ++energy_it)
          {
            /*   //reset atom iterator
        it = atoms.active_atoms_begin();
        double temp_atomic_density = 0.0;
        for (; it != end; ++it)
        {
          const unsigned int atom_id = it.id();
          std::map<unsigned int, double>::iterator curr_it = energy_it->second.find(atom_id);
          temp_atomic_density += curr_it->second;
        }
             */
            //output to file
            double temp_current = (energy_it->second)[i];
            temp_current *= -1 / NemoPhys::elementary_charge; //convert from A to A/eV
            temp_energy_resolved[energy_count] = temp_current;
            energy_count++;
          }
          for (int i = 0; i < energy_count; i++)
            energy_resolved_local_current_file << temp_energy_resolved[i] << " \t";
          energy_resolved_local_current_file << "\n";

        }
        energy_resolved_local_current_file.close();
      }
    }
  }
  //.clear();




}





void ScatteringBackwardRGFSolver::prepare_HG_matrices(const std::vector<NemoMeshPoint>& momentum, unsigned int i,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& H_blocks,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& offdiagonal_indices,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gL_blocks,
    PetscMatrixParallelComplex*& left_coupling, PetscMatrixParallelComplex*& gR_block,
    PetscMatrixParallelComplex*& GL_block, PetscMatrixParallelComplex*& gL_block,
    PetscMatrixParallelComplex*& GR_block)
{
  //int i = subdomain_names.size() - 1;

        std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator off_block_cit =
            H_blocks.find(offdiagonal_indices[i - 1]);
        NEMO_ASSERT(off_block_cit != H_blocks.end(),
            "ScatteringBackwardRGFSolver(\"" + get_name()
            + "\")::run_backward_RGF_for_momentum have not found offdiagonal block Hamiltonian\n");
        PetscMatrixParallelComplex* left_coupling_temp = off_block_cit->second;

        //get pointer to gL(i-1)
        std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator gL_block_cit = gL_blocks.find(diagonal_indices[i - 1]);
        NEMO_ASSERT(gL_block_cit != gL_blocks.end(),
            "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of gL\n");
        gL_block = gL_block_cit->second;

        //get pointer to gR(i-1)
        std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator gR_block_cit = gR_blocks.find(diagonal_indices[i - 1]);
        NEMO_ASSERT(gR_block_cit != gR_blocks.end(),
            "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of gL\n");
        gR_block = gR_block_cit->second;

        //get pointer to Gl
        //PetscMatrixParallelComplex* GL_block = NULL;
        {
          Propagator::PropagatorMap& result_prop_map = lesser_green_propagator->propagator_map;
          Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

          PetscMatrixParallelComplexContainer* GL_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
          const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GL_blocks =
              GL_container->get_const_container_blocks();
          std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GL_block_cit =
              GL_blocks.find(diagonal_indices[i]);
          NEMO_ASSERT(GL_block_cit != GL_blocks.end(),
              "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of GL\n");
          GL_block = GL_block_cit->second;
        }

        //get pointer to GR
        //PetscMatrixParallelComplex* GR_block = NULL;
        {
          Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
          Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);

          PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
          const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GR_blocks =
              GR_container->get_const_container_blocks();
          std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GR_block_cit =
              GR_blocks.find(diagonal_indices[i]);
          NEMO_ASSERT(GR_block_cit != GR_blocks.end(),
              "ScatteringBackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum have not found sub-block of GR\n");
          GR_block = GR_block_cit->second;
        }

        //Domain* name_domain = Domain::get_domain(subdomain_names[i]);
        //PetscMatrixParallelComplex left_coupling2(left_coupling->get_num_cols(), left_coupling->get_num_rows(),
        //    left_coupling->get_communicator());
        left_coupling = new PetscMatrixParallelComplex(left_coupling_temp->get_num_cols(), left_coupling_temp->get_num_rows(),
                left_coupling_temp->get_communicator());
        left_coupling_temp->hermitian_transpose_matrix(*left_coupling, MAT_INITIAL_MATRIX);

}

void ScatteringBackwardRGFSolver::solve_HG(std::pair<std::vector<int>, std::vector<int> >& diagonal_index,
    const std::vector<NemoMeshPoint>& momentum, PetscMatrixParallelComplex* left_coupling, PetscMatrixParallelComplex* gR_block,
    PetscMatrixParallelComplex* GL_block, PetscMatrixParallelComplex* gL_block, PetscMatrixParallelComplex* GR_block,
    PetscMatrixParallelComplex*& HG)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_HG: ";
  NemoUtils::tic(tic_toc_prefix);

  //solve_HG(diagonal_indices[i] ,momentum, left_coupling, gR_block, GL_block, gL_block, GR_block, HG);

  bool diagonal_result = options.get_option("diagonal_Greens_function", true);
  PropagationUtilities::core_H_lesser(left_coupling, gR_block, GL_block, gL_block, GR_block, HG, diagonal_result);
  //store in container
  Propagator::PropagatorMap& result_prop_map = Hlesser_green_propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    Hlesser_green_propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*HG, diagonal_index.first, diagonal_index.second);
  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::solve_nonlocal_HG( std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
    const std::vector<NemoMeshPoint>& momentum,int index,  const std::map<std::pair<std::vector<int>, std::vector<int> >,
    PetscMatrixParallelComplex*>& H_blocks,
    const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& GL_blocks,
       PetscMatrixParallelComplex*& HG)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_nonlocal_HG: ";
  NemoUtils::tic(tic_toc_prefix);

  std::pair<std::vector<int>, std::vector<int> > temp_coupling_pair(diagonal_indices[index].first,diagonal_indices[index+1].second);

  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator off_block_cit =
      H_blocks.find(temp_coupling_pair);
  //PetscMatrixParallelComplex* coupling = off_block_cit->second;
  //transpose
  PetscMatrixParallelComplex left_coupling(off_block_cit->second->get_num_cols(), off_block_cit->second->get_num_rows(),
      off_block_cit->second->get_communicator());
  off_block_cit->second->hermitian_transpose_matrix(left_coupling, MAT_INITIAL_MATRIX);
  //get G<(i+1,i)
  std::pair<std::vector<int>, std::vector<int> > temp_pair(diagonal_indices[index+1].first,diagonal_indices[index].second);

  std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>::const_iterator GL_block_cit =
      GL_blocks.find(temp_pair);
  NEMO_ASSERT(GL_block_cit != GL_blocks.end(),
      tic_toc_prefix +  " have not found offdiagonal sub-block of GL\n");
  PetscMatrixParallelComplex* GL_off_block = GL_block_cit->second;
  PetscMatrixParallelComplex GL_conj(GL_block_cit->second->get_num_cols(), GL_block_cit->second->get_num_rows(),
      GL_block_cit->second->get_communicator());
  GL_block_cit->second->hermitian_transpose_matrix(GL_conj, MAT_INITIAL_MATRIX);
  PetscMatrixParallelComplex* temp_HG = NULL;

  PetscMatrixParallelComplex::mult(*GL_block_cit->second,*off_block_cit->second, &temp_HG);
  *temp_HG*=std::complex<double>(0.0,1.0);

  temp_HG->get_diagonal_matrix(HG);
  *HG *=std::complex<double>(2*NemoPhys::elementary_charge*NemoPhys::elementary_charge/NemoPhys::hbar,0);

  HG->assemble();
  delete temp_HG;
  Propagator::PropagatorMap& result_prop_map = Hlesser_green_propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    Hlesser_green_propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*HG, diagonal_indices[index+1].first, diagonal_indices[index+1].second);
  NemoUtils::toc(tic_toc_prefix);

}
void ScatteringBackwardRGFSolver::transform_data_to_atoms(Propagator* propagator, const std::vector<NemoMeshPoint>& momentum)
{
  //PetscMatrixParallelComplex* basis_functions = NULL;
  Propagator::PropagatorMap& result_prop_map = propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  NEMO_ASSERT(prop_it != result_prop_map.end(), get_name() + " could not find HG for transformation");
  //prop_it->second->assemble();

  //delete result;
  //result=NULL;
  PetscMatrixParallelComplex *temp_matrix = prop_it->second;

  std::vector<cplx> temp_vector;
  if (!temp_matrix->is_ready())
    temp_matrix->assemble();
  temp_matrix->get_diagonal(&temp_vector);

  PetscMatrixParallelComplex* Matrix = NULL;
  const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*>(get_simulation_domain());
  DOFmapInterface& dof_map = Hamilton_Constructor->get_dof_map(get_simulation_domain());
  std::map<int, int> index_to_atom_id_map;
  dof_map.build_atom_id_to_local_atom_index_map(&index_to_atom_id_map, true);

  const AtomicStructure& atoms = domain->get_atoms();
  ConstActiveAtomIterator it = atoms.active_atoms_begin();
  int counter = 0;
  //get the number of active atoms to resize the result_vector
  for (; it != atoms.active_atoms_end(); ++it)
    counter++;
  int number_of_atoms = counter;
  it = atoms.active_atoms_begin();

  //2.2 transform the source vector
  std::vector<cplx> result_vector(number_of_atoms, cplx(0.0, 0.0));
  counter = 0;
  for (it = atoms.active_atoms_begin(); it != atoms.active_atoms_end(); ++it)
  {
    //const unsigned int atom_id = it.id();
    const unsigned int atom_id = it.id();
    const std::map<short, unsigned int>* atom_dof_map = dof_map.get_atom_dof_map(atom_id);
    /*std::map<int,int>::const_iterator c_it=index_to_atom_id_map.find(counter);
    NEMO_ASSERT(c_it!=index_to_atom_id_map.end(),get_name()+"have not found the atom_id\n");
    const AtomStructNode& nd        = it.node();
    //cerr << "x y z " << nd.position[0] << " " << nd.position[1] << " " << nd.position[2] << " \n";
    const std::map<short, unsigned int>* atom_dof_map = dof_map.get_atom_dof_map(c_it->second);
*/
    cplx temp_container(0.0, 0.0);
    std::map<short, unsigned int>::const_iterator c_it2 = atom_dof_map->begin();
    for (; c_it2 != atom_dof_map->end(); c_it2++)
      temp_container += temp_vector[c_it2->second];
    result_vector[counter] = temp_container;
    counter++;
  }

  //3. store vector in result_matrix
  //3.1 put result vector into petsc vector
  PetscVectorNemo<cplx> petsc_vector_diagonal(number_of_atoms, number_of_atoms, holder.geometry_communicator);
  std::vector<int> indices(number_of_atoms, 0);
  for (unsigned int i = 0; i < indices.size(); i++)
    indices[i] = i; //dense vector...
  petsc_vector_diagonal.set_values(indices, result_vector);
  //3.2 store into result matrix
  Matrix = new PetscMatrixParallelComplex(number_of_atoms, number_of_atoms, holder.geometry_communicator);
  //temporary_matrices.insert(result_matrix);
  Matrix->set_num_owned_rows(number_of_atoms);
  for (int i = 0; i < number_of_atoms; i++)
    Matrix->set_num_nonzeros(i, 1, 0);
  Matrix->allocate_memory();
  Matrix->set_to_zero();
  Matrix->matrix_diagonal_shift(petsc_vector_diagonal);
  Matrix->assemble();

  delete temp_matrix;

  Propagator::PropagatorMap::iterator prop_it2 = propagator->propagator_map.find(momentum);
  prop_it->second = Matrix;
}

void ScatteringBackwardRGFSolver::transform_current_to_slabs(Propagator* propagator, const std::vector<NemoMeshPoint>& momentum,
                              std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices)
{

  double prefactor = 1;
  const std::vector<std::string>& temp_mesh_names = propagator->momentum_mesh_names;
  for (unsigned int i = 0; i < temp_mesh_names.size(); i++)
  {
    if (temp_mesh_names[i].find("energy") != std::string::npos || temp_mesh_names[i].find("momentum_1D") != std::string::npos)
      prefactor /= 2.0 * NemoMath::pi;
    else if (temp_mesh_names[i].find("momentum_2D") != std::string::npos)
      prefactor /= 4.0 * NemoMath::pi * NemoMath::pi;
    else if (temp_mesh_names[i].find("momentum_3D") != std::string::npos)
      prefactor /= 8.0 * NemoMath::pi * NemoMath::pi * NemoMath::pi;
  }

  double real_part_of_prefactor2(options.get_option("real_part_of_density_prefactor2", 0.0));
  double imaginary_part_of_prefactor2(options.get_option("imaginary_part_of_density_prefactor2", -1.0));
  std::complex<double> prefactor2(real_part_of_prefactor2, imaginary_part_of_prefactor2);


  Propagator::PropagatorMap& result_prop_map = propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  NEMO_ASSERT(prop_it != result_prop_map.end(), get_name() + " could not find HG for transformation");

  PetscMatrixParallelComplex *temp_matrix_container = prop_it->second;

  int num_slabs = diagonal_indices.size();

  vector<double> slab_current(num_slabs,0.0);

  for (int slab_idx = 0; slab_idx < num_slabs; ++slab_idx)
  {
    //get values for this subdomain
    PetscMatrixParallelComplex* temp_submatrix = NULL;
    temp_matrix_container->get_submatrix(diagonal_indices[slab_idx].first, diagonal_indices[slab_idx].second, MAT_REUSE_MATRIX, temp_submatrix);
    if(!temp_submatrix->is_ready())
      temp_submatrix->assemble();

    std::vector<cplx> temp_vector;
    temp_submatrix->get_diagonal(&temp_vector);
    cplx tmp_cplx_value(0.0,0.0);
    int slab_size = diagonal_indices[slab_idx].first.size();
    for (int i = 0; i < slab_size; ++i)
      tmp_cplx_value += temp_vector[i];

    double tmp_current_value = (prefactor*prefactor2*tmp_cplx_value).real();
    slab_current[slab_idx] = tmp_current_value;
  }
  if(use_explicit_blocks && subdomain_names.size() > 2)
    slab_current[subdomain_names.size()-1] = slab_current[subdomain_names.size()-2];
   //cerr << "slab current " << slab_current[1] << " \n";
  slab_resolved_local_current_energy_k_resolved[momentum] = slab_current;
  delete_propagator_matrices(Hlesser_green_propagator,&momentum);




}

/*
void ScatteringBackwardRGFSolver::solve_offdiagonal_GR_nonlocal_blocks(const std::vector<NemoMeshPoint>& momentum,
    std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
             int number_offdiagonal_blocks, int row_index, int col_index, int start_index, std::string subdomain_names,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks,
             const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
             PetscMatrixParallelComplexContainer*& GR_container, PetscMatrixParallelComplex*& result_Gr)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_offdiagonal_GR_nonlocal_blocks: ";
  NemoUtils::tic(tic_toc_prefix);

  msg.threshold(1) << " row " << row_index << " col " << col_index << " \n";
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
      PetscMatrixParallelComplex::mult(*Gr_row_j, *L_j_col, &temp_matrix);
    else
    {
      PetscMatrixParallelComplex L_col_j(L_j_col->get_num_cols(), L_j_col->get_num_rows(),
          L_j_col->get_communicator());
      L_j_col->transpose_matrix(L_col_j, MAT_INITIAL_MATRIX);
      PetscMatrixParallelComplex::mult(L_col_j, *Gr_row_j, &temp_matrix); // Gr_row_j is really Gr_j_row
    }

    if(Gr_row_row_temp == NULL)
      Gr_row_row_temp = new PetscMatrixParallelComplex(*temp_matrix);
    else
    {
      Gr_row_row_temp->add_matrix(*temp_matrix,SAME_NONZERO_PATTERN,cplx(1.0,0.0));
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
  if(!use_tranpose)
    PetscMatrixParallelComplex::mult(*Gr_row_row_temp,*gR_col_col,&result_Gr);
  else
    PetscMatrixParallelComplex::mult(*gR_col_col,*Gr_row_row_temp,&result_Gr);

  if(row_index!=col_index)
  {
    msg.threshold(1) << " row2 " << row_index << " col2 " << col_index << " \n";
    msg.threshold(1) << " diagonal_indices[row_index].first " <<  diagonal_indices[row_index].first[0] << " \n";
    msg.threshold(1) << " diagonal_indices[col_index].second " <<  diagonal_indices[col_index].second[0] << " \n";
    result_Gr->matrix_convert_dense();
    GR_container->set_block_from_matrix1(*result_Gr, diagonal_indices[row_index].first, diagonal_indices[col_index].second);
    //hopefully temporary
    //PetscMatrixParallelComplex temp_GR_transpose(result_Gr->get_num_cols(), result_Gr->get_num_rows(),
    //    result_Gr->get_communicator());
    //      result_Gr->transpose_matrix(temp_GR_transpose, MAT_INITIAL_MATRIX);
    //GR_container->set_block_from_matrix1(temp_GR_transpose, diagonal_indices[col_index].second, diagonal_indices[row_index].first);


  }

  if(debug_output)
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
  NemoUtils::tic(tic_toc_prefix);
}
*/

void ScatteringBackwardRGFSolver::solve_diagonal_GR_nonlocal(std::vector < std::pair<std::vector<int>, std::vector<int> > >& diagonal_indices,
       const std::vector<NemoMeshPoint>& momentum, int number_of_offdiagonal_blocks, int col_index,
       const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& L_blocks,
       PetscMatrixParallelComplex* this_gR_block,
       const std::map<std::pair<std::vector<int>, std::vector<int> >, PetscMatrixParallelComplex*>& gR_blocks,
       PetscMatrixParallelComplex*& this_GR_block)
{
  std::string tic_toc_prefix = "ScatteringBackwardRGFSolver::(\"" + this->get_name() + "\")::solve_diagonal_GR_nonlocal: ";
  NemoUtils::tic(tic_toc_prefix);

  //temp_matrix = gR(i,i)*sum(L(i,j)*GR(j,i))
  PetscMatrixParallelComplex* temp_matrix = NULL;
  Propagator::PropagatorMap& result_prop_map = writeable_Propagator->propagator_map;
  Propagator::PropagatorMap::iterator prop_it = result_prop_map.find(momentum);
  PetscMatrixParallelComplex* temp_matrix_container = prop_it->second;
  PetscMatrixParallelComplexContainer* GR_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(temp_matrix_container);
  NEMO_ASSERT(GR_container != NULL,
      "BackwardRGFSolver(\"" + get_name() + "\")::run_backward_RGF_for_momentum GR is not received in the container format\n");
  PropagationUtilities::solve_offdiagonal_GR_nonlocal_blocks(this, momentum, diagonal_indices, number_of_offdiagonal_blocks-1, col_index, col_index, col_index+1,
               subdomain_names, L_blocks, gR_blocks, GR_container, temp_matrix);
  if(debug_output)
  {
    temp_matrix->save_to_matlab_file("diag_GR_nonlocal_temp_matrix_"+subdomain_names[col_index]+".m");
  }

//  //solve temp_matrix2= gR(i,i)*temp_matrix
//  PetscMatrixParallelComplex* temp_matrix2 = NULL;
//  PetscMatrixParallelComplex::mult(*temp_matrix, *this_gR_block, &temp_matrix2);
//  if(debug_output)
//  {
//    temp_matrix2->save_to_matlab_file("diag_GR_nonlocal_temp_matrix2_"+subdomain_names[col_index]+".m");
//  }

  //delete temp_matrix;
  //temp_matrix = NULL;

  //this_GR_block = gR(i,i)+temp_matrix2;
  this_GR_block = new PetscMatrixParallelComplex(*this_gR_block);
  this_GR_block->add_matrix(*temp_matrix, SAME_NONZERO_PATTERN, std::complex<double>(1.0, 0.0));
  if(debug_output)
   {
     this_GR_block->save_to_matlab_file("diag_GR_nonlocal_result_"+subdomain_names[col_index]+".m");
   }
  delete temp_matrix;
  temp_matrix = NULL;

  //NemoMath::symmetry_type type = NemoMath::symmetric;
  //PropagationUtilities::symmetrize(this,this_GR_block,type);
  //store in container
  ///*Propagator::PropagatorMap&*/ result_prop_map = writeable_Propagator->propagator_map;
  ///*Propagator::PropagatorMap::iterator*/ prop_it = result_prop_map.find(momentum);
  if (prop_it == result_prop_map.end())
  {
    result_prop_map[momentum] = NULL;
    prop_it = result_prop_map.find(momentum);
  }
  if (prop_it->second == NULL)
  {
    PetscMatrixParallelComplexContainer* new_container = new PetscMatrixParallelComplexContainer(Hamiltonian->get_num_rows(),
        Hamiltonian->get_num_cols(), Hamiltonian->get_communicator());
    writeable_Propagator->allocated_momentum_Propagator_map[momentum] = true;
    prop_it->second = new_container;
  }
  PetscMatrixParallelComplexContainer* temp_container = dynamic_cast<PetscMatrixParallelComplexContainer*>(prop_it->second);
  temp_container->set_block_from_matrix1(*this_GR_block, diagonal_indices[col_index].first, diagonal_indices[col_index].second);
  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::calculate_slab_resolved_local_current(int iteration_index, bool energy_resolved)
{
  Domain* this_domain = Domain::get_domain(subdomain_names[iteration_index]);
  const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*>(this_domain);
  const AtomicStructure& atoms = domain->get_atoms();
  ConstActiveAtomIterator it = atoms.active_atoms_begin();
  ConstActiveAtomIterator end = atoms.active_atoms_end();
  //const DOFmapInterface& dof_map = Hamilton_Constructor->get_const_dof_map(this_domain);

  double temp_value = 0.0;
  unsigned int atom_num = 0;
  for (; it != end; ++it)
  {
    const unsigned int atom_id = it.id();
    std::map<unsigned int, double>::iterator curr_it = HG_density.find(atom_id);
    //atom_resolved_local_current[iteration_index][atom_num] = curr_it->second*-1;
    temp_value += curr_it->second;
    atom_num++;
  }
  slab_resolved_local_current[iteration_index] = temp_value;

  if (energy_resolved)
  {
    //std::map<double, std::map<unsigned int, double>, Compare_double_or_complex_number> energy_resolved_current_data
    //                                                                                  = &energy_resolved_HG_density_for_output;
    std::string suffix = get_output_suffix();    //options.get_option("output_suffix", std::string(""));
    std::string output_file_name_e_resolved = get_name() + "_slab_resolved_energy_current_" + suffix + ".dat";

    //loop through and set header
    if (iteration_index == 0)
    {
      std::string energy_resolved_file_name = get_name() + "_energy.dat";
      std::ofstream energy_resolved_file;
      energy_resolved_file.open(energy_resolved_file_name.c_str());

      energy_resolved_local_current_file << "%\t";
      std::map<double, std::map<unsigned int, double> >::iterator temp_it = energy_resolved_HG_density_for_output.begin();

      for (; temp_it != energy_resolved_HG_density_for_output.end(); ++temp_it)
      {
        energy_resolved_local_current_file << temp_it->first << "[eV] \t";
        energy_resolved_file << temp_it->first << " \n";
      }
      energy_resolved_local_current_file << "\n";
      energy_resolved_local_current_file << iteration_index << " \t";
      energy_resolved_file.close();


    }
    vector<double> temp_energy_resolved(energy_resolved_HG_density_for_output.size(), 0.0);
    int energy_count = 0;
    std::map<double, std::map<unsigned int, double> >::iterator energy_it = energy_resolved_HG_density_for_output.begin();
    for (; energy_it != energy_resolved_HG_density_for_output.end(); ++energy_it)
    {
      //reset atom iterator
      it = atoms.active_atoms_begin();
      double temp_atomic_density = 0.0;
      for (; it != end; ++it)
      {
        const unsigned int atom_id = it.id();
        std::map<unsigned int, double>::iterator curr_it = energy_it->second.find(atom_id);
        temp_atomic_density += curr_it->second;
      }

      //output to file
      temp_atomic_density *= -1 / NemoPhys::elementary_charge; //convert from A to A/eV
      temp_energy_resolved[energy_count] = temp_atomic_density;
      energy_count++;
    }
    for (int i = 0; i < energy_count; i++)
      energy_resolved_local_current_file << temp_energy_resolved[i] << " \t";
    energy_resolved_local_current_file << "\n";
  }

}

void ScatteringBackwardRGFSolver::get_data(const std::string& variable, std::vector<double>& data)
{

  if (variable == "slab_resolved_local_current")
  {
    if (slab_resolved_local_current.size() != subdomain_names.size())
    {
      slab_resolved_local_current.resize(subdomain_names.size(), 0.0);
      output_slab_resolved_current();
    }
    data = slab_resolved_local_current;
  }
}

void ScatteringBackwardRGFSolver::get_data(const std::string& variable, double& data)
{
  if(variable == "current")
  {
    if(!landauer_current)
    {
      if (slab_resolved_local_current.size() != subdomain_names.size())
      {
        slab_resolved_local_current.resize(subdomain_names.size(), 0.0);
        output_slab_resolved_current();
      }
      data = slab_resolved_local_current[0];
    }
    else
      Propagation::get_data(variable,data);
  }
}

void ScatteringBackwardRGFSolver::get_data(const std::string& variable, std::map<std::vector<double>,std::vector<double> >& data,vector<double>*momentum)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("ScatteringBackwardRGFSolver(\""+tic_toc_name+"\")::get_data refineable mesh ");
  NemoUtils::tic(tic_toc_prefix);
  if(variable.find("k_resolved_density")!=std::string::npos)
  {
    if(density_is_ready)
    {

      //k_resolved_density = false;;
      //}
      if(k_resolved_1D_vector.empty())
      {
        std::map<vector<double>, std::map<unsigned int, double> >* k_resolved_density_for_output = NULL;

        if (use_density_for_resonance_mesh)
          k_resolved_density_for_output = &k_resolved_electron_density_for_output;


        //loop through k_resolved_density_for_output and store into flattened vector
        std::map<vector<double>, std::map<unsigned int, double> >::iterator it = k_resolved_density_for_output->begin();
        for(; it!=k_resolved_density_for_output->end(); ++it)
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

      if(solve_greater_greens_function)
      {
        //TODO:JC
        std::map<vector<double>, std::map<unsigned int, double> >* k_resolved_density_for_output = NULL;

        if (use_density_for_resonance_mesh)
          k_resolved_density_for_output = &k_resolved_hole_density_for_output;

        //loop through k_resolved_density_for_output and store into flattened vector
        std::map<vector<double>, std::map<unsigned int, double> >::iterator it = k_resolved_density_for_output->begin();

        for(; it!=k_resolved_density_for_output->end(); ++it)
        {
          std::map<unsigned int, double>::iterator data_it = it->second.begin();
          vector<double> temp_vector(it->second.size());
          int counter = 0;
          for(; data_it != it->second.end(); ++data_it)
          {
            temp_vector[counter] = data_it->second;
            counter++;
          }
          //k_resolved_1D_vector[it->first] = temp_vector;
          //std::vector<double> temp_key_vector(1,it->first);
          //energy_resolved_1D_vector[temp_key_vector] = temp_vector;
          std::map<vector<double>,vector<double> >::iterator it2 = k_resolved_1D_vector.find(it->first);
          for(unsigned int i = 0; i < temp_vector.size(); ++i)
            (it2->second)[i] += temp_vector[i];
        }

      }
      data = k_resolved_1D_vector;
    }
    else
      data.empty();
  }
  else if(variable.find("e_resolved_density")!=std::string::npos)
  {
    //TODO:JC
    //electron density so far
    std::map<double, std::map<unsigned int, double>, Compare_double_or_complex_number>* energy_resolved_density_for_output = NULL;
    if (use_density_for_resonance_mesh)
      energy_resolved_density_for_output = &energy_resolved_electron_density_for_output;

    if(!electron_density.empty())
    {
      if(momentum == NULL) //assume nonrectangular
      {

        //if(electron_density.empty())
        //{
        //   calculate_density(lesser_green_propagator, &electron_density, false);

        //}
        if(energy_resolved_1D_vector.empty())
        {
          msg.threshold(1) << get_name() << ": calling energy resolved electron density for 1D vector \n ";

          //loop through e_resolved_density_for_output and store into flattened vector
          std::map<double, std::map<unsigned int, double> >::iterator it = energy_resolved_density_for_output->begin();
          for(; it!=energy_resolved_density_for_output->end(); ++it)
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
        }

        if(solve_greater_greens_function)
        {
          msg.threshold(1) << get_name() << ": calling energy resolved hole density for 1D vector \n ";
          if (use_density_for_resonance_mesh)
            energy_resolved_density_for_output = &energy_resolved_hole_density_for_output;

          //loop through e_resolved_density_for_output and store into flattened vector
          std::map<double, std::map<unsigned int, double> >::iterator it = energy_resolved_density_for_output->begin();
          for(; it!=energy_resolved_density_for_output->end(); ++it)
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
            //energy_resolved_1D_vector[temp_key_vector] = temp_vector;
            std::map<vector<double>,vector<double> >::iterator it = energy_resolved_1D_vector.find(temp_key_vector);
            for(unsigned int i = 0; i < temp_vector.size(); ++i)
              (it->second)[i] += temp_vector[i];
          }
        }

        data = energy_resolved_1D_vector;
      }
      else
      {

        msg.threshold(1) << get_name() << ": calling energy resolved per k electron density for 1D vector \n ";

        //TODO:JC
        //electron density so far
        std::map<vector<double>, std::map<double, std::map<unsigned int, double> > >* energy_resolved_per_k_density_for_output = NULL;

        if(use_density_for_resonance_mesh)
          energy_resolved_per_k_density_for_output = &energy_resolved_per_k_electron_density_for_output;

        std::map<vector<double>, std::map<double, std::map<unsigned int, double> > > ::iterator k_it = energy_resolved_per_k_density_for_output->find(*momentum);
        //NEMO_ASSERT(k_it!=energy_resolved_per_k_density_for_output->end(), tic_toc_prefix + " did not find momentum in energy resolved per k density");
        if(k_it==energy_resolved_per_k_density_for_output->end())
        {

          cerr << " looking for electron density momentum : (" << (*momentum)[0] << "," << (*momentum)[1] << "," << (*momentum)[2] << ") \n";
          cerr << "map has: \n";
          k_it = energy_resolved_per_k_density_for_output->begin();
          int counter = 0;
          for(k_it=energy_resolved_per_k_density_for_output->begin(); k_it!=energy_resolved_per_k_density_for_output->end(); ++k_it)
          {

            cerr<<"momentum index" << counter << " : (" << k_it->first[0] << "," << k_it->first[1] << "," << k_it->first[2] << ") \n";
            counter++;
          }
          throw std::runtime_error("ScatteringBackwardRGFSolver(\""+get_name()+"\")::  did not find momentum in energy resolved per k electron density \n");

        }

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

        //TODO:JC
        if(solve_greater_greens_function)
        {
          std::map<vector<double>, std::map<double, std::map<unsigned int, double> > >* energy_resolved_per_k_density_for_output = NULL;

          if(use_density_for_resonance_mesh)
            energy_resolved_per_k_density_for_output = &energy_resolved_per_k_hole_density_for_output;

          std::map<vector<double>, std::map<double, std::map<unsigned int, double> > > ::iterator k_it = energy_resolved_per_k_density_for_output->find(*momentum);


          //NEMO_ASSERT(k_it!=energy_resolved_per_k_density_for_output->end(), tic_toc_prefix + " did not find momentum in energy resolved per k density");
          if(k_it==energy_resolved_per_k_density_for_output->end())
          {
            cerr << " looking for hole density momentum : (" << (*momentum)[0] << "," << (*momentum)[1] << "," << (*momentum)[2] << ") \n";
            cerr << "map has: \n";
            k_it = energy_resolved_per_k_density_for_output->begin();
            int counter = 0;
            for(k_it=energy_resolved_per_k_density_for_output->begin(); k_it!=energy_resolved_per_k_density_for_output->end(); ++k_it)
            {

              cerr<<"momentum index" << counter << " : (" << k_it->first[0] << "," << k_it->first[1] << "," << k_it->first[2] << ") \n";
              counter++;
            }
            throw std::runtime_error("ScatteringBackwardRGFSolver(\""+get_name()+"\")::  did not find momentum in energy resolved per k hole density \n");

          }

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
            //energy_resolved_per_k_1D_vector[temp_key_vector] = temp_vector;
            //std::map<vector<double>,vector<double> >::iterator it = energy_resolved_1D_vector.find(temp_key_vector);
            //for(unsigned int i = 0; i < temp_vector.size(); ++i)
            //  (it->second)[i] += temp_vector[i];
            energy_resolved_per_k_1D_vector[temp_key_vector] = temp_vector;

          }
        }

        data = energy_resolved_per_k_1D_vector;
      }
    }
    else
      data.empty();

  }
  NemoUtils::toc(tic_toc_prefix);

}

void ScatteringBackwardRGFSolver::get_atom_blockdiagonal_matrix(PetscMatrixParallelComplex* in_matrix, PetscMatrixParallelComplex*& out_matrix)
{
  std::string tic_toc_prefix = NEMOUTILS_PREFIX("ScatteringBackwardRGFSolver(\""+tic_toc_name+"\")::get_atom_blockdiagonal_matrix ");
  NemoUtils::tic(tic_toc_prefix);

  //1. get sparsity pattern for out_matrix
  //std::set<std::pair<int,int> > set_of_row_col_indices;
  std::map<int, set<int> > map_row_col_indices;

  //loop through all atoms and couple orbitals
  //1.1 loop over all atoms
  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (this->get_simulation_domain());
  const AtomicStructure&  atoms  = domain->get_atoms();
  ConstActiveAtomIterator it     = atoms.active_atoms_begin();
  ConstActiveAtomIterator end    = atoms.active_atoms_end();
  const DOFmapInterface&           dof_map = Hamilton_Constructor->get_dof_map(get_const_simulation_domain());
  //int num_rows = dof_map.get_global_dof_number();
  int counter = 0;
  for (; it != end; ++it)
  {
    //const AtomStructNode& nd = it.node();
    //const Atom*           atom = nd.atom;
    const unsigned int    atom_id   = it.id();
    const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
    //loop over all all orbitals
    map<short, unsigned int>::const_iterator c_it=atom_dofs->begin();

    for(; c_it!=atom_dofs->end(); c_it++)
    {
      //std::pair<int,int> temp_pair(c_it->second,c_it->second);
      //set_of_row_col_indices.insert(temp_pair);
      //std::map<int, set<int> >::iterator map_it = map_row_col_indices.find(c_it->second);
      /*if(map_it == map_row_col_indices.begin())
      {
       std::set<int> temp_set;
       map_it
      }*/
      std::set<int> temp_set;
      map<short, unsigned int>::const_iterator c_it2=atom_dofs->begin();
      //int counter2 = counter+1;
      for(; c_it2!=atom_dofs->end(); c_it2++)
      {
        //std::pair<int,int> temp_pair(c_it->second,c_it2->second);
        //set_of_row_col_indices.insert(temp_pair);
        temp_set.insert(c_it2->second);
      }
      map_row_col_indices[c_it->second] = temp_set;

    }
  }

    int start_local_row;
    int end_local_row;
    in_matrix->assemble();
    in_matrix->get_ownership_range(start_local_row,end_local_row);

    //2. allocate out_matrix
    delete out_matrix;
    //3.1 set up the result matrix
    out_matrix = new PetscMatrixParallelComplex(in_matrix->get_num_rows(),in_matrix->get_num_cols(),
        get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);

    //4.1 loop over the set_of_row_col_indices
    /*std::set<std::pair<int,int> >::const_iterator set_cit=set_of_row_col_indices.begin();
    std::map<int, int> nonzero_map; //key is the row index, value is the count
    std::map<int, int>::iterator nonzero_it = nonzero_map.begin();
    for(; set_cit!=set_of_row_col_indices.end(); set_cit++)
    {
      int row_index = (*set_cit).first;
      //int col_index = (*set_cit).second;
      nonzero_it = nonzero_map.find(row_index);
      if(nonzero_it==nonzero_map.end())
        nonzero_map[row_index] = 1; //first count
      else
        nonzero_it->second += 1; //increment count
    }*/

    out_matrix->set_num_owned_rows(in_matrix->get_num_owned_rows());
    for (int i = 0; i < start_local_row; i++)
      out_matrix->set_num_nonzeros(i,0,0);
    for (int i = start_local_row; i < end_local_row; i++)
    {
      std::map<int, set<int> >::iterator nonzero_it =  map_row_col_indices.find(i);

      NEMO_ASSERT(nonzero_it != map_row_col_indices.end(),tic_toc_prefix + " could not find the number of nonzeros ");

      double num_nonzeros =  nonzero_it->second.size();
      //if(nonzero_it != nonzero_map.end())
      //  num_nonzeros = nonzero_it->second;
      //cerr << " num nonzeros " << num_nonzeros << "\n";
      out_matrix->set_num_nonzeros(i,num_nonzeros,0);
      //result->set_num_nonzeros(i,off_diagonals+1,off_diagonals);
    }
    for (unsigned int i = end_local_row; i < in_matrix->get_num_rows(); i++)
      out_matrix->set_num_nonzeros(i,0,0);
    out_matrix->allocate_memory();
    out_matrix->set_to_zero();


    cplx temp_val(0.0,0.0);
    const std::complex<double>* pointer_to_data= NULL;
    vector<cplx> data_vector;
    vector<int> col_index;
    int n_nonzeros=0;
    const int* n_col_nums=NULL;
    for(int i=(start_local_row); i<(end_local_row); i++)
    {
      std::map<int, set<int> >::iterator nonzero_it =  map_row_col_indices.find(i);
      //in_matrix->get(i,nonzero_it->second);
      in_matrix->get_row(i-start_local_row,&n_nonzeros,n_col_nums,pointer_to_data);
      col_index.resize(nonzero_it->second.size(),0);
      data_vector.resize(nonzero_it->second.size(),cplx(0.0,0.0));
      int counter = 0;
      NEMO_ASSERT(nonzero_it!=map_row_col_indices.end(),tic_toc_prefix + " did not find nonzeros in map_row_col_indices");

      for(int j=0; j<n_nonzeros; j++)
      {
        std::set<int>::iterator set_it = nonzero_it->second.find(n_col_nums[j]);
        if(set_it!=nonzero_it->second.end())
        {
          col_index[counter]=n_col_nums[j]+start_local_row;
          temp_val=pointer_to_data[j];
          data_vector[counter]=temp_val;
          counter++;
        }
      }
      if (n_nonzeros > 0)
      {
        out_matrix->set(i,col_index,data_vector);
      }
      //in_matrix->store_row(i-start_local_row,&n_nonzeros,n_col_nums,pointer_to_data);
    }

    NemoUtils::toc(tic_toc_prefix);
}


void ScatteringBackwardRGFSolver::set_description()
{
  description = "This solver is the library-kind version of the Backward RGF suitable for scattering";
}

void ScatteringBackwardRGFSolver::set_input_options_map()
{
  Greensolver::set_input_options_map();
  //set_input_option_map("OPTION_NAME",InputOptions::Req_Def("OPTION_DESCRIPTION")); //TODO: Set options
  //set_input_option_map("OPTION_NAME",InputOptions::NonReq_Def("DEFAULT_VALUE","OPTION_DESCRIPTION")); //TODO: Set options
}
