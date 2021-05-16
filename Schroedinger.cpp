// NEMO5, The Nanoelectronics simulation package.
// This package is a free software.
// It is distributed under the NEMO5 Non-Commercial License (NNCL).
// Purdue Research Foundation, 1281 Win Hentschel Blvd., West Lafayette, IN 47906, USA
//$Id: Schroedinger.cpp 24174 $

#include "Schroedinger.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <set>
#include <list>
#include <sstream>
#include <numeric> 
#include <iomanip>
#include <mpi.h>
#include "NemoDictionary.h"
#include "NemoMath.h"
#include "NemoPhys.h"
#include "NemoUtils.h"
#include "Nemo.h"
#include "Material.h"
#include "Domain.h"
#include "NemoMesh.h"
#include "HamiltonConstructor.h"
#include "EigensolverSlepc.h"
#include "EigensolverBlockLanczos.h"
#include "EigensolverLanczos.h"
#include "EigensolverLanczosThenSlepc.h"
#include "TBSchroedinger.h"
#include "PropagationUtilities.h"
#ifndef NOLIBMESH
#include "mesh.h" // libmesh
#include "serial_mesh.h"
#include "mesh_generation.h" // libmesh
#endif

#include "PseudomorphicDomain.h" // for lattice constant only
#include "Crystal.h"             // for lattice constant only
#include "HexagonalCrystal.h"    // for lattice constant only

#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/regex.hpp"
#include "Fermi.H"

#ifdef USE_PAPI
extern "C" {
void krp_rpt_init_( int* iam , MPI_Fint* commf , int* hw_counters , long long int* rcy  , long long int* rus , long long int* ucy ,
    long long int* uus );
}
extern long long int krp_rcy, krp_rus, krp_ucy, krp_uus; // defined in Nemo.cpp
extern int krp_hw_counters, krp_rank;
#endif

using namespace std;
using namespace NemoUtils;
using namespace libMesh;

Schroedinger::Schroedinger():
      hamiltonian                 (NULL),   // assemble_hamiltonian()
      overlap                     (NULL),
      overlap_coupling            (NULL),
      //mpi_env                     (NULL),   // base_init()
      kspace                      (new Kspace()),
      analytical_k                (false),  // create_k_space_object()
      k_degeneracy                (1.0),    // base_init()
      potential_calculator        (NULL),   // base_init()
      use_potential               (false),  // base_init()
      _electron_threshold_energy  (0.0),    // set_eigensolver_options()
      _hole_threshold_energy      (0.0),    // set_eigensolver_options()
      _automatic_threshold        (false),  // base_init()
      spatially_varying_threshold (false),  // base_init()
      electron_density            (false),  // read_job_list()
      electron_current            (false),  // read_job_list()<srm>
      ion_density                 (false),  // read_job_list()
      calculate_band_structure    (false),  // read_job_list()
      passivate_H                 (false),  // read_job_list()
      assemble_H                  (false),  // read_job_list()
      combine_H                   (false),  //combine the Hamiltonian instead of assembling it
      output_H_all_k    (false),  //
      derive_electron_density_over_potential(false), // read_job_list()
      derive_hole_density_over_potential    (false), // read_job_list()
      include_strain_H            (false),  // read_job_list()
      include_shear_strain_H      (false),  // read_job_list()
      constant_strain             (false),  // read_job_list()
      spatial_density_of_states   (false),  // read_job_list()
      density_of_states           (false),  // read_job_list()
      project_density_of_states   (false),  // Optio_list()
      energy_gradient             (false),  // read_job_list()
      matrix_vector_mult_time     (false),  // read_job_list()
      block_matrix_vector_mult_time (false),  // read_job_list()
      spin_projection             (false),  // read_job_list()
      scattering_probability      (false),  // read_job_list()
      calculate_Fermi_level (false),  // read_job_list()
      output_precision            (12),     // base_init()
      DOS_broadening_model        (2),      // base_init()
      DOS_gamma                   (1e-3),   // base_init()
      gather_density_everywhere   (true),
      atom_resolved_position_operator (true)
{
  last_valley = "DEFAULT";
  solver = NULL;
  k_space_dimensionality = 0;
  hole_current = false;
  hole_density = false;
  DOS_spin_factor = 1.0;
  overlap_coupling = NULL;
  //PDOS =NULL;
}


Schroedinger::~Schroedinger()
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::~Schroedinger";
  NemoUtils::tic(tic_toc_prefix);

  msg.print_message(MSG_INFO,"Schroedinger(\""+this->get_name()+"\"): destroying...");

  if (hamiltonian) delete hamiltonian;
  if (kspace)      delete kspace;

  if (overlap_coupling)
  {
    delete overlap_coupling;
    overlap_coupling = NULL;
  }
  //if(overlap)
  //{
  //  delete overlap;
  //  overlap=NULL;
  //}

  delete solver;

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::base_reinit()
{

  //reinit_material_properties();

  reuse_hamiltonian = true;
  string opt_type = options.get_option("reuse_hamiltonian", string("true"));
  if (opt_type == "true" || opt_type == "FullHamiltonian")
  {
    reuse_type = FullHamiltonian;
  }
  else if (opt_type == "SubstractElementPotential")
  {
    reuse_type = SubstractElementPotential;
  }
  else if (opt_type == "SubstractDiagonalPotential")
  {
    reuse_type = SubstractDiagonalPotential;
  }
  else if (opt_type == "false")
  {
    reuse_type = Disabled;
    reuse_hamiltonian = false;
  }
  else
  {
    reuse_type = Disabled;
    reuse_hamiltonian = false;
  }


  output_precision = options.get_option("output_precision", 12);


  //DM: this is required for Schroedinger reinitialization
  //create eigenvalue solver
  {
    string eigen_values_solver = options.get_option("eigen_values_solver", string("lapack"));
    if (eigen_values_solver=="lanczos")
    {
      solver = new EigensolverLanczos();
    }
    else if(eigen_values_solver=="block_lanczos")
    {
      solver = new EigensolverBlockLanczos();
    }
    else if(eigen_values_solver=="lanczos_then_krylovschur")    //JC: added for lanczoz_then_krylovschur eigensolver;
    {
      solver = new EigensolverLanczosThenSlepc();
    }
    else if(eigen_values_solver=="feast")
    {
#ifndef FEAST_UNAVAILABLE
      solver = new EigensolverFeastComplex();
#endif
    }
    else
    {
      solver = new EigensolverSlepc();
    }
  }


  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::base_reinit";
  NemoUtils::tic(tic_toc_prefix);

  msg.print_message(MSG_INFO,"[Schroedinger(\""+get_name()+"\")] re-initializing...");

  output_precision = options.get_option("output_precision", 12);






  // -----------------------------------
  // read in the parameters and job list
  // -----------------------------------
  read_job_list();

  // read what kind of output should be done
  vector<string> output_list_v;
  options.get_option("output",output_list_v);
  output_list.clear();
  for (unsigned int i = 0; i < output_list_v.size(); i++)
    output_list.insert(output_list_v[i]);

  //Roza Kotlyar: 3/11/2011/
  // moved determinations of constant strain to
  // here to modify the k-grid by strain if constant_strain is requested
  if (constant_strain)
  {//constant strain
    if(options.check_option("epsilon_matrix"))
    {
      options.get_option("epsilon_matrix",strain_matrix);

      if(strain_matrix.size()==3)
      {
        for (unsigned int i=0; i<3; i++)
        {
          if(strain_matrix[i].size()!=3)
            throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix as a 3*3 Matrix over real numbers\n");
          strain_matrix[i][i]=1.0+strain_matrix[i][i];
        }

        //Roza Kotlyar: 3/7/2011
        //adding output of strain matrix in this case
        Tensor2Sym eps_lab ;
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j <= i; j++)
          {
            eps_lab(i+1, j+1) = strain_matrix[i][j];
          }
        }
        msg << "User defined strain tensor in the lab system\n" << std::setprecision(8)<<std::setw(18) << eps_lab;
      }
      else
        throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix in three dimensions\n");
    }
    else if(options.check_option("epsilon_matrix_crystal"))
    {
      options.get_option("epsilon_matrix_crystal",strain_matrix);

      if(strain_matrix.size()!=3)
        throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix_crystal as a 3*3 Matrix over real numbers\n");

      for (unsigned int i=0; i<3; i++)
        strain_matrix[i][i]=1.0+strain_matrix[i][i];

      //  do the transformation
      const Tensor2Gen& R = dynamic_cast<const AtomisticFemBase*>(get_simulation_domain())->get_rotation_matrix ();
      Tensor2Sym eps_lab ;

      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j <= i; j++)
        {
          eps_lab(i+1, j+1) = strain_matrix[i][j];
        }
      }

      msg << "User defined strain tensor in the crystal system\n" << std::setprecision(8)<<std::setw(18) << eps_lab;

      eps_lab = sym(R * eps_lab * R.transpose());

      msg << "Strain tensor is converted to the laboratory system\n" << std::setprecision(8) << std::setw(18) << eps_lab;
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j <=i; j++)
        {
          strain_matrix[i][j] = eps_lab(i+1, j+1);
          strain_matrix[j][i] = eps_lab(i+1, j+1);
        }
      }
    }
    else
      throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix or epsilon_matrix_crystal or do not use constant_strain\n");

    //M.P. will modify domain translations here
    get_simulation_domain()->transform_translations(strain_matrix);



  }

  bool create_new_k_space = false; //M.P. Another hack to change the k-space at a reinit level.
  create_new_k_space = options.get_option("create_k_space_at_reinit", false);


  //Hack to allow change in k_points for E(k) grid when do_solve of band structure is called from adaptive grid for each k (Bozidar)
  if (calculate_band_structure && options.get_option("allow_reinit_k_points",false))
  {//hack
    if (options.check_option("k_points"))
    {
      vector< vector<double> > k_pointsV;
      vector<unsigned int>    number_of_nodesV;

      msg<< "entered K-points\n";//rv

      // calculate multiple k-points along some path in k-space
      // k_points parameter may contain:
      // - a single point (only 1 k-point is solved)
      // - two points A,B (the segment from A to B is solved)
      // - more than two points A,B,C,D,... (the segments A-B, B-C, C-D etc. are solved)
      options.get_option("k_points",k_pointsV);
      NEMO_ASSERT(k_pointsV.size()>0, "[Schroedinger(\""+get_name()+"\")] k_points parameter seemed to be empty.");

      // discretization of segments: number_of_nodesV
      if(options.check_option("number_of_nodes"))
      {
        options.get_option("number_of_nodes", number_of_nodesV);
        NEMO_ASSERT((number_of_nodesV.size()==k_pointsV.size()-1)||(number_of_nodesV.size()==1 && number_of_nodesV[0]==1 &&
            k_pointsV.size()==1), "[Schroedinger(\""+get_name()+"\")] need to have 1 number_of_nodes entry for each k-segment.\n");
        if (number_of_nodesV.size()==1 && number_of_nodesV[0]==1 && k_pointsV.size()==1)
          number_of_nodesV.clear();
      }
      else if (k_pointsV.size()!=1)
      {
        msg << "[Schroedinger(\""+get_name()+"\")] did not find parameter number_of_nodes in input deck - assume  points per k-segment.\n";
        number_of_nodesV.resize(k_pointsV.size()-1, 11);
      }

      kspace->clean_points();

      kspace->set_mesh_boundary(k_pointsV);
      kspace->set_segmental_mesh(true);

      for (unsigned int ii=0; ii<number_of_nodesV.size(); ii++)
      {
        kspace->set_segmental_num_points(ii, number_of_nodesV[ii]);
      }
      kspace->calculate_mesh();
      msg << "K-mesh done\n";//rv

      // transform k_space into cartesian coordinates (if necessary)
      string k_space_basis = options.get_option("k_space_basis",string("reciprocal"));
      if (k_space_basis!="cartesian")
      {
        msg<<"transform k_space into cartesian coordinates (if necessary)\n";//rv
        vector <vector<double> > transformation_matrix;
        unsigned int number_of_rec_vectors=get_simulation_domain()->get_reciprocal_vectors().size();

        //cerr << number_of_rec_vectors << "\n";
        vector<vector<double> > reciprocal_vectors_strained;

        transformation_matrix.resize(3);
        for (unsigned int i=0; i<3; i++)
        {
          transformation_matrix[i].resize(number_of_rec_vectors,0.0);
          for (unsigned int j=0; j<number_of_rec_vectors; j++)
            // rvedula if statements to correct k-space for constant strain
            if(constant_strain)
            {

              transformation_matrix[i][j]= reciprocal_vectors_strained[j][i]; //transpose transformation==inverse transformation for orthonormal vectors

            }
            else
              transformation_matrix[i][j]=
                  get_simulation_domain()->get_reciprocal_vectors()[j][i]; //transpose transformation==inverse transformation for orthonormal vectors
        }
        kspace->transform_basis(transformation_matrix);
      }
    }
    else if (options.get_option("k_mesh",false))
    {
      //Allow reading a vector of k_points directly (no LIBMESH grid generation) (Bozidar)
      if (!options.check_option("k_points_vector"))
      {
        int number_of_k_points = options.get_option("number_of_k_points",10);

        //Allow different number of points in different directions (Bozidar)
        int number_of_kx_points = options.get_option("number_of_kx_points",number_of_k_points);
        int number_of_ky_points = options.get_option("number_of_ky_points",number_of_k_points);
        int number_of_kz_points = options.get_option("number_of_kz_points",number_of_k_points);

        double kxmin;
        double kymin;
        double kzmin;
        double kxmax;
        double kymax;
        double kzmax;

        string k_space_basis = options.get_option("k_space_basis",string("reciprocal"));
        const vector< vector<double> >& rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
#ifndef NOLIBMESH
	Mesh mesh(Parallel::Communicator(MPI_COMM_SELF));

        //Allow input in Cartesian coordinates (Bozidar)
        if (k_space_basis!="cartesian")
        {
          kxmin = options.get_option("kxmin", 0.0);
          kymin = options.get_option("kymin", 0.0);
          kzmin = options.get_option("kzmin", 0.0);
          kxmax = options.get_option("kxmax", 1.0);
          kymax = options.get_option("kymax", 1.0);
          kzmax = options.get_option("kzmax", 1.0);

          mesh.set_mesh_dimension(rec_lattice.size());
#if LIBMESH_MAJOR_VERSION < 1
          mesh.partitioner() =  AutoPtr<Partitioner>(NULL);  //no mesh partitioning!
#endif
          if (rec_lattice.size()==1)
          {
            MeshTools::Generation::build_line(mesh, number_of_kx_points, kxmin, kxmax, EDGE2);
          }
          else if (rec_lattice.size()==2)
          {
            MeshTools::Generation::build_square(mesh, number_of_kx_points, number_of_ky_points, kxmin, kxmax, kymin, kymax,  QUAD4);
          }
          else if (rec_lattice.size()==3)
          {
            MeshTools::Generation::build_cube(mesh, number_of_kx_points, number_of_ky_points, number_of_kz_points, kxmin, kxmax, kymin, kymax, kzmin, kzmax,  HEX8);
          }
          else throw invalid_argument("[Schroedinger(\""+get_name()+"\"):create_k_space_object] wrong dimensionality of reciprocal_lattice!\n");
        }
        else
        {
          kxmin = options.get_option("kxmin", 0.0);
          kymin = options.get_option("kymin", 0.0);
          kzmin = options.get_option("kzmin", 0.0); 
          kxmax = options.get_option("kxmax", 0.0);
          kymax = options.get_option("kymax", 0.0);
          kzmax = options.get_option("kzmax", 0.0);

          std::vector<double> kminmax;
          std::vector<double> num_k_points;
          if (kxmin != kxmax)
          {
            kminmax.push_back(kxmin);
            kminmax.push_back(kxmax);
            num_k_points.push_back(number_of_kx_points);
          }
          if (kymin != kymax)
          {
            kminmax.push_back(kymin);
            kminmax.push_back(kymax);
            num_k_points.push_back(number_of_ky_points);
          }
          if (kzmin != kzmax)
          {
            kminmax.push_back(kzmin);
            kminmax.push_back(kzmax);
            num_k_points.push_back(number_of_kz_points);
          }

          NEMO_ASSERT(kminmax.size()/2 > 0, "[Schroedinger(\""+get_name()+"\"):create_k_space_object()] k-mesh must at leat be one-dimensional.\n");

          mesh.set_mesh_dimension(kminmax.size()/2);
#if LIBMESH_MAJOR_VERSION < 1
          mesh.partitioner() =  AutoPtr<Partitioner>(NULL);  //no mesh partitioning!
#endif
          if (kminmax.size()/2==1)
          {
            MeshTools::Generation::build_line(mesh, num_k_points[0], kminmax[0], kminmax[1], EDGE2);
          }
          else if (kminmax.size()/2==2)
          {
            MeshTools::Generation::build_square(mesh, num_k_points[0], num_k_points[1], kminmax[0], kminmax[1], kminmax[2], kminmax[3],  QUAD4);
          }
          else if (kminmax.size()/2==3)
          {
            MeshTools::Generation::build_cube(mesh, num_k_points[0], num_k_points[1], num_k_points[2], kminmax[0], kminmax[1],
                kminmax[2], kminmax[3], kminmax[4], kminmax[5],  HEX8);
          }
        }

        integrator.set_mesh(&mesh);
        int integration_order  = options.get_option("integration_order", 1);
        integrator.set_order(integration_order);
        integrator.build_integral_sum();
        const vector<vector<double> >& x = integrator.get_points();

        kspace->clean_points();

        int n = x.size();
        for (int i=0; i<n; i++)
        {
          vector<double> k_pointV(3,0.0);
          if (k_space_basis!="cartesian") //If cartesian is input, no need to transform (Bozidar)
          {
            for (unsigned int i1=0; i1<rec_lattice.size(); i1++)
            {
              for (unsigned int i2=0; i2<3; i2++)
              {
                k_pointV[i2]+=x[i][i1]*rec_lattice[i1][i2];
              }
            }
          }
          else
          {
            double dim = 0;
            if (kxmin == kxmax)
              k_pointV[0] = kxmin;
            else
              k_pointV[0] = x[i][dim++];
            if (kymin == kymax)
              k_pointV[1] = kymin;
            else
              k_pointV[1] = x[i][dim++];
            if (kzmin == kzmax)
              k_pointV[2] = kzmin;
            else
              k_pointV[2] = x[i][dim];
          }
          kspace->add_point(k_pointV);
        }

#else
        NEMO_EXCEPTION("[Schroedinger(\""+get_name()+"\")] NEMO5 was compiled without libmesh.");
#endif
      }
      else
      {
        std::vector<std::vector<double> > k_points;
        options.get_option("k_points_vector", k_points);

        NEMO_ASSERT(options.get_option("k_space_basis",string("reciprocal")) == string("cartesian"), "[Schroedinger(\""+get_name()+"\")] with k_points_vector option" +
            " cartesian basis is required for now.\n");
        NEMO_ASSERT(k_points.size() >= 1, "[Schroedinger(\""+get_name()+"\")] k_points_vector must have at least one point.\n");

        kspace->clean_points();
        for (unsigned int i = 0; i < k_points.size(); i++)
        {
          NEMO_ASSERT(k_points[i].size() == 3, "[Schroedinger(\""+get_name()+"\")] k_points_vector must contain 3-dimensional k-points.\n");
          kspace->add_point(k_points[i]);
        }
      }
    }
  }
  else
  {
    if (create_new_k_space)
    {
      delete kspace; 
      kspace = new Kspace();
      create_k_space_object();
    }
  }


  // ------------------------------
  // assign potential_calculator
  // ------------------------------
  string potential_sim_name = options.get_option("potential_solver", string(""));
  if (potential_sim_name != string(""))
  {
    //To use read from .xyz potential energy
    potential_calculator = this->find_simulation(potential_sim_name);
    NEMO_ASSERT(potential_calculator != NULL, "[Schroedinger(\""+get_name()+"\")] wrong potential_solver");
    // use an electrostatic potential if we can find one
    use_potential = true;
  }


  bool parallelize_here = options.get_option("parallelize_here", true); //must be true
  bool reinit_parallelize = options.get_option("reinit_parallelize", false); //must be true:
  if ((parallelize_here && reinit_parallelize) || (parallelize_here && create_new_k_space))
  {
    MPI_Comm t_communicator;
    int gran = options.get_option("shadow_copies", 1);
    //get the total number of k-points
    //vector<unsigned int> number_of_nodesV;
    //options.get_option("number_of_nodes", number_of_nodesV);
    //unsigned int temp_int=number_of_nodesV.size();
    //if(temp_int>0)
    //  temp_int=number_of_nodesV[0];
    unsigned int temp_int=kspace->get_mesh_points().size();
    PropagationUtilities::prepare_communicator_for_parallelization(this,get_simulation_domain()->get_one_partition_total_communicator(), 
      t_communicator, calculate_band_structure, gran, temp_int);

    std::vector<std::string> Mesh_tree_names(1,std::string("Schroedinger_k_mesh"));
    std::map<std::string, NemoMesh*> Momentum_meshes;
    Momentum_meshes[Mesh_tree_names[0]]=kspace;
    std::map<std::string, Simulation*> Mesh_Constructors;
    Mesh_Constructors[Mesh_tree_names[0]]=this;
    std::map<std::string, NemoPhys::Momentum_type> momentum_mesh_types;
    momentum_mesh_types[Mesh_tree_names[0]]=NemoPhys::Momentum_2D;
    std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
    std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
    PropagationUtilities::set_parallel_environment(this, Mesh_tree_names, Mesh_tree_downtop,Mesh_tree_topdown,
      Momentum_meshes,momentum_mesh_types,Mesh_Constructors,this, t_communicator);
    kspace->get_communicator(k_space_communicator);
  }


  // is the electron / hole discrimination threshold set autmatically by NEMO 5?
  _automatic_threshold = options.get_option("automatic_threshold", false);

  // is that threshold a global constant or spatially varying (with the potential)?
  spatially_varying_threshold = options.get_option("spatially_varying_threshold", false);


  gather_density_everywhere = !(options.get_option("solve_on_single_replica", false));

  NemoUtils::toc(tic_toc_prefix);
}



void Schroedinger::base_init()
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::base_init";
  NemoUtils::tic(tic_toc_prefix);

  msg.print_message(MSG_INFO,"[Schroedinger(\""+get_name()+"\")] initializing...");

  //DM replaced: reuse_hamiltonian = options.get_option("reuse_hamiltonian", true);
  reuse_hamiltonian = true;
  string opt_type = options.get_option("reuse_hamiltonian", string("true"));
  if (opt_type == "true" || opt_type == "FullHamiltonian")
  {
    reuse_type = FullHamiltonian;
  }
  else if (opt_type == "SubstractElementPotential")
  {
    reuse_type = SubstractElementPotential;
  }
  else if (opt_type == "SubstractDiagonalPotential")
  {
    reuse_type = SubstractDiagonalPotential;
  }
  else if (opt_type == "false")
  {
    reuse_type = Disabled;
    reuse_hamiltonian = false;
  }
  else
  {
    reuse_type = Disabled;
    reuse_hamiltonian = false;
  }


  output_precision = options.get_option("output_precision", 12);

  //Roza Kotlyar: 3/2/2011
  //for calculation of integrated densities
  do_electron_integrated_den=false;
  do_hole_integrated_den=false;
  electron_integrated_den=0.0;
  hole_integrated_den=0.0;
  rspace_multfact=1.0;

  //for automatic shift for esolver using the previous iteration solution
  do_esolvershift_fromsolution = false;
  done_firstiter_esolvershift_fromsolution = 0.0;

  eminecsub=0.0;
  emaxecsub=0.0;
  eminevsub=0.0;
  emaxevsub=0.0;
  eminecsubiter=0.0;
  emaxecsubiter=0.0;
  eminevsubiter=0.0;
  emaxevsubiter=0.0;
  first_k_point=0;
  dnk=0.0;
  nsetopt=0.0;
  my_rank_sch=0;

  kmagref_foresolvershift = 0.0;
  kmagreftol_foresolvershift = 0.001;
  enthresh_foresolvershift = 0.1;
  enrate_foresolvershift = 0.2;
  //create eigenvalue solver
  {
    string eigen_values_solver = options.get_option("eigen_values_solver", string("lapack"));
    if (eigen_values_solver=="lanczos")
    {
      solver = new EigensolverLanczos();
    }
    else if(eigen_values_solver=="block_lanczos")
    {
      solver = new EigensolverBlockLanczos();
    }
    else if(eigen_values_solver=="lanczos_then_krylovschur")    //JC: added for lanczos_then_krylovschur eigensolver;
    {
      solver = new EigensolverLanczosThenSlepc();
    }
    else if(eigen_values_solver=="feast")
    {
#ifndef FEAST_UNAVAILABLE
      solver = new EigensolverFeastComplex();
#endif
    }
    else
    {
      solver = new EigensolverSlepc();
    }
  }



  // -----------------------------------
  // read in the parameters and job list
  // -----------------------------------
  read_job_list();

  // read what kind of output should be done
  vector<string> output_list_v;
  options.get_option("output",output_list_v);
  for (unsigned int i = 0; i < output_list_v.size(); i++)
    output_list.insert(output_list_v[i]);

  //Roza Kotlyar: 3/11/2011/
  // moved determinations of constant strain to
  // here to modify the k-grid by strain if constant_strain is requested
  if (constant_strain)
  {
    if(options.check_option("epsilon_matrix"))
    {
      options.get_option("epsilon_matrix",strain_matrix);

      if(strain_matrix.size()==3)
      {
        for (unsigned int i=0; i<3; i++)
        {
          if(strain_matrix[i].size()!=3)
            throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix as a 3*3 Matrix over real numbers\n");
          strain_matrix[i][i]=1.0+strain_matrix[i][i];
        }

        //Roza Kotlyar: 3/7/2011
        //adding output of strain matrix in this case
        Tensor2Sym eps_lab ;
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j <= i; j++)
          {
            eps_lab(i+1, j+1) = strain_matrix[i][j];
          }
        }
        msg << "User defined strain tensor in the lab system\n" << std::setprecision(8)<<std::setw(18) << eps_lab;
      }
      else
        throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix in three dimensions\n");
    }
    else if(options.check_option("epsilon_matrix_crystal"))
    {
      options.get_option("epsilon_matrix_crystal",strain_matrix);

      if(strain_matrix.size()!=3)
        throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix_crystal as a 3*3 Matrix over real numbers\n");

      for (unsigned int i=0; i<3; i++)
        strain_matrix[i][i]=1.0+strain_matrix[i][i];

      //  do the transformation
      const Tensor2Gen& R = dynamic_cast<const AtomisticFemBase*>(get_simulation_domain())->get_rotation_matrix ();
      Tensor2Sym eps_lab ;

      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j <= i; j++)
        {
          eps_lab(i+1, j+1) = strain_matrix[i][j];
        }
      }

      msg << "User defined strain tensor in the crystal system\n" << std::setprecision(8)<<std::setw(18) << eps_lab;

      eps_lab = sym(R * eps_lab * R.transpose());

      msg << "Strain tensor is converted to the laboratory system\n" << std::setprecision(8) << std::setw(18) << eps_lab;
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j <=i; j++)
        {
          strain_matrix[i][j] = eps_lab(i+1, j+1);
          strain_matrix[j][i] = eps_lab(i+1, j+1);
        }
      }
    }
    else
      throw invalid_argument("Schroedinger::base_init: define parameter epsilon_matrix or epsilon_matrix_crystal or do not use constant_strain\n");
  }

  // ------------------------------
  // initialize k-space
  // ------------------------------
  create_k_space_object();

  //Roza Kotlyar: 3/2/2011
  //initialize doing shift for esolver from solution if requested
  do_esolvershift_fromsolution = options.get_option("calculate_automatic_shift_subbandbased", false);
  if(do_esolvershift_fromsolution)
  {
    kmagref_foresolvershift = options.get_option("kpoint_to_monitor_subbands_for_shift", kmagref_foresolvershift);
    kmagreftol_foresolvershift = options.get_option("kpoint_tolerance_to_monitor_subbands_for_shift", kmagreftol_foresolvershift);
    enthresh_foresolvershift = options.get_option("energy_threshold_for_shift",enthresh_foresolvershift);
    enrate_foresolvershift = options.get_option("rate_of_shift_changes_between_iterations",enrate_foresolvershift);
  }

  // each k-point is assumed to be equivalent to k_degeneracy points in total (of which only 1 is simulated)
  // used in density calculation
  k_degeneracy = options.get_option("k_degeneracy", 1.0);

  // ------------------------------
  // assign potential_calculator
  // ------------------------------
  string potential_sim_name = options.get_option("potential_solver", string(""));
  if (potential_sim_name != string(""))
  {
    //To use read from .xyz potential energy
    potential_calculator = this->find_simulation(potential_sim_name);
    NEMO_ASSERT(potential_calculator != NULL, "[Schroedinger(\""+get_name()+"\")] wrong potential_solver");
    // use an electrostatic potential if we can find one
    use_potential = true;
  }

  // is the electron / hole discrimination threshold set autmatically by NEMO 5?
  _automatic_threshold = options.get_option("automatic_threshold", false);

  // is that threshold a global constant or spatially varying (with the potential)?
  spatially_varying_threshold = options.get_option("spatially_varying_threshold", false);

  // ------------------------------
  // set parallel environment
  // ------------------------------
  kspace->set_strategy(MPIVariable::scatter);
  kspace->set_num_points(kspace->get_num_points()); //obsolete?
  int granularity;

  NEMO_ASSERT(get_simulation_domain()->get_communicator() != MPI_COMM_NULL,
      "[Schroedinger(\""+get_name()+"\")] get_simulation_domain()->get_communicator() was MPI_COMM_NULL.");
  MPI_Comm_size(get_simulation_domain()->get_communicator(), &granularity);


  int total_size;
  //MPI_Comm_size(holder.total_communicator, &total_size);
  MPI_Comm_size(get_simulation_communicator(), &total_size);


  kspace->set_granularity(granularity);
  for (unsigned int i=0; i<kspace->get_num_points(); i++)
    kspace->set_load(i,1);  //equal load for all k-points

  NEMO_ASSERT(/*holder.total_communicator*/get_simulation_communicator() != MPI_COMM_NULL,
      "[Schroedinger(\""+get_name()+"\")] get_simulation_communicator() was MPI_COMM_NULL.");

  bool parallelize_here = options.get_option("parallelize_here", true); //must be true
  if (parallelize_here)
  {
    MPI_Comm t_communicator;
    int gran = options.get_option("shadow_copies", 1);
    /*vector<unsigned int> number_of_nodesV;
    options.get_option("number_of_nodes", number_of_nodesV);*/
    unsigned int temp_int=kspace->get_mesh_points().size();
    PropagationUtilities::prepare_communicator_for_parallelization(this,get_simulation_domain()->get_one_partition_total_communicator(), 
      t_communicator, calculate_band_structure, gran, temp_int);

    //MPI_Comm t_communicator, ori_comm = get_simulation_domain()->get_one_partition_total_communicator();
    //int gran = options.get_option("shadow_copies", 1);
    //int local_rank, local_size;
    //MPI_Comm_rank(ori_comm, &local_rank);
    //MPI_Comm_size(ori_comm, &local_size);
    //if (local_size%gran != 0)
    //{
    //  throw invalid_argument("[Schroedinger(\""+get_name()+"\")] Number of cores needs to be a multiple of granularity\n");
    //}
    ////check the number of geometry partitioning
    //MPI_Comm temp_communicator=get_const_simulation_domain()->get_communicator();
    //int temp_size;
    //MPI_Comm_size(temp_communicator,&temp_size);
    //if(gran==1 && calculate_band_structure && temp_size==1 && options.get_option("enable_hack",true))
    //{
    //  //get the total number of k-points //corrected by M.P.


    //  vector<unsigned int> number_of_nodesV;
    //  options.get_option("number_of_nodes", number_of_nodesV);
    //  int number_of_nodes=kspace->get_mesh_points().size(); // number_of_nodesV[0];
    //  // for(unsigned int i=1; i<number_of_nodesV.size(); i++)
    //  //  number_of_nodes+=number_of_nodesV[i];
    //  //check whether the number of CPUs is larger than the number_of_nodes (for bandstructure calulations)
    //  if(local_size>number_of_nodes)
    //  {
    //    MPI_Comm_split(ori_comm, local_rank/number_of_nodes, local_rank%number_of_nodes, &t_communicator);
    //  }
    //  else
    //    MPI_Comm_split(ori_comm, local_rank%gran, local_rank/gran, &t_communicator);
    //}
    //else
    //  MPI_Comm_split(ori_comm, local_rank%gran, local_rank/gran, &t_communicator);
    
    //this->_top_mpi_variable = new MPIEnvironment(t_communicator);
    //dynamic_cast<MPIEnvironment*>(_top_mpi_variable)->set_variable(kspace);
    //msg.print_message(MSG_INFO,"[Schroedinger(\""+get_name()+"\")] parallelizing...");
    //dynamic_cast<MPIEnvironment*>(_top_mpi_variable)->parallelize();
    //_top_mpi_variable->display_info();
    //kspace->get_communicator(k_space_communicator);

    std::vector<std::string> Mesh_tree_names(1,std::string("Schroedinger_k_mesh"));
    std::map<std::string, NemoMesh*> Momentum_meshes;
    Momentum_meshes[Mesh_tree_names[0]]=kspace;
    std::map<std::string, Simulation*> Mesh_Constructors;
    Mesh_Constructors[Mesh_tree_names[0]]=this;
    std::map<std::string, NemoPhys::Momentum_type> momentum_mesh_types;
    momentum_mesh_types[Mesh_tree_names[0]]=NemoPhys::Momentum_2D;
    std::map<NemoMesh*, NemoMesh* > Mesh_tree_downtop;
    std::map<NemoMesh*,std::vector<NemoMesh*> > Mesh_tree_topdown;
    PropagationUtilities::set_parallel_environment(this, Mesh_tree_names, Mesh_tree_downtop,Mesh_tree_topdown,
      Momentum_meshes,momentum_mesh_types,Mesh_Constructors,this,t_communicator);

    kspace->get_communicator(k_space_communicator);
    //this->_top_mpi_variable = mpi_env;
    this->_parallelized_by_this = true;
  }
  else
  {
    msg.print_message(MSG_INFO,"[Schroedinger(\""+get_name()+"\")] parallelization is done outside.");
    k_space_communicator = MPI_COMM_SELF;//was MPI_COMM_NULL
    this->_top_mpi_variable = kspace;
    this->_parallelized_by_this = false;
  }

  if (density_of_states || spatial_density_of_states)
  {
    vector<double> energy_input;
    if (options.check_option("DOS_energy_grid"))
      options.get_option("DOS_energy_grid", energy_input);
    else
      throw invalid_argument("[Schroedinger(\""+get_name()+"\")] DOS_energy_grid option (Emin, Emax, dE) must be given for DOS calculation\n");

    if (energy_input.size() == 3)
    {
      double Emin = energy_input[0];
      double Emax = energy_input[1];
      double dE = energy_input[2];

      int NE = ceil((Emax - Emin)/dE) + 1;
      msg << "[Schroedinger(\""+get_name()+"\")] energy grid for density of states (DOS): Emin=" << Emin << ", Emax=" << Emax << ", dE=" << dE << ", NE=" <<
          NE << "\n";
      DOS_broadening_model = options.get_option("DOS_broadening_model",2);
      DOS_gamma = options.get_option("DOS_broadening", dE);
      DOS_spin_factor = options.get_option("DOS_spin_factor", 1.0);
      msg << "[Schroedinger(\""+get_name()+"\")] Lorentzian broadening for DOS: " << DOS_gamma << "\n";

      DOS_energy_grid.resize(NE);

      for (int i = 0; i < NE; i++)
        DOS_energy_grid[i] = Emin + dE * i;

      if (spatial_density_of_states)
      {
        spatially_resolved_DOS.clear();
        spatially_resolved_DOS.resize(NE);
      }
    }
    else
    {
      throw invalid_argument("[Schroedinger(\""+get_name()+"\")] DOS_energy_grid must contain 3 arguments \n");
    }
  }

  //==================================================
  //Roza Kotlyar: 3/2/2011
  //read the shift to pass to eigensolver if requested for the restart option
  ifinitialized_shift_foresolver = false;
  shift_foresolver = 0.0;
  bool do_input_nlp_restartinfo  = options.get_option("input_shift_subbandbased_for_restart",false);
  if(do_input_nlp_restartinfo)
  {
    string fileoutname = options.get_option("input_shiftfilename_for_restart", string(""));
    if(fileoutname == "")
    {
      throw runtime_error("Schroedinger:: file name for restarting info for eigensolver is not provided\n");
    }

    ifinitialized_shift_foresolver = true;
    double valtemp;
    int places=15;
    std::ifstream nlpaf_file(fileoutname.c_str(),std::ios_base::in);
    nlpaf_file.precision(places);

    nlpaf_file >> valtemp;
    shift_foresolver=valtemp;
    nlpaf_file.close();

    msg << "[Schroedinger(\""+get_name()+"\")] read restart info from a file " << fileoutname <<" shift =  "<<shift_foresolver<<".\n";
  }


  gather_density_everywhere = !(options.get_option("solve_on_single_replica", false));


  //M.P. If not calculate_Fermi_level Ef has to be set to something
  {
    Ef = options.get_option("chem_pot", 0.0);
  }


  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::Compute_k_mesh_from_file(string  filename, vector< vector<double> > & Kvec, vector<double> & Weight)
{
  std::string line;
  ifstream myfile (filename.c_str());
  if (myfile.is_open())
  {
    int nk;
    //double w_temp;
    int num=0;
    int temp=0;
    NEMO_ASSERT(options.check_option("number_of_k_points"), "Schroedinger::Compute_k_mesh_from_file please define the number_of_k_points \n");
    //int num_kpoints = options.get_option("number_of_k_points", 0);
    vector<double> temp_Weight;

    //first two lines are comments
    getline(myfile,line);
    getline(myfile,line);
    {
      nk=std::atoi(line.c_str());
      Kvec.resize(nk, vector<double>(3));
      Weight.resize(nk);
      temp_Weight.resize(nk);
    }
    getline(myfile,line);
    while (! myfile.eof() )
    {
      std::vector<std::string> v(0);
      getline(myfile,line);
      string trimmed_line = boost::algorithm::trim_copy(line);
      /*
    boost::regex pattern{"[[:space:]]+", boost::regex_constants::egrep};
    string result = boost::regex_replace(trimmed_line, pattern, " ");

    vector<string> split_string;
    boost::algorithm::split(split_string, result,  boost::is_any_of(" "), boost::token_compress_on );
          for(unsigned int k =0; k<split_string.size()-1; k++)
      Kvec[num][k]=std::atof( (split_string[k]).c_str() );
          temp_Weight[num]=std::atof(split_string[split_string.size()-1].c_str() );
          temp +=std::atoi(split_string[split_string.size()-1].c_str() );
       */
      num +=1;
    }
    myfile.close();
    for(int k =0; k<nk; k++)
      Weight[k]=temp_Weight[k]/temp ; //*pow(num_kpoints,k_space_dimensionality)
  }
  else msg.print_message(MSG_LEVEL_2,"[Schroedinger] Unable to open file K mesh file .\n");
}

// Roza Kotlyar 3/2/2011: request the shift for eigensolver
void Schroedinger::get_ifinitialized_shift_foresolver(double& shift)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_ifinitialized_shift_foresolver";
  NemoUtils::tic(tic_toc_prefix);


  if (ifinitialized_shift_foresolver)
  {

    shift=shift_foresolver;
    ifinitialized_shift_foresolver=false;
  }

  NemoUtils::toc(tic_toc_prefix);

}

//Roza Kotlyar 3/2/2011:  set the shift for eigensolver
void Schroedinger::set_shift_foresolver(double shift)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::set_shift_foresolver";
  NemoUtils::tic(tic_toc_prefix);


  shift_foresolver=shift;

  NemoUtils::toc(tic_toc_prefix);

}


void Schroedinger::do_solve()
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::do_solve ";
  NemoUtils::tic(tic_toc_prefix);
#ifdef USE_PAPI
  MPI_Comm krpcomm = MPI_COMM_WORLD;
  krp_rpt_init_(&krp_rank, &krpcomm, &krp_hw_counters, &krp_rcy, &krp_rus, &krp_ucy, &krp_uus);
#endif

  double tstart = NemoUtils::get_time();

  num_requested_eigenvalues  = options.get_option("number_of_eigenvalues", get_const_dof_map_base()->get_global_dof_number());
  number_of_eigenvalues_to_extract = options.get_option("number_of_eigenvalues_to_use", num_requested_eigenvalues);
  /*
  //M.P. commented this block out, moved to a different place
  this->set_eigensolver_options();
  // do NOT execute this in do_init() because shift values might change with potential and do_solve might be called several times!

  solver->set_matrix(hamiltonian);

  if (overlap != NULL)
    solver->set_B_matrix(overlap);
   */
  // =================================================================
  // depending on the job list, solve the stuff!
  // =================================================================

  if (   hole_density || electron_density
      || derive_electron_density_over_potential || derive_hole_density_over_potential
      || density_of_states || spatial_density_of_states)
  {
    // --------------------------------------------------------------------
    // solve energy levels, calculate density and integrate over k-space
    // --------------------------------------------------------------------
    EnergyV.assign(kspace->get_num_points(), vector<double>(num_requested_eigenvalues,0.0));


    //  msg.print_message2(MSG_LEVEL_1,"[Schroedinger] number of requested eigenvalues: %d. eigensolver shift: %g",
    //    num_requested_eigenvalues, solver.get_shift_sigma().real());

    // clear density, because it is to be integrated over k-points
    this->clear_density();

    /*double Brillouin_zone_volume = 0.0;
      for (unsigned int i=0;i<integrator.get_weights().size();i++)
      Brillouin_zone_volume += integrator.get_weights()[i];*/

    //msg.print_message2(MSG_INFO,"[Schroedinger] Brillouin zone volume: %g", Brillouin_zone_volume);
    // <ss> according to Michael 2pi factors cancel out

    bool finite_k_space = (get_simulation_domain()->get_reciprocal_vectors().size()>0);

    MPI_Barrier(k_space_communicator);
    MPI_Barrier(get_simulation_domain()->get_communicator());

    if(scattering_probability)
    {
      //get the external state solver
      NEMO_ASSERT(options.check_option("State_solver"),"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate define \"State_solver\"\n");
      std::string state_solver_name = options.get_option("State_solver",std::string(""));
      Simulation* state_solver = this->find_simulation(state_solver_name);
      NEMO_ASSERT(state_solver!=NULL,"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate have not found simulation \""+state_solver_name
          +"\"\n");
      //check:
      //external state solver is defined on the same domain as this solver
      const Domain* external_domain=state_solver->get_const_simulation_domain();
      NEMO_ASSERT(external_domain->get_name()==get_const_simulation_domain()->get_name(),"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate"
          +" mismatch with domain of simulation \""+state_solver_name+"\"\n");
      state_solver->get_data(std::string("subset_of_momenta"),subset_of_momenta);
      state_solver->get_data(std::string("subset_of_eigenvalues"),subset_of_eigenvalues);
      state_solver->get_data(std::string("subset_of_eigenvectors"),subset_of_eigenvectors);
    }

    // loop over the k-space points of respective MPI-process
    vector<int> my_k_points = kspace->get_my_points();
    for (unsigned int ii=0; ii<my_k_points.size(); ii++)
    {
      const vector<double>& k_point = kspace->get_point(my_k_points[ii]).get_coords();

      msg.print_message2(MSG_INFO,"[Schroedinger] loop at: %2.0f%%, k=(%0.6g,%0.6g,%0.6g)",
          1.0*ii/(1.0*my_k_points.size())*100,
          k_point[0], k_point[1], k_point[2]);

      /*
     msg.print_message2(MSG_INFO,"[Schroedinger] loop at: %2.0f%%, k=(%0.6g,%0.6g,%0.6g), Weight =%0.6g ",
                               1.0*ii/(1.0*my_k_points.size())*100,
                               k_point[0], k_point[1], k_point[2], k_weight_file[ii] );
       */

      // convergence scheme option based on monitoring eigenenergies at first_k_point
      // Roza Kotlyar: 3/2/2011
      first_k_point=0; // not a point to monitor by default
      dnk=dnk+1.0;     // counting k-points
      if (do_esolvershift_fromsolution)
      {
        double kmagtemp=0.0;
        for (unsigned int kv=0; kv<k_point.size(); kv++)
        {
          kmagtemp+=k_point[kv]*k_point[kv];
        }
        kmagtemp=sqrt(kmagtemp);
        if(fabs(kmagtemp-kmagref_foresolvershift)<kmagreftol_foresolvershift)
        {
          first_k_point=1;
        }
      }

      // SOLVE!
      solve_tb_single_kpoint(k_point, EnergyV[my_k_points[ii]]);

      // add to density and/or derivative_density_over_potential using an integration weight
      msg.print_message(MSG_LEVEL_2,"   adding k-point to density...");
      double weight = 1.0;
      if (finite_k_space && !analytical_k)
      {
        weight  = integrator.get_weights()[my_k_points[ii]];
      }
      if(options.check_option("K_points_file"))
        weight  =  k_weight_file[ii];
      else
        weight *= k_degeneracy;

      if (electron_density)
        determine_electron_density(weight);

      if (hole_density)
        determine_hole_density(weight);

      if (derive_electron_density_over_potential)
        determine_electron_density_Jacobian(weight);

      if (derive_hole_density_over_potential)
        determine_hole_density_Jacobian(weight);

      if (spatial_density_of_states)   // calculate spatially resolved DOS
      {
        for (unsigned int i = 0; i < DOS_energy_grid.size(); i++)
        {
          calculate_spatially_resolved_DOS(DOS_energy_grid[i], DOS_gamma, &(spatially_resolved_DOS[i]), weight*DOS_spin_factor);
        }
      }
      if (density_of_states)   // calculate for every process separately
      {
        if (project_density_of_states) //D.V calculate the project density of states
        {
          //calculate_DOS(DOS_energy_grid, DOS_gamma, &DOS, DOS_broadening_model, weight*DOS_spin_factor);
          //calculate_PDOS(DOS_energy_grid, DOS_gamma, PDOS, DOS_broadening_model, weight * DOS_spin_factor);
          calculate_PDOS(DOS_energy_grid,DOS_gamma, PDOS, DOS_broadening_model, weight * DOS_spin_factor);
        }
        else
          calculate_DOS(DOS_energy_grid, DOS_gamma, &DOS, DOS_broadening_model, weight*DOS_spin_factor);
      }
      if(scattering_probability)
      {
        //MPI_Barrier(k_space_communicator);
        calculate_scattering_rate(weight,k_point);
        //std::cerr<<"scattering is done\n";
      }
    } // end of loop over local k-points
    // ----------------------------------------------------------------------
    // MPI: collect the results and sum up density from different processes
    // ----------------------------------------------------------------------
    MPI_Barrier(k_space_communicator);
    msg.print_message(MSG_LEVEL_1,"[Schroedinger(\""+get_name()+"\")] collecting...");
    if(scattering_probability)
    {
      std::vector<std::vector<double> > combined_angles(all_scattering_angles.size());
      std::vector<std::map<double,double> > combined_angles_contributions(all_scattering_angles.size());
      std::vector<std::map<double,double> > combined_momenta_contributions(all_scattering_angles.size());
      std::vector<std::map<std::vector<double>, double> > combined_full_momenta_contributions(all_scattering_angles.size());
      for(unsigned int i=0; i<all_scattering_angles.size(); i++)
      {
        ////------------------collect scattering angles--------------
        ////Gather the total number of angles from all MPI processes
        int number_of_processes;
        MPI_Comm_size(k_space_communicator,&number_of_processes);
        //std::vector<int> angles_found_per_CPU(number_of_processes,0);
        int send_size=all_scattering_angles.find(i)->second.size();


        //------------------collect scattering delta_k--------------
        std::vector<int> delta_k_found_per_CPU(number_of_processes,0);
        send_size=all_momenta_scat_contributions[i].size();
        MPI_Allgather(&send_size, 1, MPI_INT, &(delta_k_found_per_CPU[0]),1, MPI_INT, k_space_communicator);
        std::vector<int> delta_k_displs(number_of_processes, 0);
        for(int ii = 1; ii < number_of_processes; ii++)
          delta_k_displs[ii] = delta_k_displs[ii-1] + delta_k_found_per_CPU[ii-1];

        std::vector<double> out_local_found_momenta(send_size);
        std::vector<double> out_local_found_momenta_contributions(send_size);
        std::map<double,double>::const_iterator delta_k_cit=all_momenta_scat_contributions[i].begin();
        for(unsigned int j=0; j<out_local_found_momenta.size(); j++)
        {
          out_local_found_momenta[j]=delta_k_cit->first;
          out_local_found_momenta_contributions[j]=delta_k_cit->second;
          delta_k_cit++;
        }
        unsigned int total_number_of_found_delta_k=0;
        for(unsigned int j=0; j<delta_k_found_per_CPU.size(); j++)
          total_number_of_found_delta_k+=delta_k_found_per_CPU[j];
        std::vector<double> in_all_found_momenta(total_number_of_found_delta_k,0.0);
        std::vector<double> in_all_found_momenta_contributions(total_number_of_found_delta_k,0.0);

        MPI_Allgatherv(&(out_local_found_momenta[0]),send_size,MPI_DOUBLE,&(in_all_found_momenta[0]),
            &(delta_k_found_per_CPU[0]),&(delta_k_displs[0]),MPI_DOUBLE,k_space_communicator);

        MPI_Allgatherv(&(out_local_found_momenta_contributions[0]),send_size,MPI_DOUBLE,&(in_all_found_momenta_contributions[0]),
            &(delta_k_found_per_CPU[0]),&(delta_k_displs[0]),MPI_DOUBLE,k_space_communicator);
        //store all contributions and abs_delta_k in the map
        for(unsigned int j=0; j<total_number_of_found_delta_k; j++)
        {
          double temp_key=in_all_found_momenta[j];
          combined_momenta_contributions[i][temp_key]=in_all_found_momenta_contributions[j];
        }


        MPI_Barrier(k_space_communicator);
        //------------------collect scattering vs. full_final_k--------------
        std::vector<int> k_found_per_CPU(number_of_processes,0);
        send_size=all_full_resolved_scat_contributions[i].size();
        MPI_Allgather(&send_size, 1, MPI_INT, &(k_found_per_CPU[0]),1, MPI_INT, k_space_communicator);
        std::vector<int> k_displs(number_of_processes, 0);
        for(int ii = 1; ii < number_of_processes; ii++)
          k_displs[ii] = k_displs[ii-1] + k_found_per_CPU[ii-1];
        std::vector<std::vector<double> > out_local_full_momenta(3,std::vector<double> (send_size,0.0));
        out_local_found_momenta_contributions.clear();
        out_local_found_momenta_contributions.resize(send_size);
        std::map<std::vector<double>,double>::const_iterator k_cit=all_full_resolved_scat_contributions[i].begin();
        for(int j=0; j<send_size; j++)
        {
          for(int jj=0; jj<3; jj++)
          {
            out_local_full_momenta[jj][j]=k_cit->first[jj];
            //std::cerr<<"("<<out_local_full_momenta[0][j]<<", "<<out_local_full_momenta[1][j]<<", "<<out_local_full_momenta[2][j]<<")"<<std::endl;
          }
          out_local_found_momenta_contributions[j]=k_cit->second;
          k_cit++;
        }
        unsigned int total_number_of_found_k=0;
        for(unsigned int j=0; j<k_found_per_CPU.size(); j++)
          total_number_of_found_k+=k_found_per_CPU[j];
        std::vector<std::vector<double> > in_all_found_full_momenta(3,std::vector<double> (total_number_of_found_k,0.0));
        for (unsigned int jj=0; jj<total_number_of_found_k; jj++)
        {
          //for debugging
          in_all_found_full_momenta[0][jj]=jj;
          in_all_found_full_momenta[1][jj]=jj+1;
          in_all_found_full_momenta[2][jj]=jj+2;
        }
        std::vector<double> in_all_found_full_momenta_contributions(total_number_of_found_k,0.0);
        //std::vector<double> temp_vector(total_number_of_found_k,0.0);
        for(unsigned int jj=0; jj<3; jj++)
        {
          MPI_Allgatherv(&(out_local_full_momenta[jj][0]),send_size,MPI_DOUBLE,&(in_all_found_full_momenta[jj][0]),
              &(k_found_per_CPU[0]),&(k_displs[0]),MPI_DOUBLE,k_space_communicator);
          /*MPI_Allgatherv(&(out_local_full_momenta[jj][0]),send_size,MPI_DOUBLE,&(temp_vector[0]),
            &(k_found_per_CPU[0]),&(k_displs[0]),MPI_DOUBLE,k_space_communicator);
          in_all_found_full_momenta[jj]=temp_vector;*/
        }
        MPI_Allgatherv(&(out_local_found_momenta_contributions[0]),send_size,MPI_DOUBLE,&(in_all_found_full_momenta_contributions[0]),
            &(k_found_per_CPU[0]),&(k_displs[0]),MPI_DOUBLE,k_space_communicator);
        //store all contributions and final_k in the map
        for(unsigned int j=0; j<total_number_of_found_k; j++)
        {
          std::vector<double> temp_key(3,0.0);
          for(unsigned int jj=0; jj<3; jj++)
          {

            temp_key[jj]=in_all_found_full_momenta[jj][j];
          }
          std::map<std::vector<double>,double>::iterator temp_it=combined_full_momenta_contributions[i].find(temp_key);
          /*if(temp_it!=combined_full_momenta_contributions[i].end())
            std::cerr << "found momentum twice\n";*/
          combined_full_momenta_contributions[i][temp_key]+=in_all_found_full_momenta_contributions[j];

        }
      }
      int temp_rank;
      MPI_Comm_rank(k_space_communicator, &temp_rank);
      if(temp_rank==0)
      {
        std::string filename = get_name() + "_scat_angles.dat";

        for(unsigned i=0; i<combined_momenta_contributions.size(); i++)
        {
          std::stringstream temp_stream;
          temp_stream<<i;
          filename = get_name() + "_momentum_resolved_scat_" + temp_stream.str()+ ".dat";
          ofstream out_file3(filename.c_str());
          std::map<double,double>::const_iterator c_momentum_it=combined_momenta_contributions[i].begin();
          for(; c_momentum_it!=combined_momenta_contributions[i].end(); c_momentum_it++)
            out_file3<<c_momentum_it->first<<"\t"<<c_momentum_it->second<<"\n";
          out_file3.close();
        }
        //write combined_full_momenta_contributions to file
        for(unsigned i=0; i<combined_full_momenta_contributions.size(); i++)
        {
          std::stringstream temp_stream;
          temp_stream<<i;
          filename = get_name() + "_full_resolved_scat_" + temp_stream.str()+ ".dat";
          ofstream out_file4(filename.c_str());
          std::map<std::vector<double>,double>::const_iterator c_momentum_it=combined_full_momenta_contributions[i].begin();
          for(; c_momentum_it!=combined_full_momenta_contributions[i].end(); c_momentum_it++)
            out_file4<<c_momentum_it->first[0]<<" "<<c_momentum_it->first[1]<<" "<<c_momentum_it->first[2]<<" "<<c_momentum_it->second<<"\n";
          out_file4.close();
        }
        //write combined_full_momenta_contributions as momentum change to file
        for(unsigned i=0; i<combined_full_momenta_contributions.size(); i++)
        {
          std::stringstream temp_stream;
          temp_stream<<i;
          filename = get_name() + "_delta_resolved_scat_" + temp_stream.str()+ ".dat";
          ofstream out_file4(filename.c_str());
          std::map<std::vector<double>,double>::const_iterator c_momentum_it=combined_full_momenta_contributions[i].begin();
          for(; c_momentum_it!=combined_full_momenta_contributions[i].end(); c_momentum_it++)
            out_file4<<c_momentum_it->first[0]-subset_of_momenta[i][0]<<" "<<c_momentum_it->first[1]-subset_of_momenta[i][1]<<" "<<c_momentum_it->first[2]
                                                                                  -subset_of_momenta[i][2]<<" "<<c_momentum_it->second<<"\n";
          out_file4.close();
        }
      }
      MPI_Barrier(k_space_communicator);
      //allreduce_map(scattering_rates);
      std::vector<double> resulting_scattering_rates(subset_of_eigenvectors.size(),0.0);
      //std::vector<double> resulting_momenta(subset_of_eigenvectors.size());
      std::map<unsigned int, double>::iterator temp_c_it=scattering_rates.begin();
      unsigned int counter=0;
      for(; temp_c_it!=scattering_rates.end(); ++temp_c_it)
      {
        //resulting_momenta[counter]=temp_c_it->first;
        resulting_scattering_rates[counter]+=temp_c_it->second;
        counter++;
      }
      MPI_Barrier(k_space_communicator);
      MPI_Allreduce(MPI_IN_PLACE,&(resulting_scattering_rates[0]),resulting_scattering_rates.size(),MPI_DOUBLE, MPI_SUM, k_space_communicator);
      //update the map
      for(unsigned int i=0; i<resulting_scattering_rates.size(); i++)
      {
        //temp_c_it=scattering_rates.find(resulting_momenta[i]);
        temp_c_it=scattering_rates.find(i);
        NEMO_ASSERT(temp_c_it!=scattering_rates.end(),"[Schroedinger(\""+get_name()+"\")] problem in finding momental\n");
        temp_c_it->second=resulting_scattering_rates[i];
      }
    }

    if (electron_density)
    {
      msg.print_message(MSG_LEVEL_2,"    electron density");
      // <ss> calling MPI_Allreduce individually for each and every vertex is very inefficient. hence we do it for the entire map
      this->allreduce_map(electron_charge);

      if (do_electron_integrated_den)   //Roza Kotlyar: 3/2/2011
      {
        map<unsigned int, double>  tmpvec1;
        tmpvec1[0]= electron_integrated_den;
        this->allreduce_map(tmpvec1);
        electron_integrated_den=tmpvec1[0];
        tmpvec1.clear();
      }
    }
    if (hole_density)
    {
      msg.print_message(MSG_LEVEL_2,"    hole density");
      this->allreduce_map(hole_charge);
      if (do_hole_integrated_den)  //Roza Kotlyar: 3/2/2011
      {
        map<unsigned int, double>  tmpvec1;
        tmpvec1[0]= hole_integrated_den;
        this->allreduce_map(tmpvec1);
        hole_integrated_den=tmpvec1[0];
        tmpvec1.clear();
      }
    }

    if (true)   //counting of k. Roza Kotlyar: 3/2/2011
    {
      map<unsigned int, double>  tmpvec1;
      tmpvec1[0]= dnk;
      this->allreduce_map(tmpvec1);
      dnk=tmpvec1[0];
      tmpvec1.clear();
    }
    if (true)   // Roza Kotlyar: 3/2/2011
    {
      map<unsigned int, double>  tmpvec1;
      tmpvec1[0]= nsetopt;
      this->allreduce_map(tmpvec1);
      nsetopt=tmpvec1[0];
      tmpvec1.clear();
    }
    if (do_esolvershift_fromsolution)   // Roza Kotlyar: 3/2/2011
    {
      map<unsigned int, double>  tmpvec1;
      tmpvec1[0]= eminecsubiter;
      tmpvec1[1]= emaxecsubiter;
      tmpvec1[2]= eminevsubiter;
      tmpvec1[3]= emaxevsubiter;

      this->allreduce_map(tmpvec1);
      eminecsub=tmpvec1[0];
      emaxecsub=tmpvec1[1];
      eminevsub=tmpvec1[2];
      emaxevsub=tmpvec1[3];
      tmpvec1.clear();

      map<unsigned int, double>  tmpvec2;
      done_firstiter_esolvershift_fromsolution = 1.0;
      tmpvec2[0]= done_firstiter_esolvershift_fromsolution;

      this->allreduce_map(tmpvec2);
      done_firstiter_esolvershift_fromsolution=tmpvec2[0];
      tmpvec2.clear();
    }
    if (derive_electron_density_over_potential)
    {
      msg.print_message(MSG_LEVEL_2,"    derive_electron_density_over_potential");
      this->allreduce_map(derivative_electron_density_over_potential);
    }
    if (derive_hole_density_over_potential)
    {
      msg.print_message(MSG_LEVEL_2,"    derive_hole_density_over_potential");
      this->allreduce_map(derivative_hole_density_over_potential);
    }
    if (spatial_density_of_states)
    {
      msg.print_message(MSG_LEVEL_2,"    spatial_density_of_states");
      int num_energy_points = DOS_energy_grid.size();
      for (int i = 0; i < num_energy_points; i++)
      {
        this->allreduce_map(spatially_resolved_DOS[i]);
      }

      // sum up the whole spatial stuff to get the device DOS
      this->DOS.assign(num_energy_points, 0.0);
      for (int i = 0; i < num_energy_points; i++)
      {
        const map<unsigned int, double>& density_e = spatially_resolved_DOS[i];
        for (map<unsigned int, double>::const_iterator it = density_e.begin(); it != density_e.end(); ++it)
        {
          DOS[i] += it->second;
        }
      }
    }
    if (density_of_states && !spatial_density_of_states)
    {
      // sum up DOS of all processes
      /* M.P. may be the next line has to be replaced with allreduce_map */
      MPI_Allreduce(MPI_IN_PLACE, &(DOS[0]), DOS.size(), MPI_DOUBLE, MPI_SUM, k_space_communicator);
    }
    if (project_density_of_states)
    {
      // sum up DOS of all processes
      /* M.P. may be the next line has to be replaced with allreduce_map */
      for(unsigned int i=0; i<4 ;i++)
        MPI_Allreduce(MPI_IN_PLACE, &(PDOS[i][0]), (PDOS[i]).size(), MPI_DOUBLE, MPI_SUM, k_space_communicator);
    }


    msg.print_message(MSG_LEVEL_2,"    eigenvalues");
    for (unsigned int ii=0; ii<EnergyV.size(); ii++)
    {
      unsigned int num_values = EnergyV[ii].size();
      /* M.P. may be the next line has to be replaced with allreduce_map */
      if (num_values <  number_of_eigenvalues_to_extract)
      {
        ostringstream o;
        o << "[Schroedinger(\""+get_name()+"\")::do_solve()] Numerical convergence issue: inconsistent number of eigenstates\n";
        o << "found " << num_values << "  requested " << num_requested_eigenvalues << "\n";
        o << "Try decreasing number_of_eigenvalues_to_use or increasing ncv." << "\n";
        throw runtime_error(o.str());

      }
      int rank(0);
      MPI_Comm_rank(k_space_communicator, &rank);
      if (rank == 0)
        MPI_Reduce(MPI_IN_PLACE, &EnergyV[ii][0], num_values, MPI_DOUBLE, MPI_SUM, 0,k_space_communicator);
      else
        MPI_Reduce(&EnergyV[ii][0], &EnergyV[ii][0], num_values, MPI_DOUBLE, MPI_SUM, 0,k_space_communicator);
      //      MPI_Allreduce(MPI_IN_PLACE, &EnergyV[ii][0], num_values, MPI_DOUBLE, MPI_SUM, k_space_communicator);
      // MPI_Reduce(MPI_IN_PLACE, &EnergyV[ii][0], num_values, MPI_DOUBLE, MPI_SUM, 0,k_space_communicator);
    }

    prepare_total_charge();

    if (density_of_states && energy_gradient)
    {
      msg << "[Schroedinger(\""+get_name()+"\")] energy_gradient option never was working and currently segfaults.\n";

    }
    msg.print_message(MSG_LEVEL_2,"    collecting is done...");
  }
  else if (calculate_band_structure)
  {

    //--------------------------------------------------
    // calculate electron band structure
    //--------------------------------------------------
    //M.P. data is calculated in a distribured way and then gathered to the CPU #0 in the kspace communicator
    int k_space_rank;
    MPI_Comm_rank(k_space_communicator, &k_space_rank);
    int k_space_comm_size;
    MPI_Comm_size(k_space_communicator, &k_space_comm_size);


    int n_k_points = kspace->get_num_points();

    //if (get_simulation_domain()->get_geometry_replica() == 0)
    {
      // determine path variable of band structure calculation
      vector<double> old_k=kspace->get_point(0).get_coords();

      k_distanceV.resize(n_k_points);


      for (unsigned int i=0; i<kspace->get_num_points(); i++)
      {
        const vector<double>& k_point = kspace->get_point(i).get_coords();
        NEMO_ASSERT(old_k.size()==3 && k_point.size()==3,
            "[Schroedinger(\""+get_name()
            +"\"):do_solve] old_k.size()==3 && k_point.size()==3 failed. Check that your specified k-points have the correct dimensionality.");
        old_k[0]=(k_point[0]-old_k[0]);
        old_k[1]=(k_point[1]-old_k[1]);
        old_k[2]=(k_point[2]-old_k[2]);
        if (i>0)
          k_distanceV[i] = k_distanceV[i-1] + NemoMath::vector_norm_3d(&old_k[0]);
        old_k = k_point;
      }
    }
    MPI_Barrier(k_space_communicator);
    vector<int> my_k_points = kspace->get_my_points(); //local k-points, need to make a copy for MPI communications;

    int n_my_k_points = my_k_points.size();

    vector<double> energy_local(n_my_k_points * number_of_eigenvalues_to_extract);

    vector<complex<double> > spin_rho_local;

    if (spin_projection)
      spin_rho_local.resize(n_my_k_points * number_of_eigenvalues_to_extract*3);

    full_energy_to_vector_map.clear();
    full_energy_to_vector_map.resize(n_my_k_points);


    for (int i = 0; i < n_my_k_points; i++)
    {
      const vector<double>& k_point = kspace->get_point(my_k_points[i]).get_coords();

      msg.print_message2(MSG_INFO,"[Schroedinger] band dispersion loop at: %2.0f%%, k=(%0.6g,%0.6g,%0.6g)",
          1.0*i/(1.0*kspace->get_num_points())*100,
          k_point[0], k_point[1], k_point[2]);

      vector<double> single_point_energies(number_of_eigenvalues_to_extract);


      // Solve eigenvalue problem
      solve_tb_single_kpoint(k_point,single_point_energies);
      //If number_of_eigenvalues_to_extract changed after solve_tb_single_kpoint call (B.N.)
      if(n_my_k_points*number_of_eigenvalues_to_extract != energy_local.size())
      {
        energy_local.resize(n_my_k_points * number_of_eigenvalues_to_extract);
        if (spin_projection)
          spin_rho_local.resize(n_my_k_points * number_of_eigenvalues_to_extract*3);
      }

      full_energy_to_vector_map[i]=energy_to_vector_map;

      // Protection (B.N.)
      unsigned int number_of_converged = single_point_energies.size();
      string num_requested;
      ostringstream convert;
      convert << num_requested_eigenvalues;
      num_requested = convert.str();
      if (options.check_option("number_of_eigenvalues_to_use"))
        NEMO_ASSERT(number_of_eigenvalues_to_extract <= number_of_converged,
            "[Schroedinger(\""+get_name()+"\")::do_solve()] number_of_eigenvalues_to_use is larger than number of converged eigenvalues." +
            " Try increasing number_of_eigenvalues in steps of 30. Currently number_of_eigenvalues = " + num_requested + ".\n");

      for (unsigned int j = 0; j < number_of_eigenvalues_to_extract; j++)
      {
        if (j < number_of_converged)
          energy_local[i * number_of_eigenvalues_to_extract + j] = single_point_energies[j];
        else
          energy_local[i * number_of_eigenvalues_to_extract + j] = single_point_energies[number_of_converged-1];

        if (spin_projection)
        {
          vector<complex<double> > spin_rho_matrix(3);

          //calculate spin-resolved projection matrix
          calculate_spin_projection(j, &spin_rho_matrix);


          for (short p = 0; p < 3; p++)
            spin_rho_local[i * number_of_eigenvalues_to_extract*3 + 3*j + p] =  spin_rho_matrix[p];
        }
      }
    }


    MPI_Barrier(k_space_communicator);


    //  M.P. The dispersion needs to be known on all CPUs
    //  Why? Not for output. (Probably, it must be used for some analysis)
    //  please, answer, if you can.


    // ------------------------------------------------------------------------------------
    // set up communicators that contain only one process out of every geometry replica.
    // holder.one_partition_total_communicator already contains one process out of every geometry replica.
    // every process within a replica computes the same k-points and has the same eigenvalues stored
    // therefore we intersect the "total" Nemo communicator with the k-space communicator.
    // --------------------------------------------------------------------------

    MPI_Comm single_part_comm;
    MPI_Group all_partition_single_variable;
    MPI_Comm_group(k_space_communicator, &all_partition_single_variable);

    MPI_Group single_partition_all_variable;
    MPI_Comm_group(holder.one_partition_total_communicator, &single_partition_all_variable);

    MPI_Group single_partition_single_variable;
    MPI_Group_intersection (all_partition_single_variable,
        single_partition_all_variable,
        &single_partition_single_variable );


    MPI_Comm_create(k_space_communicator,
        single_partition_single_variable,
        &single_part_comm );


    int single_part_comm_rank;
    MPI_Comm_rank(single_part_comm, &single_part_comm_rank);
    int single_part_comm_size;
    MPI_Comm_size(single_part_comm, &single_part_comm_size);


    // how many k-points do I compute on every CPU?
        vector<int>  n_distributed_k_points(single_part_comm_size);
        MPI_Allgather(&n_my_k_points, 1, MPI_INT,  &(n_distributed_k_points[0]), 1, MPI_INT, single_part_comm);

        vector<int> displs(single_part_comm_size, 0);
        for (int i = 1; i < single_part_comm_size; i++)
          displs[i] = displs[i-1] + n_distributed_k_points[i-1];

        // communicate the computed k-point indices of all CPUs
        vector<int> k_point_index_gathered(n_k_points, 0);
        MPI_Allgatherv(&(my_k_points[0]), n_my_k_points, MPI_INT,
            &(k_point_index_gathered[0]), &(n_distributed_k_points[0]), &(displs[0]), MPI_INT, single_part_comm);

        for (int i = 0; i < single_part_comm_size; i++)
        {
          n_distributed_k_points[i] *= number_of_eigenvalues_to_extract;
          displs[i] *=  number_of_eigenvalues_to_extract;
        }


        MPI_Barrier(k_space_communicator);

        // communicate the eigenenergies of all CPUs
        vector<double> eigen_energy_gathered(n_k_points * number_of_eigenvalues_to_extract, 0.0);
        vector<complex<double> > spin_rho_gathered(3*n_k_points * number_of_eigenvalues_to_extract, complex<double>(0,0));

        MPI_Allgatherv(&(energy_local[0]), n_my_k_points*number_of_eigenvalues_to_extract, MPI_DOUBLE,
            &(eigen_energy_gathered[0]), &(n_distributed_k_points[0]), &(displs[0]), MPI_DOUBLE,
            single_part_comm);



        // communicate the index ordering of energy_to_vector_map of all CPUs
        vector<unsigned int> energy_to_vector_map_gathered(n_k_points * number_of_eigenvalues_to_extract, 1);

        //transform full_energy_to_vector_map into 1D-vector
        vector<unsigned int> temp_full_energy_to_vector_map(number_of_eigenvalues_to_extract*n_my_k_points);
        for(int ii=0; ii<n_my_k_points; ii++)
          for(unsigned int iii=0; iii<number_of_eigenvalues_to_extract; iii++)
            temp_full_energy_to_vector_map[ii*number_of_eigenvalues_to_extract+iii]=full_energy_to_vector_map[ii][iii];

        MPI_Barrier(k_space_communicator);

        MPI_Allgatherv(&(temp_full_energy_to_vector_map[0]), n_my_k_points*number_of_eigenvalues_to_extract, MPI_UNSIGNED,
            &(energy_to_vector_map_gathered[0]), &(n_distributed_k_points[0]), &(displs[0]), MPI_UNSIGNED,
            single_part_comm);

        if (spin_projection)
        {
          //gather the spin density matrix
          //for each eigestate there are 3 complex numbers
          for (int i = 0; i < single_part_comm_size; i++)
          {
            n_distributed_k_points[i] *= 3;
            displs[i] *=  3;
          }

          MPI_Allgatherv(&(spin_rho_local[0]),
              n_my_k_points*number_of_eigenvalues_to_extract *3, MPI_DOUBLE_COMPLEX,
              &(spin_rho_gathered[0]), &(n_distributed_k_points[0]), &(displs[0]), MPI_DOUBLE_COMPLEX,
              single_part_comm);


        }

        // the single-process-per-replica communicator is not needed anymore
        MPI_Comm_free(&single_part_comm);
        MPI_Barrier(k_space_communicator);

        // set up EnergyV array (=result of this whole exercise)
        EnergyV.assign(n_k_points, vector<double>(number_of_eigenvalues_to_extract, 0.0));
        full_energy_to_vector_map.assign(n_k_points, vector<unsigned int>(number_of_eigenvalues_to_extract, 1));
        if (spin_projection)
        {
          vector<vector<complex<double> > > temp(number_of_eigenvalues_to_extract, vector<complex<double> >(3)) ;
          spin_projection_matrix.resize(kspace->get_num_points(),temp);
        }


        for (int i = 0; i < n_k_points; i++)
        {
          for (unsigned int j = 0; j < number_of_eigenvalues_to_extract; j++)
          {
            NEMO_ASSERT((int)k_point_index_gathered.size()>i, "[Schroedinger(\""+get_name()+"\")] k_point_index_gathered.size()>i failed.");
            NEMO_ASSERT((int)EnergyV.size()>k_point_index_gathered[i], "[Schroedinger(\""+get_name()+"\")] EnergyV.size()>k_point_index_gathered[i] failed.");
            NEMO_ASSERT(EnergyV[k_point_index_gathered[i]].size()>j, "[Schroedinger(\""+get_name()+"\")] EnergyV[k_point_index_gathered[i]].size()>j failed.");
            NEMO_ASSERT(eigen_energy_gathered.size()>i * number_of_eigenvalues_to_extract + j,
                "[Schroedinger(\""+get_name()+"\")] eigen_energy_gathered.size()>i * number_of_eigenvalues_to_extract + j failed.");

            EnergyV[k_point_index_gathered[i]][j] = eigen_energy_gathered[i * number_of_eigenvalues_to_extract + j];
            //transform energy_to_vector_map_gathered to vector of vector
            full_energy_to_vector_map[k_point_index_gathered[i]][j] = energy_to_vector_map_gathered[i * number_of_eigenvalues_to_extract + j];

            if (spin_projection)
              for (short p = 0; p < 3; p++)
                spin_projection_matrix[k_point_index_gathered[i]][j][p]
                                           = spin_rho_gathered[i * number_of_eigenvalues_to_extract * 3 + j*3 +p];



          }
        }

        MPI_Barrier(k_space_communicator);

        bool calculate_elastic_overlap = options.get_option("calculate_elastic_overlap", false);

        if (calculate_elastic_overlap)
          if (get_simulation_domain()->get_geometry_replica() == 0)
          {
            double E = options.get_option("energy_for_elastic_overlap", 0.0);
            double dE = options.get_option("dE_for_elastic_overlap", 0.0);
            std::vector< std::vector<double> > k_poins;
            std::vector< std::vector < std::complex <double> > > matrix_elements;
            calculate_overlap_for_elastic_scattering(E, dE);

          }

        Nemo::system_flops(n_k_points*(16+number_of_eigenvalues_to_extract*3+single_part_comm_size+n_k_points*number_of_eigenvalues_to_extract*8));

  }  // end of calculate_band_structure
  else if (assemble_H)
  {
    // --------------------------------
    // assemble (only) Hamiltonian
    // --------------------------------
    vector<int> my_k_points = kspace->get_my_points();

    k_vector  = kspace->get_point(my_k_points[0]).get_coords();

    if(!combine_H)
      this->assemble_hamiltonian(); // uses this->k_vector
    else
      combine_hamiltonian();
  }
  else if (output_H_all_k)
  {
    //--------------------------------------------------
    // calculate electron band structure
    //--------------------------------------------------
    //M.P. data is calculated in a distribured way and then gathered to the CPU #0 in the kspace communicator
    int k_space_rank;
    MPI_Comm_rank(k_space_communicator, &k_space_rank);
    int k_space_comm_size;
    MPI_Comm_size(k_space_communicator, &k_space_comm_size);

    int n_k_points = kspace->get_num_points();

    //if (get_simulation_domain()->get_geometry_replica() == 0)
    {
      // determine path variable of band structure calculation
      vector<double> old_k=kspace->get_point(0).get_coords();

      k_distanceV.resize(n_k_points);


      for (unsigned int i=0; i<kspace->get_num_points(); i++)
      {
        const vector<double>& k_point = kspace->get_point(i).get_coords();
        NEMO_ASSERT(old_k.size()==3 && k_point.size()==3,
            "[Schroedinger(\""+get_name()
            +"\"):do_solve] old_k.size()==3 && k_point.size()==3 failed. Check that your specified k-points have the correct dimensionality.");
        old_k[0]=(k_point[0]-old_k[0]);
        old_k[1]=(k_point[1]-old_k[1]);
        old_k[2]=(k_point[2]-old_k[2]);
        if (i>0)
          k_distanceV[i] = k_distanceV[i-1] + NemoMath::vector_norm_3d(&old_k[0]);
        old_k = k_point;
      }
    }

    MPI_Barrier(k_space_communicator);

    vector<int> my_k_points = kspace->get_my_points(); //local k-points, need to make a copy for MPI communications;

    int n_my_k_points = my_k_points.size();

    for (int i = 0; i < n_my_k_points; i++)
    {
      const vector<double>& k_point = kspace->get_point(my_k_points[i]).get_coords();

      this->k_vector = k_point;
      msg.print_message2(MSG_INFO,"[Schroedinger] k=(%0.6g,%0.6g,%0.6g)",
          k_point[0], k_point[1], k_point[2]);

      if(!combine_H)
        this->assemble_hamiltonian();
      else
        combine_hamiltonian();
      std::stringstream temp_stream;
      temp_stream<<i;


      string temp = get_name()+get_output_suffix()+"_k_"+ temp_stream.str()+ "_Ham.m";

      hamiltonian->save_to_matlab_file(temp.c_str());
      //hamiltonian->save_to_matlab_file("hamiltonian.m");
    }
    MPI_Barrier(k_space_communicator);
  }

  if(calculate_Fermi_level)
  {//author: Bozidar N., Zhengping J.

    MsgLevel msg_level = msg.get_level();
    msg.set_level(MsgLevel(0));

    msg<<"[Schroedinger(\""+get_name()+"\")] Calculating Fermi level according to doping. "<<endl;
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    const AtomicStructure& atoms   = domain->get_atoms();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();

    //Aryan change for CS
    ConstActiveAtomIterator it_ref  = atoms.active_atoms_begin();
    int ref_domain_atom_4matprop = options.get_option("ref_domain_atom_4matprop",0); //default 1st atom of this domain as before
    it_ref=it_ref+ref_domain_atom_4matprop;
    const AtomStructNode& nd        = it_ref.node();

    //const AtomStructNode& nd        = it.node();
    const Atom*           atom      = nd.atom;
    const Material*       material  = atom->get_material();
    HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(material));

    double cell_area = 0;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = nd.position[0];
    xmax = nd.position[0];
    ymin = nd.position[1];
    ymax = nd.position[1];
    zmin = nd.position[2];
    zmax = nd.position[2];

    for ( ; it != end; ++it)
    {
      const AtomStructNode& nd = it.node();
      if(nd.position[0]>xmax)
        xmax=nd.position[0];
      else if(nd.position[0]<xmin)
        xmin=nd.position[0];

      if(nd.position[1]>ymax)
        ymax=nd.position[1];
      else if(nd.position[1]<ymin)
        ymin=nd.position[1];

      if(nd.position[2]>zmax)
        zmax=nd.position[2];
      else if(nd.position[2]<zmin)
        zmin=nd.position[2];
    }
    std::list<double> kx_v, ky_v, kz_v;

    for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
    {
      const NemoMeshPoint& ki = kspace->get_point(ii);
      kx_v.push_back(ki[0]);
      ky_v.push_back(ki[1]);
      kz_v.push_back(ki[2]);
    }
    kx_v.sort();
    ky_v.sort();
    kz_v.sort();
    kx_v.unique(NemoMath::CompareDoubleEqual());
    ky_v.unique(NemoMath::CompareDoubleEqual());
    kz_v.unique(NemoMath::CompareDoubleEqual());

    double Ndop, rho_sign;
    string doping_type_str = material->get_material_options()->get_option("doping_type",string("i"));
    Ndop = material->get_material_options()->get_option("doping_density",0.0);

    double Temp = options.get_option("electron_temperature",NemoPhys::temperature);
    if(doping_type_str=="N" || doping_type_str=="n" || doping_type_str=="i")
      rho_sign = 1;
    else rho_sign = -1;

    int Nk = kx_v.size();
    int Nky = ky_v.size();
    int Nkz = kz_v.size();
    double kmin = kx_v.front();
    double kmax = kx_v.back();
    double kymin = ky_v.front();
    double kymax = ky_v.back();
    double kzmin = kz_v.front();
    double kzmax = kz_v.back();

    //to account for the case when x is not transport direction
    double dk;
    if (Nk == 1)
      dk = 1.e-9;
    else
      dk = (kmax-kmin) / (Nk-1);

    std::vector<bool> periodicity(3,false);
    get_simulation_domain()->get_domain_options()->get_option("periodic",periodicity);
    std::vector<std::vector<double> > bravais_vectors = get_simulation_domain()->get_translation_vectors();

    //Limitation 1: bravais vectors must be orthogonal.
    {
      if(!NemoMath::CompareDoubleEqual()(inner_product(bravais_vectors[0].begin(),bravais_vectors[0].end(),bravais_vectors[1].begin(),0),0.0))
        throw std::runtime_error("[Schroedinger(\""+get_name()+"\")] crystal directions must be orthogonal.");
      if(!NemoMath::CompareDoubleEqual()(inner_product(bravais_vectors[0].begin(),bravais_vectors[0].end(),bravais_vectors[2].begin(),0),0.0))
        throw std::runtime_error("[Schroedinger(\""+get_name()+"\")] crystal directions must be orthogonal.");
      if(!NemoMath::CompareDoubleEqual()(inner_product(bravais_vectors[1].begin(),bravais_vectors[1].end(),bravais_vectors[2].begin(),0),0.0))
        throw std::runtime_error("[Schroedinger(\""+get_name()+"\")] crystal directions must be orthogonal.");
    }

    //Limitation 2: bravais and space axes must overlap (in arbitrary order, no need for crystal_direction1 = x, crystal_direction2 = y, crystal_direction3 = z).
    if(Nk*Nky*Nkz != (int)(kspace->get_num_points()))
    {
      throw std::runtime_error("[Schroedinger(\""+get_name()+"\")] bravais and space axes must overlap (in arbitrary order, no need for " +
        "crystal_direction1 = x, crystal_direction2 = y, crystal_direction3 = z)");
    }

    cell_area = 1.0;
    if(Nk == 1)
      cell_area *= (xmax - xmin)*1.e-9;
    if(Nky == 1)
      cell_area *= (ymax - ymin)*1.e-9;
    if(Nkz == 1)
      cell_area *= (zmax - zmin)*1.e-9;

    //Multiply cell_area by (2*pi)^n which comes from periodic directions (necessary because Fermi.cpp does not know about n (dimensionality)).
    if(Nk > 1)
      cell_area *= 2.0*M_PI;
    if(Nky > 1)
      cell_area *= 2.0*M_PI;
    if(Nkz > 1)
      cell_area *= 2.0*M_PI;

    int spin_factor = tb_ham->get_spin_degeneracy();
    double ECmin = tb_ham->get_conduction_band_edge();
    double EVmax = tb_ham->get_valence_band_edge();
    msg<<"\t[Schroedinger(\""+get_name()+"\")] information on Fermi level calculation:"<<endl;
    msg<<"\tNdop="<<Ndop* rho_sign<<"cm-3; spin_factor="<<spin_factor<<"; Temp="<<Temp<<endl;
    msg<<"\tNkx="<<Nk<<"; Nky="<<Nky<<"; Nkz="<<Nkz<<endl;
    msg<<"\tEVmax="<<EVmax<<"; ECmin="<<ECmin<<"; cell_area="<<cell_area<<endl;
    msg<<"\tDone... begin fermi_energy()"<<endl;

    //Commented EVmax, ECmin argument to avoid warning - D.L.
    fermi_energy(Ndop*1e6,rho_sign,spin_factor,Temp,kmin*1e9,kmax*1e9,dk*1e9,Nk,
        kymin*1e9,kymax*1e9,Nky,kzmin*1e9,kzmax*1e9,Nkz, /*EVmax,*/ /*ECmin,*/
        cell_area, 1, 0);

    msg<<"Done with fermi level finding..."<<endl;
    msg.set_level(msg_level);
  }

  // Block matrix vector multiplication test
  if(block_matrix_vector_mult_time)
  {
    const int tb_param=20; // <-- this works only for sp3d5sstar_SO
    string pfx="[Schroedinger(\""+get_name()+"\")] block matrix vector multiplication : ";
    msg << pfx+"Block scaling test\n";
    msg << pfx+"Assembling hamiltonian (the matrix must be sp3d5sstar_SO)\n";

    if(!combine_H)
      assemble_hamiltonian();
    else
      combine_hamiltonian();
    int ne=hamiltonian->get_num_rows();
    msg << pfx+"Number of rows of original Hamiltonian is = " << ne << "\n";
    // 0. declare the matrix in PETSc block format
    msg << pfx+"Declaring the matrix in block format\n";
    PetscMatrixBlock<cplx>* block_matrix;
    block_matrix=new PetscMatrixBlock<cplx>(ne,ne,get_simulation_domain()->get_communicator(),tb_param);
    msg << pfx+"Setting the block matrix to zero\n";
    block_matrix->set_to_zero();
    // 1. get the values from the original Hamiltonian (PetscMatrixParallelComplex format)
    // and put them in the PETSc block matrix block_matrix
    msg << pfx+"Declaring the temporary submatrix\n";
    PetscMatrixParallelComplex* tmp_submatrix;
    tmp_submatrix=new PetscMatrixParallelComplex(tb_param,tb_param,get_simulation_domain()->get_communicator());
    msg << pfx+"Setting temporary rows and cols for hamiltonian extraction\n";
    std::vector<int> rows,cols;
    for(register int i=0; i<tb_param; i++)
    {
      rows.push_back(i);
      cols.push_back(i);
    }
    msg << pfx+"extracting the submatrix from the hamiltonian\n";
    hamiltonian->get_submatrix(rows,cols,MAT_INITIAL_MATRIX,tmp_submatrix);
    msg << pfx+"The values of the extracted submatrix are:\n";
    for(int i=0; i<tb_param; i++)
    {
      for(int j=0; j<tb_param; j++)
      {
        msg << "v[" << i << "][" << j << "]=" << tmp_submatrix->get(i,j) << "\n";
      }
    }
    //   block_matrix->set_block(1,1,tmp_submatrix);
    // end of block matrix test
  }

  if (matrix_vector_mult_time)
  {

    msg << "[Schroedinger(\""+get_name()+"\")] scaling test\n";
    if(!combine_H)
      assemble_hamiltonian();
    else
      combine_hamiltonian();

    int num_local_elements = hamiltonian->get_num_owned_rows ();
    int num_elements = hamiltonian->get_num_rows ();


    PetscVectorNemo<std::complex<double> > vec1 (num_local_elements, num_elements, get_simulation_domain()->get_communicator());
    PetscVectorNemo<std::complex<double> >  vec2 (num_local_elements, num_elements, get_simulation_domain()->get_communicator());

    vec1.set(std::complex<double> (1.0, 1.0));

    msg << "[Schroedinger(\""+get_name()+"\")] Doing 500 vector matrix multiplications\n";

    double t0 = NemoUtils::get_time();

    for (int i = 0; i < 500; i++)
      PetscMatrixParallelComplex::multMatVec(*hamiltonian, vec1, vec2);

    double t1 = NemoUtils::get_time();

    int comm_size;
    MPI_Comm_size(get_simulation_domain()->get_communicator(), &comm_size);

    if (get_simulation_domain()->get_geometry_rank() == 0)
    {
      ofstream result_file;

      result_file.open("MatrixVectorScaling.dat", std::ios::app);

      result_file << comm_size << "       " << t1 - t0 << "      "  << num_elements <<  "\n";
      result_file.close();
    }
  }


  if (options.get_option("compute_edges_masses",false))
  {
    const PseudomorphicDomain* pseudo = dynamic_cast<const PseudomorphicDomain*>(get_simulation_domain());
    NEMO_ASSERT(pseudo!=NULL, "[Schroedinger(\""+get_name()+"\")] could not cast into PseudomorphicDomain");
    const Crystal* cr = pseudo->get_crystal();
    const vector<BasisAtom>& basis = cr->get_primitive_basis();
    msg << "[Schroedinger(\""+get_name()+"\")] calculating k-points necessary for finding band edges and masses..." << std::endl;
    const Domain* domain = get_simulation_domain();
    int n = domain->get_reciprocal_vectors().size();
    if(n == 3)
    {
      if (basis.size()==1) // for effective mass or k.p method, the basis size is 1
      {
        const std::vector<std::vector<double> >& bravais_vectors = cr->get_primitive_bravais_vectors();
        // find lattice constant
        double bond[3] = { bravais_vectors[0][0], bravais_vectors[0][1], bravais_vectors[0][2] };
        double a = NemoMath::vector_norm_3d(bond);

        this->compute_edges_masses_zincblende(a);

      }
      else if (basis.size()==2)
      {
        // find lattice constant
        double bond[3] = { basis[1].coord[0] - basis[0].coord[0], basis[1].coord[1] - basis[0].coord[1], basis[1].coord[2] - basis[0].coord[2] };
        double a = 4.0/NemoMath::sqrt(3.0) * NemoMath::vector_norm_3d(bond); // distance between atoms = sqrt(3)/4*a for diamond / zincblende

        this->compute_edges_masses_zincblende(a);

      }
      else if (basis.size()==4)
      {
        const WurtziteCrystal* wz = dynamic_cast<const WurtziteCrystal*>(cr);
        NEMO_ASSERT(wz!=NULL, "[Schroedinger(\""+get_name()+"\")] could not cast into WurtziteCrystal.");
        double c = wz->get_0001_bond_length();
        double a = wz->get_second_bond_length();
        this->compute_edges_masses_wurtzite(a, c);
      }
      else
      {
        NEMO_EXCEPTION("[Schroedinger(\""+get_name()+"\")] expected 2- or 4-atom primitive basis (code is written for diamond or zincblende crystals).");
      }
    }
    else
    {
      this->compute_edge_masses_wire_utb();

    }
  }

  MPI_Barrier(k_space_communicator);
  double tstop = NemoUtils::get_time();

  // Roza Kotlyar: 3/2/2011
  if (do_hole_integrated_den || do_electron_integrated_den)
  {
    msg << "[Schroedinger(\""+get_name()+"\")] do_solve is finished in " << tstop-tstart << " seconds. Integrated densities: electron hole "
        << electron_integrated_den<<" "<<hole_integrated_den<<" nk= "<<dnk<<" nsetopt= "<<nsetopt;
    if (do_esolvershift_fromsolution)
    {
      msg << " eminevsub= "<<eminevsub<<" emaxevsub= "<<emaxevsub<<std::endl;//MessagingSystem::endl;
    }
    else
    {
      msg << std::endl;//MessagingSystem::endl;
    }
  }
  else
  {
    //DL - It seems like MessagingSystem::endl will always print out newlines
    msg << "[Schroedinger(\""+get_name()+"\")] do_solve finished in " << tstop-tstart << " seconds." << std::endl;//MessagingSystem::endl;
  }
  nsetopt=0.0;
  //To track task complexity
  Nemo::instance().set_hamiltonianSize((1.0*hamiltonian->get_num_rows())*hamiltonian->get_num_cols());
  Nemo::instance().set_kSpaceMeshPoints(kspace->get_num_points());

  NemoUtils::toc(tic_toc_prefix);
}



//Calculate effective masses for wires and utb
void Schroedinger::compute_edge_masses_wire_utb()
{

  vector<double> E_kzero, E_kzerop, E_kzerom;
  vector<double> EG_kzero, EG_kzerop, EG_kzerom;
  vector<double> dk_point(3),dkG_point(3),diff_mdir_from_gamma(3);
  const double del = options.get_option("compute_edges_masses_delta", 0.01);
  const int number_of_subbands = options.get_option("number_of_subbands", 4);
  int subband_number = 0 ,ii=0;
  int array_indx =0 , valley_count = 0 , minima_count = 0,no_pts =0;
  double mstar =0 ,dk,dkG,hbar2del2_m0,distance_from_Gamma,normdistance_of_effective_mass_dirs, eps = 1e-6;
  vector<double> coord(3, 0.0);
  vector<int> my_k_points = kspace->get_my_points();
  int n_my_k_points = my_k_points.size();
  const vector<double>& k_point = kspace->get_point(my_k_points[n_my_k_points-1]).get_coords();
  bool find_minima = false;
  if (options.get_option("compute_edges_masses_at_minima",true))
    find_minima = true;


  dkG_point[0] = del*k_point[0];
  dkG_point[1] = del*k_point[1];
  dkG_point[2] = del*k_point[2];
  dkG = NemoMath::vector_norm_3d(&(dkG_point[0]));
  if (dkG < 1e-4)
  {
    double old_dk = dkG;
    dkG = 1e-4;
    dkG_point[0] = (dkG/old_dk)*dkG_point[0];
    dkG_point[1] = (dkG/old_dk)*dkG_point[1];
    dkG_point[2] = (dkG/old_dk)*dkG_point[2];
  }


  coord[0] = dkG_point[0] ;
  coord[1] = dkG_point[1] ;
  coord[2] = dkG_point[2] ;
  solve_tb_single_kpoint(coord, EG_kzerop);
  coord[0] = -dkG_point[0] ;
  coord[1] = -dkG_point[1] ;
  coord[2] = -dkG_point[2] ;
  solve_tb_single_kpoint(coord, EG_kzerom);
  coord[0] = 0 ;
  coord[1] = 0 ;
  coord[2] = 0 ;
  solve_tb_single_kpoint(coord, EG_kzero);
  double me0_nemounits = NemoPhys::me0_nemo;
  eff_masses_info.clear();
  // Calculate effective masses for each subband
  for(subband_number = 0; subband_number < number_of_subbands ; subband_number++)
  {
    //Calculate effective mass at Gamma point always for each subband
    // The direction of this effective mass is the last k-point specified in the schrodinger solver
    hbar2del2_m0 = (dkG)*(dkG) * NemoPhys::hbar_nemo*NemoPhys::hbar_nemo / me0_nemounits;
    double mG_CB1  =     hbar2del2_m0 / (EG_kzerop[subband_number]-2.0*EG_kzero[subband_number]+EG_kzerom[subband_number]);
    meff_info* eff_massCB1 = new meff_info();
    eff_massCB1->no_valley = 1;
    eff_massCB1->band_number = subband_number + 1;
    eff_massCB1->eff_mass = mG_CB1;
    eff_massCB1->Ene = EG_kzero[subband_number];
    eff_massCB1->mdir[0] = dkG_point[0]/dkG;
    eff_massCB1->mdir[1] = dkG_point[1]/dkG;
    eff_massCB1->mdir[2] = dkG_point[2]/dkG;
    eff_massCB1->kpos[0] = coord[0]/NemoMath::pi;
    eff_massCB1->kpos[1] = coord[1]/NemoMath::pi;
    eff_massCB1->kpos[2] = coord[2]/NemoMath::pi;

    eff_masses_info.push_back(eff_massCB1);


    int size_of_kspace = kspace->get_num_points();
    double* Eband = new double[size_of_kspace];

    // determine path variable of band structure calculation
    vector<double> old_k=kspace->get_point(0).get_coords();
    vector<double> dk_point_m(3) ,dk_point_p(3);
    double*  k_distanceV = new double[size_of_kspace];

    for (int i=0; i<size_of_kspace; i++)
    {
      const vector<double>& k_point = kspace->get_point(i).get_coords();

      old_k[0]=(k_point[0]-old_k[0]);
      old_k[1]=(k_point[1]-old_k[1]);
      old_k[2]=(k_point[2]-old_k[2]);

      Eband[i] = EnergyV[i][subband_number] ;
      if (i>0)
        k_distanceV[i] = k_distanceV[i-1] + NemoMath::vector_norm_3d(&old_k[0]);
      old_k = k_point;
    }
    int* index = new int[size_of_kspace];
    no_pts = 0;
    //In the EK find points of local maxima or minima
    if(find_minima)
      NemoMath::obtain_extrema(size_of_kspace,k_distanceV,Eband,1,index,&no_pts);
    else
      NemoMath::obtain_extrema(size_of_kspace,k_distanceV,Eband,0,index,&no_pts);
    array_indx =0 ;
    valley_count = 2;
    minima_count = 0;
    mstar =0;
    //In all the local maxima and minima found, calculate the effective mass
    // We have already calculated the effective mass at the gamma point.
    // if the Gamma comes again in the set of minima ,then it should be rejected.
    while(minima_count <no_pts)
    {
      array_indx = index[minima_count];
      if((array_indx + 1) < size_of_kspace)
      {
        const vector<double>& k_point = kspace->get_point(array_indx).get_coords();
        const vector<double>& next_k_point = kspace->get_point(array_indx+1).get_coords();
        dk_point[0] = del*(next_k_point[0] - k_point[0]);
        dk_point[1] = del*(next_k_point[1] - k_point[1]);
        dk_point[2] = del*(next_k_point[2] - k_point[2]);
        dk = NemoMath::vector_norm_3d(&(dk_point[0]));
        dk_point_p[0] = k_point[0] +  dk_point[0];
        dk_point_p[1] = k_point[1] +  dk_point[1];
        dk_point_p[2] = k_point[2] +  dk_point[2];
        dk_point_m[0] = k_point[0] -  dk_point[0];
        dk_point_m[1] = k_point[1] -  dk_point[1];
        dk_point_m[2] = k_point[2] -  dk_point[2];
        solve_tb_single_kpoint(k_point, E_kzero);
        solve_tb_single_kpoint(dk_point_m, E_kzerom);
        solve_tb_single_kpoint(dk_point_p, E_kzerop);

        hbar2del2_m0 = (dk)*(dk) * NemoPhys::hbar_nemo*NemoPhys::hbar_nemo / me0_nemounits;

        mstar  =     hbar2del2_m0 / (E_kzerop[subband_number]-2.0*E_kzero[subband_number ]+E_kzerom[subband_number ]);
        meff_info* eff_massInfo = new meff_info();
        eff_massInfo->band_number = subband_number + 1;
        eff_massInfo->no_valley = valley_count;
        eff_massInfo->eff_mass = mstar;
        eff_massInfo->Ene = E_kzero[subband_number];
        eff_massInfo->mdir[0] = dk_point[0]/dk;
        eff_massInfo->mdir[1] = dk_point[1]/dk;
        eff_massInfo->mdir[2] = dk_point[2]/dk;
        eff_massInfo->kpos[0] = k_point[0]/NemoMath::pi;
        eff_massInfo->kpos[1] = k_point[1]/NemoMath::pi;
        eff_massInfo->kpos[2] = k_point[2]/NemoMath::pi;

        distance_from_Gamma =  NemoMath::vector_norm_3d(&eff_massInfo->kpos[0]);
        for(ii=0 ; ii<3; ii++)
        {
          diff_mdir_from_gamma[ii] =  dkG_point[ii] - eff_massInfo->mdir[ii] ;
        }
        normdistance_of_effective_mass_dirs = NemoMath::vector_norm_3d(&diff_mdir_from_gamma[0]);
        if((normdistance_of_effective_mass_dirs > eps) && (distance_from_Gamma > eps))
        {
          eff_masses_info.push_back(eff_massInfo);
          valley_count++;
        }
        minima_count++;
      }
    }
  }
}

void Schroedinger::compute_edges_masses_zincblende(double a)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::compute_edges_masses_zincblende";
  NemoUtils::tic(tic_toc_prefix);



  // determine which band index corresponds to conduction band (CB), heavy-hole (HH), light-hole (LH), split-off (SO)
  string tb_basis = options.get_option("tb_basis", std::string("sp3"));
  int nCB=-1, nHH=-1, nLH=-1, nSO=-1;
  if (tb_basis=="s")
  {
    nCB=0;
  }
  else if (tb_basis == "sp3sstar_SO")
  {
    nCB=8;
    nHH=6;
    nLH=4;
    nSO=2;
  }
  else if (tb_basis == "sp3sstar")
  {
    nCB=4;
    nHH=3;
    nLH=2;
    nSO= -1;
  }
  else if (tb_basis == "sp3d5sstar_SO")
  {
    nCB=8;
    nHH=6;
    nLH=4;
    nSO=2;
  }
  else if (tb_basis == "sp3d5sstar")
  {
    nCB=4;
    nHH=3;
    nLH=2;
    nSO= -1;
  }
  else if (tb_basis == "em")
  {
    nCB=2;
    nHH=2;
    nLH=2;
    nSO=2;
  }
  else if (tb_basis == "kp")
  {
    nCB=6;
    nHH=4;
    nLH=2;
    nSO=0;
  }
  else
  {
    NEMO_EXCEPTION("[Schroedinger(\""+get_name()
        +"\")] did not recognize band model - please implement band indices in Schroedinger.cpp around line 350.");
  }

  // ----------------------------------------------
  // compute energies at selected points in k-space
  // ----------------------------------------------

  double pi_a = NemoMath::pi/a;
  const double del = options.get_option("compute_edges_masses_delta", 0.001); // in pi/a
  vector<double> coord(3, 0.0);

  // ----------------------
  // Gamma
  // ----------------------

  // calculate G, G+pi/a*(delta,0,0), G-pi/a*(delta,0,0), G+pi/a*(delta,delta,0), G-pi/a*(delta,delta,0), G+pi/a*(delta,delta,delta), G-pi/a*(delta,delta,delta),
  vector<double> EG0, EGp_100, EGm_100, EGp_110, EGm_110, EGp_111, EGm_111;
  coord[0] =  0.0*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EG0);
  coord[0] =  del*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EGp_100);
  coord[0] = -del*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EGm_100);
  coord[0] =  del*pi_a;
  coord[1] =  del*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EGp_110);
  coord[0] = -del*pi_a;
  coord[1] = -del*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EGm_110);
  coord[0] =  del*pi_a;
  coord[1] =  del*pi_a;
  coord[2] =  del*pi_a;
  solve_tb_single_kpoint(coord, EGp_111);
  coord[0] = -del*pi_a;
  coord[1] = -del*pi_a;
  coord[2] = -del*pi_a;
  solve_tb_single_kpoint(coord, EGm_111);
  /*  out_param["EG0"] = EG0[0];
    out_param["EGp_100"] = EGp_100;
    out_param["EGm_100"] = EGm_100;
    out_param["EGp_110"] = EGp_110;
    out_param["EGm_110"] = EGm_110;
    out_param["EGp_111"] = EGp_111;
    out_param["EGm_111"] = EGm_111;*/


  // note that the stepsize in [110] direction is actually sqrt(2)*delta, in [111] direction sqrt(3)*delta

  // ----------------------
  // X-direction
  // ----------------------

  // find coordinate along X where CB is minimal (apart from Gamma). note: BZ extends until 2pi/a in X-direction

  /*
  vector<double> EX0, p_result;
  vector<double> p1(3);
  p1[0] = 1.2*pi_a;
  p1[1] = 0.0*pi_a;
  p1[2] = 0.0*pi_a;
  solve_tb_single_kpoint(p1, EX0);
  double EcX_1 = EX0[nCB];
  vector<double> p2(3);
  p2[0] = 1.6*pi_a;
  p2[1] = 0.0*pi_a;
  p2[2] = 0.0*pi_a;
  solve_tb_single_kpoint(p2, EX0);
  double EcX_2 = EX0[nCB];
  vector<double> p3(3);
  p3[0] = 2.0*pi_a;
  p3[1] = 0.0*pi_a;
  p3[2] = 0.0*pi_a;
  solve_tb_single_kpoint(p3, EX0);
  double EcX_3 = EX0[nCB];
  double EXmin; // result
  double eps = 0.001*pi_a;



  if (EcX_1<EcX_2 && EcX_1<EcX_3)
  {
    p2 = p1;
    this->find_minimum_energy(nCB, p1, p2, p3, EcX_1, p_result, EXmin, eps);
  }
  if (EcX_2<EcX_1 && EcX_2<EcX_3)
  {
    this->find_minimum_energy(nCB, p1, p2, p3, EcX_2, p_result, EXmin, eps);
  }
  if (EcX_3<EcX_1 && EcX_3<EcX_2)
  {
    p2 = p3;
    this->find_minimum_energy(nCB, p1, p2, p3, EcX_3, p_result, EXmin, eps);
  }
  double Xmin = p_result[0];

   */

  msg << "Calculating Schrodinger in Gamma-X direction. Searching for Band Minima. \n";
  vector<double> p_Gamma(3), p_Xpoint(3),p_X_inbetween(3) , pXmin(3);
  p_Gamma[0] = 0.0*pi_a;
  p_Gamma[1] = 0.0*pi_a;
  p_Gamma[2] = 0.0*pi_a;

  p_Xpoint[0] = 2.0*pi_a;
  p_Xpoint[1] = 0.0*pi_a;
  p_Xpoint[2] = 0.0*pi_a;

  int number_of_kpoints = 31;
  double Emin_X = 100;
  vector<double> EXmin;
  pXmin[0] = 0;
  pXmin[1] = 0;
  pXmin[2] = 0;

  for(int ii=0; ii< (number_of_kpoints -1) ; ii++)
  {
    for(int jj=0 ; jj < 3; jj++)
    {
      p_X_inbetween[jj] = p_Gamma[jj] + (p_Xpoint[jj] - p_Gamma[jj])*(((double)ii)/(number_of_kpoints -1));
    }
    solve_tb_single_kpoint(p_X_inbetween, EXmin);
    cout<<p_X_inbetween[0]<<","<<p_X_inbetween[1]<<","<<p_X_inbetween[2]<<","<<EXmin[nCB]<<endl;
    if(Emin_X > EXmin[nCB])
    {
      Emin_X = EXmin[nCB];
      pXmin[0] = p_X_inbetween[0] ;
      pXmin[1] = p_X_inbetween[1] ;
      pXmin[2] = p_X_inbetween[2] ;
      //cout<<pXmin[0]<<","<<pXmin[1]<<","<<pXmin[2]<<","<<Emin_X<<","<<EXmin[nCB]<<endl;
    }
  }

  /*
  msg << "CB along X has a minimum at " << Xmin/pi_a << "+-" << eps/2/pi_a << " [pi/a] (E=" << EXmin << ")\n";
  NEMO_ASSERT(Xmin>1.2*pi_a, "[Schroedinger(\""+get_name()+"\")] X-minimum was at the boundary of the test domain. please increase test domain.");
   */
  // calculate X, X+pi/a*(delta,0,0), X-pi/a*(delta,0,0), X+pi/a*(0,delta,0), X-pi/a*(0,delta,0)
  vector<double> EXlp, EXlm, EXtp, EXtm,EX0;
  coord[0] = pXmin[0];
  coord[1] = pXmin[1];
  coord[2] = pXmin[2];
  solve_tb_single_kpoint(coord, EX0);
  coord[0] = pXmin[0]+del*pi_a;
  coord[1] = pXmin[1];
  coord[2] = pXmin[2];
  solve_tb_single_kpoint(coord, EXlp);
  coord[0] = pXmin[0]-del*pi_a;
  coord[1] = pXmin[1];
  coord[2] = pXmin[2];
  solve_tb_single_kpoint(coord, EXlm);
  coord[0] = pXmin[0];
  coord[1] = pXmin[1] + del*pi_a;
  coord[2] = pXmin[2];
  solve_tb_single_kpoint(coord, EXtp);
  coord[0] = pXmin[0];
  coord[1] = pXmin[1]-del*pi_a;
  coord[2] = pXmin[2];
  solve_tb_single_kpoint(coord, EXtm);




  // ----------------------
  // L-direction
  // ----------------------

  // find coordinate along L where CB is minimal (apart from Gamma). note: BZ extends until pi/a in L-direction

  msg << "Calculating Schrodinger in Gamma-L direction. Searching for Band Minima. \n";
  vector<double>  p_Lpoint(3),p_L_inbetween(3), pLmin(3);


  p_Lpoint[0] = 1.0*pi_a;
  p_Lpoint[1] = 1.0*pi_a;
  p_Lpoint[2] = 1.0*pi_a;


  double Emin_L = 100;
  vector<double> ELmin;
  pLmin[0] = 0;
  pLmin[1] = 0;
  pLmin[2] = 0;

  for(int ii=0; ii< (number_of_kpoints -1) ; ii++)
  {
    for(int jj=0 ; jj < 3; jj++)
    {
      p_L_inbetween[jj] = p_Gamma[jj] + (p_Lpoint[jj] - p_Gamma[jj])*(((double)ii)/(number_of_kpoints -1));
    }
    solve_tb_single_kpoint(p_L_inbetween, ELmin);
    if(Emin_L > ELmin[nCB])
    {
      Emin_L = ELmin[nCB];
      pLmin[0] = p_L_inbetween[0] ;
      pLmin[1] = p_L_inbetween[1] ;
      pLmin[2] = p_L_inbetween[2] ;
    }
  }

  /*
  vector<double> EL0;
  p1[0] = 0.6*pi_a;
  p1[1] = 0.6*pi_a;
  p1[2] = 0.6*pi_a;
  solve_tb_single_kpoint(p1, EL0);
  double EcL_1 = EL0[nCB];
  p2[0] = 0.8*pi_a;
  p2[1] = 0.8*pi_a;
  p2[2] = 0.8*pi_a;
  solve_tb_single_kpoint(p2, EL0);
  double EcL_2 = EL0[nCB];
  p3[0] = 1.0*pi_a;
  p3[1] = 1.0*pi_a;
  p3[2] = 1.0*pi_a;
  solve_tb_single_kpoint(p3, EL0);
  double EcL_3 = EL0[nCB];
  double ELmin; // result
  if (EcL_1<EcL_2 && EcL_1<EcL_3)
  {
    p2 = p1;
    this->find_minimum_energy(nCB, p1, p2, p3, EcL_1, p_result, ELmin, eps);
  }
  if (EcL_2<EcL_1 && EcL_2<EcL_3)
  {
    this->find_minimum_energy(nCB, p1, p2, p3, EcL_2, p_result, ELmin, eps);
  }
  if (EcL_3<EcL_1 && EcL_3<EcL_2)
  {
    p2 = p3;
    this->find_minimum_energy(nCB, p1, p2, p3, EcL_3, p_result, ELmin, eps);
  }
  double Lmin = p_result[0];

  //Lmin = 1.0*pi_a;
  msg << "CB along L has a minimum at " << Lmin/pi_a << "+-" << eps/2/pi_a << " [pi/a] (E=" << ELmin << ")\n";
  NEMO_ASSERT(Lmin>0.6*pi_a, "[Schroedinger(\""+get_name()+"\")] L-minimum was at the boundary of the test domain. please increase test domain.");

  // calculate L, L+pi/a*(delta,delta,delta), L-pi/a*(delta,delta,delta), L+pi/a*(delta,-delta,0), L-pi/a*(delta,-delta,0)

   */

  vector<double> ELlp, ELlm, ELtp, ELtm,EL0;
  coord[0] = pLmin[0]         ;
  coord[1] = pLmin[1]         ;
  coord[2] = pLmin[2]         ;
  solve_tb_single_kpoint(coord, EL0);
  coord[0] = pLmin[0]+del*pi_a;
  coord[1] = pLmin[1]+del*pi_a;
  coord[2] = pLmin[2]+del*pi_a;
  solve_tb_single_kpoint(coord, ELlp);
  coord[0] = pLmin[0]-del*pi_a;
  coord[1] = pLmin[1]-del*pi_a;
  coord[2] = pLmin[2]-del*pi_a;
  solve_tb_single_kpoint(coord, ELlm);
  coord[0] = pLmin[0]+del*pi_a;
  coord[1] = pLmin[1]-del*pi_a;
  coord[2] = pLmin[2]         ;
  solve_tb_single_kpoint(coord, ELtp);
  coord[0] = pLmin[0]-del*pi_a;
  coord[1] = pLmin[1]+del*pi_a;
  coord[2] = pLmin[2]         ;
  solve_tb_single_kpoint(coord, ELtm);
  // note that the stepsize in [111] direction is actually sqrt(3)*delta for long, sqrt(2)*delta for trans

  // -------------------------------------------------------------
  // find band edges, masses
  // E''(k0) = hbar^2/m ~ (E(k0+del)-2*E(k0)+E(k0-del)) / (del^2)
  // --> m/m0 = hbar^2/m0*del^2 / (E(k0+del)-2*E(k0)+E(k0-del))
  // -------------------------------------------------------------

  double EG_CB=0.0, EG_HH=0.0, EG_LH=0.0, EG_SO=0.0,
      EX_CB=0.0, EX_HH=0.0, EX_LH=0.0, EX_SO=0.0,
      EL_CB=0.0, EL_HH=0.0, EL_LH=0.0, EL_SO=0.0,
      mG100_CB=0.0, mG100_HH=0.0, mG100_LH=0.0, mG100_SO=0.0,
      mG110_CB=0.0, mG110_HH=0.0, mG110_LH=0.0, mG110_SO=0.0,
      mG111_CB=0.0, mG111_HH=0.0, mG111_LH=0.0, mG111_SO=0.0,
      mXl_CB=0.0, mXt_CB=0.0, mLl_CB=0.0, mLt_CB=0.0;
  double me0_nemounits = NemoPhys::me0_nemo;
  //double me0_nemounits = NemoPhys::electron_mass * NemoPhys::elementary_charge * 1e18;
  double hbar2del2_m0 = (del*pi_a)*(del*pi_a) * NemoPhys::hbar_nemo*NemoPhys::hbar_nemo / me0_nemounits;
  if (nCB!=-1)
  {
    EG_CB = EG0[nCB];
    EX_CB = EX0[nCB];
    EL_CB = EL0[nCB];
    mG100_CB  =     hbar2del2_m0 / (EGp_100[nCB]-2.0*EG0[nCB]+EGm_100[nCB]);
    mG110_CB  = 2.0*hbar2del2_m0 / (EGp_110[nCB]-2.0*EG0[nCB]+EGm_110[nCB]);
    mG111_CB  = 3.0*hbar2del2_m0 / (EGp_111[nCB]-2.0*EG0[nCB]+EGm_111[nCB]);
    mXl_CB    =     hbar2del2_m0 / (EXlp   [nCB]-2.0*EX0[nCB]+EXlm   [nCB]);
    mXt_CB    =     hbar2del2_m0 / (EXtp   [nCB]-2.0*EX0[nCB]+EXtm   [nCB]);
    mLl_CB    = 3.0*hbar2del2_m0 / (ELlp   [nCB]-2.0*EL0[nCB]+ELlm   [nCB]);
    mLt_CB    = 2.0*hbar2del2_m0 / (ELtp   [nCB]-2.0*EL0[nCB]+ELtm   [nCB]);
  }
  if (nHH!=-1)
  {
    EG_HH = EG0[nHH];
    EX_HH = EX0[nHH];
    EL_HH = EL0[nHH];
    mG100_HH  = -    hbar2del2_m0 / (EGp_100[nHH]-2.0*EG0[nHH]+EGm_100[nHH]);
    mG110_HH  = -2.0*hbar2del2_m0 / (EGp_110[nHH]-2.0*EG0[nHH]+EGm_110[nHH]);
    mG111_HH  = -3.0*hbar2del2_m0 / (EGp_111[nHH]-2.0*EG0[nHH]+EGm_111[nHH]);
  }
  if (nLH!=-1)
  {
    EG_LH = EG0[nLH];
    EX_LH = EX0[nLH];
    EL_LH = EL0[nLH];
    mG100_LH  = -    hbar2del2_m0 / (EGp_100[nLH]-2.0*EG0[nLH]+EGm_100[nLH]);
    mG110_LH  = -2.0*hbar2del2_m0 / (EGp_110[nLH]-2.0*EG0[nLH]+EGm_110[nLH]);
    mG111_LH  = -3.0*hbar2del2_m0 / (EGp_111[nLH]-2.0*EG0[nLH]+EGm_111[nLH]);
  }
  if (nSO!=-1)
  {
    EG_SO = EG0[nSO];
    EX_SO = EX0[nSO];
    EL_SO = EL0[nSO];
    mG100_SO  = -    hbar2del2_m0 / (EGp_100[nSO]-2.0*EG0[nSO]+EGm_100[nSO]);
    mG110_SO  = -2.0*hbar2del2_m0 / (EGp_110[nSO]-2.0*EG0[nSO]+EGm_110[nSO]);
    mG111_SO  = -3.0*hbar2del2_m0 / (EGp_111[nSO]-2.0*EG0[nSO]+EGm_111[nSO]);
  }

  // ----------------------------------------------
  // print / save
  // ----------------------------------------------
  stringstream ss;
  ss << "\n[Schroedinger(\""+get_name()+"\")] displaying band edges [eV] and masses [m0]:\n";
  ss << "  Xmin = pi/a*(" << pXmin[0]/pi_a << ",0,0)     (BZ edge at pi/a*(2,0,0))\n";
  ss << "  Lmin = pi/a*(" << pLmin[0]/pi_a << "," << pLmin[1]/pi_a << "," << pLmin[2]/pi_a << ")     (BZ edge at pi/a*(1,1,1))\n";
  ss << "  band         E(G)   E(Xmin)   E(Lmin)    m(G)[100]  m(G)[110]  m(G)[111]  ml(Xmin)  mt(Xmin)  ml(Lmin)  mt(Lmin)\n";
  ss << "  ----------------------------------------------------------------------------------------------------------------\n";
  string eff_mass_string_info1;
  eff_mass_string_info1 = ss.str();
  eff_mass_string_info.push_back(eff_mass_string_info1);
  ss.str("");
  char buf[1000];
  if (nCB!=-1)
  {
    sprintf(buf,"    CB     %+8.5f  %+8.5f  %+8.5f      %7.5f    %7.5f    %7.5f   %7.5f  %7.5f   %7.5f  %7.5f\n", EG_CB, EX_CB, EL_CB, mG100_CB, mG110_CB,
        mG111_CB, mXl_CB, mXt_CB, mLl_CB, mLt_CB);
    ss << buf;

  }
  if (nHH!=-1)
  {
    sprintf(buf,"    HH     %+8.5f  %+8.5f  %+8.5f      %7.5f    %7.5f    %7.5f\n", EG_HH, EX_HH, EL_HH, mG100_HH, mG110_HH, mG111_HH);
    ss << buf;
  }
  if (nLH!=-1)
  {
    sprintf(buf,"    LH     %+8.5f  %+8.5f  %+8.5f      %7.5f    %7.5f    %7.5f\n", EG_LH, EX_LH, EL_LH, mG100_LH, mG110_LH, mG111_LH);
    ss << buf;
  }
  if (nSO!=-1)
  {
    sprintf(buf,"    SO     %+8.5f  %+8.5f  %+8.5f      %7.5f    %7.5f    %7.5f\n", EG_SO, EX_SO, EL_SO, mG100_SO, mG110_SO, mG111_SO);
    ss << buf;
  }
  if (nCB!=-1 && nHH!=-1)
  {
    // nonparabolicity
    vector<double> EGpp_100, EGpp_111;
    coord[0] =  10*del*pi_a;
    coord[1] =     0.0*pi_a;
    coord[2] =     0.0*pi_a;
    solve_tb_single_kpoint(coord, EGpp_100);
    coord[0] =  10*del*pi_a;
    coord[1] =  10*del*pi_a;
    coord[2] =  10*del*pi_a;
    solve_tb_single_kpoint(coord, EGpp_111);

    double Egap = EG_CB - EG_HH;
    double x =   100*hbar2del2_m0*pi_a*pi_a / mG100_CB;
    double alpha  = Egap/x - Egap*EGpp_100[nCB]/(x*x);
    double y = 3*100*hbar2del2_m0*pi_a*pi_a / mG111_CB; // in [111] del was sqrt(3) bigger, and x ~ del^2
    double alpha2 = Egap/y - Egap*EGpp_111[nCB]/(y*y);
    ss << "    CB nonparabolicity: " << alpha << "=" << alpha2 << "  (del=10*" << del << ", Egap=" << Egap << ", x=" << x << ")\n";
  }

  ss << "\n";
  string eff_mass_string_info2;
  eff_mass_string_info2 = ss.str();
  eff_mass_string_info.push_back(eff_mass_string_info2);

  out_param["EG0"] = EG0[0];
  out_param["EGp_100"] = EGp_100[0];
  out_param["EGm_100"] = EGm_100[0];
  out_param["EGp_110"] = EGp_110[0];
  out_param["EGm_110"] = EGm_110[0];
  out_param["EGp_111"] = EGp_111[0];
  out_param["EGm_111"] = EGm_111[0];
  out_param["EG_CB"] = EG_CB;
  out_param["EX_CB"] = EX_CB;
  out_param["EL_CB"] = EL_CB;
  out_param["mG100_CB"] = mG100_CB;
  out_param["mG110_CB"] = mG110_CB;
  out_param["mG111_CB"] = mG111_CB;
  out_param["mXl_CB"] = mXl_CB;
  out_param["mXt_CB"] = mXt_CB;
  out_param["mLl_CB"] = mLl_CB;
  out_param["mLt_CB"] = mLt_CB;
  //  out_param[""] = ;




  //  out_param["EG_HH"] = EG_HH;
  //out_param["Egap"] = Egap;
  //out_param["alpha"] = alpha;
  OUT_NORMAL << "\n\n XXXXXXXXXXXXXXX \n We calculated out_param!  " << mLt_CB <<endl;
  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::compute_edges_masses_wurtzite(double a, double)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::compute_edges_masses_wurtzite";
  NemoUtils::tic(tic_toc_prefix);



  // determine which band index corresponds to conduction band (CB), heavy-hole (HH), light-hole (LH), split-off (SO)
  string tb_basis = options.get_option("tb_basis", std::string("sp3"));
  int nCB=-1, nHH=-1, nLH=-1, nCH=-1;
  if (tb_basis=="s")
  {
    nCB=0;
  }
  else if (tb_basis == "sp3sstar_SO")
  {
    nCB=8;
    nHH=7;
    nLH=6;
    nCH=5;
  }
  else if (tb_basis == "sp3d5sstar_SO")
  {
    nCB=16;
    nHH=14;
    nLH=12;
    nCH=10;
  }
  else if (tb_basis == "em")
  {
    nCB=2;
    nHH=2;
    nLH=2;
    nCH=2;
  }
  else if (tb_basis == "kp")
  {
    nCB=2;
    nHH=2;
    nLH=2;
    nCH=2;
  }
  else
  {
    NEMO_EXCEPTION("[Schroedinger(\""+get_name()
        +"\")] did not recognize band model - please implement band indices in Schroedinger.cpp around line 350.");
  }

  // ----------------------------------------------
  // compute energies at selected points in k-space
  // ----------------------------------------------

  double pi_a = NemoMath::pi/a;
  const double del = options.get_option("compute_edges_masses_delta", 0.001); // in pi/a
  vector<double> coord(3, 0.0);

  // ----------------------
  // Gamma
  // ----------------------

  // calculate G, G+pi/a*(delta,0,0), G-pi/a*(delta,0,0), G+pi/a*(delta,delta,0), G-pi/a*(delta,delta,0), G+pi/a*(delta,delta,delta), G-pi/a*(delta,delta,delta),
  vector<double> EG0, EGp_100, EGm_100, EGp_001, EGm_001;
  coord[0] =  0.0*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EG0);
  coord[0] =  del*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EGp_100);
  coord[0] = -del*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] =  0.0*pi_a;
  solve_tb_single_kpoint(coord, EGm_100);
  coord[0] =  0.0*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] =  del*pi_a;
  solve_tb_single_kpoint(coord, EGp_001);
  coord[0] =  0.0*pi_a;
  coord[1] =  0.0*pi_a;
  coord[2] = -del*pi_a;
  solve_tb_single_kpoint(coord, EGm_001);
  // note that the stepsize in [110] direction is actually sqrt(2)*delta, in [111] direction sqrt(3)*delta

  // -------------------------------------------------------------
  // find band edges, masses
  // E''(k0) = hbar^2/m ~ (E(k0+del)-2*E(k0)+E(k0-del)) / (del^2)
  // --> m/m0 = hbar^2/m0*del^2 / (E(k0+del)-2*E(k0)+E(k0-del))
  // -------------------------------------------------------------

  double EG_CB=0.0, EG_HH=0.0, EG_LH=0.0, EG_CH=0.0,
      mG100_CB=0.0, mG100_HH=0.0, mG100_LH=0.0, mG100_CH=0.0,
      mG001_CB=0.0, mG001_HH=0.0, mG001_LH=0.0, mG001_CH=0.0;
  double me0_nemounits = NemoPhys::me0_nemo;
  //double me0_nemounits = NemoPhys::electron_mass * NemoPhys::elementary_charge * 1e18;
  double hbar2del2_m0 = (del*pi_a)*(del*pi_a) * NemoPhys::hbar_nemo*NemoPhys::hbar_nemo / me0_nemounits;
  if (nCB!=-1)
  {
    EG_CB = EG0[nCB];
    mG100_CB  = hbar2del2_m0 / (EGp_100[nCB]-2.0*EG0[nCB]+EGm_100[nCB]);
    mG001_CB  = hbar2del2_m0 / (EGp_001[nCB]-2.0*EG0[nCB]+EGm_001[nCB]);
  }
  if (nHH!=-1)
  {
    EG_HH = EG0[nHH];
    mG100_HH  = -hbar2del2_m0 / (EGp_100[nHH]-2.0*EG0[nHH]+EGm_100[nHH]);
    mG001_HH  = -hbar2del2_m0 / (EGp_001[nHH]-2.0*EG0[nHH]+EGm_001[nHH]);
  }
  if (nLH!=-1)
  {
    EG_LH = EG0[nLH];
    mG100_LH  = -hbar2del2_m0 / (EGp_100[nLH]-2.0*EG0[nLH]+EGm_100[nLH]);
    mG001_LH  = -hbar2del2_m0 / (EGp_001[nLH]-2.0*EG0[nLH]+EGm_001[nLH]);
  }
  if (nCH!=-1)
  {
    EG_CH = EG0[nCH];
    mG100_CH  = -hbar2del2_m0 / (EGp_100[nCH]-2.0*EG0[nCH]+EGm_100[nCH]);
    mG001_CH  = -hbar2del2_m0 / (EGp_001[nCH]-2.0*EG0[nCH]+EGm_001[nCH]);
  }

  // ----------------------------------------------
  // print / save
  // ----------------------------------------------
  stringstream ss;
  ss << "\n[Schroedinger(\""+get_name()+"\")] displaying band edges [eV] and masses [m0]:\n";
  ss << "  band         E(G)      m(G)[100]  m(G)[001] \n";
  ss << "  --------------------------------------------------------------------------\n";
  string eff_mass_string_info1;
  eff_mass_string_info1 = ss.str();
  eff_mass_string_info.push_back(eff_mass_string_info1);
  ss.str("");
  char buf[1000];
  if (nCB!=-1)
  {
    sprintf(buf,"    CB     %+8.5f    %7.5f    %7.5f\n", EG_CB, mG100_CB, mG001_CB);
    ss << buf;
  }
  if (nHH!=-1)
  {
    sprintf(buf,"    HH     %+8.5f    %7.5f    %7.5f\n", EG_HH, mG100_HH, mG001_HH);
    ss << buf;
  }
  if (nLH!=-1)
  {
    sprintf(buf,"    LH     %+8.5f    %7.5f    %7.5f\n", EG_LH, mG100_LH, mG001_LH);
    ss << buf;
  }
  if (nCH!=-1)
  {
    sprintf(buf,"    CH     %+8.5f    %7.5f    %7.5f\n", EG_CH, mG100_CH, mG001_CH);
    ss << buf;
  }
  ss << "\n";
  string eff_mass_string_info2;
  eff_mass_string_info2 = ss.str();
  eff_mass_string_info.push_back(eff_mass_string_info2);

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::find_minimum_energy(int idx, const vector<double>& p1, const vector<double>& p3, const vector<double>& p5, double E3,
    vector<double>& p_result, double& E_result, double eps)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::find_minimum_energy";
  NemoUtils::tic(tic_toc_prefix);



  // if p1[d]-p3[d]<eps for all dimensions d then we're done
  if (NemoMath::abs(p1[0]-p3[0])<eps && NemoMath::abs(p1[1]-p3[1])<eps && NemoMath::abs(p1[2]-p3[2])<eps)
  {
    p_result = p3;
    E_result = E3;
    return;
  }

  vector<double> p2(3);
  for (short d=0; d<3; d++) p2[d] = 0.5*(p1[d]+p3[d]);
  vector<double> p4(3);
  for (short d=0; d<3; d++) p4[d] = 0.5*(p3[d]+p5[d]);
  vector<double> Etmp;
  solve_tb_single_kpoint(p2, Etmp);
  double E2 = Etmp[idx];
  solve_tb_single_kpoint(p4, Etmp);
  double E4 = Etmp[idx];

  // we know E1>E3<E5

  if (E2<E3)          // E2 is lowest - solve recursively with 1-2-3
  {
    find_minimum_energy(idx, p1, p2, p3, E2, p_result, E_result, eps);
  }
  else if (E4<E3)     // E4 is lowest - solve recursively with 3-4-5
  {
    find_minimum_energy(idx, p3, p4, p5, E4, p_result, E_result, eps);
  }
  else                // E3 is lowest - solve recursively with 2-3-4
  {
    find_minimum_energy(idx, p2, p3, p4, E3, p_result, E_result, eps);
  }

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::allreduce_map(map<unsigned int, double>& data)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::allreduce_map";
  NemoUtils::tic(tic_toc_prefix);


  int num_verts = (int) data.size();
  vector<double> tmpvec(num_verts);
  int counter=0;
  for(map<unsigned int, double>::iterator it = data.begin(); it!=data.end(); it++)
  {
    tmpvec[counter] = it->second;
    counter++;
  }
  NEMO_ASSERT(counter==num_verts, "[Schroedinger(\""+get_name()+"\")::allreduce_map] counter==num_verts failed (1).");

  vector<double> tmpsum(num_verts, 0.0);
  //vector<double> & tmpsum = tmpvec;
  double* sendbuf = &tmpvec[0];  // buffers must not be aliased for MPI_Allreduce
  double* recvbuf = &tmpsum[0];


  //we have to reduce among all k-points but for a single geometry partition
  //for that we take create intersection of two communicators

  MPI_Group all_partition_single_variable;
  MPI_Comm_group(k_space_communicator, &all_partition_single_variable);

  MPI_Group single_partition_all_variable;
  MPI_Comm_group(holder.one_partition_total_communicator, &single_partition_all_variable);

  MPI_Group single_partition_single_variable;
  MPI_Group_intersection (all_partition_single_variable,
      single_partition_all_variable,
      &single_partition_single_variable );


  MPI_Comm single_part_comm;
  MPI_Comm_create(k_space_communicator,
      single_partition_single_variable,
      &single_part_comm );

  int my_rank;
  MPI_Comm_rank(single_part_comm, &my_rank);

  if (gather_density_everywhere)
    MPI_Allreduce(sendbuf, recvbuf, num_verts, MPI_DOUBLE, MPI_SUM, single_part_comm);
  else
    MPI_Reduce(sendbuf, recvbuf, num_verts, MPI_DOUBLE, MPI_SUM, 0,single_part_comm);

  if (gather_density_everywhere || my_rank == 0)
  {
    counter = 0;
    for(map<unsigned int, double>::iterator it = data.begin(); it!=data.end(); it++)
    {
      it->second = tmpsum[counter];
      counter++;
    }
  }
  MPI_Comm_free(&single_part_comm);
  NEMO_ASSERT(counter==num_verts, "[Schroedinger(\""+get_name()+"\")::allreduce_map] counter==num_verts failed (2).");

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::set_eigensolver_options()
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::set_eigensolver_options";
  NemoUtils::tic(tic_toc_prefix);


  std::string prefix="Schroedinger::set_eigensolver_options() :";

  //Roza Kotlyar: 3/2/2011
  nsetopt=nsetopt+1;

  //---------------------------------------------------------------------
  // check input deck for user-defined eigensolver options
  //---------------------------------------------------------------------
  string eigen_values_solver = options.get_option("eigen_values_solver", string("lapack")); // <ss> do not take away string()
  num_requested_eigenvalues  = options.get_option("number_of_eigenvalues", hamiltonian->get_num_rows());
  unsigned int max_number_iterations = options.get_option("max_number_iterations",300);
  double max_error = options.get_option("convergence_limit",1e-8);
  //JF and JMS added below for lanczos options
  if (eigen_values_solver=="lanczos" || eigen_values_solver=="block_lanczos" || eigen_values_solver == "lanczos_then_krylovschur")
  {
    string prefix="Schroedinger::set_eigensolver_options(lanczos) - > ";
    string tmp;
    int ISEED;
    //    int block_p;
    int number_of_eigenvalues = 0; //CB
    int number_of_eigenvalues2 = 0; //VB
    int maximum_number_of_iterations = 0;
    int convergence_method = 0;
    int check_every_steps_number = 1;
    int init_check_steps_number = 0;
    double energy_minimum = 0.0; //CB
    double energy_maximum = 0.0; //CB
    double energy_minimum2 = 0.0; //VB
    double energy_maximum2 = 0.0; //VB
    double resolution = 0.0;
    double tolerance = 0.0;
    double beta_degradation = 0.0;
    string tb_basis;
    bool tridiag_output;
    bool trace_output;
    int max_inv_iter = 0;
    double conv_bias = 0.0;
    double conv_gain = 0.0;

    //need to know whether to calculate eigenfunctions or not
    bool eigfun_output    =     has_output("eigenfunctions")      || has_output("eigenfunctions_VTK")
                                    || has_output("eigenfunctions_VTP")   || has_output("eigenfunctions_VTU")
                    || has_output("eigenfunctions_DX")   || has_output("eigenfunctions_XYZ")
                    || has_output("eigenfunctions_Silo") || has_output("eigenfunctions_1D") || has_output("eigenfunctions_Point3D");

    bool eigfun_k0_output =     has_output("eigenfunctions_k0")      || has_output("eigenfunctions_VTK_k0")
                                    || has_output("eigenfunctions_VTP_k0")   || has_output("eigenfunctions_VTU_k0")
                    || has_output("eigenfunctions_DX_k0")   || has_output("eigenfunctions_XYZ_k0")
                    || has_output("eigenfunctions_Silo_k0") || has_output("eigenfunctions_1D_k0" ) || has_output("eigenfunctions_Point3D");

    bool calculate_eigenfun = eigfun_output || eigfun_k0_output;

    // read the lanczos input deck
    ISEED = 38467;  //  initial value for random number generator
    if(options.check_option("lanczos_iseed")) ISEED=options.get_option("lanczos_iseed",0.);
    msg << prefix << "Lanczos Initial Random Numbers Seed = " << ISEED << "\n";

    beta_degradation = 1.e-6;  // default value
    if(options.check_option("beta_degradation")) beta_degradation=options.get_option("beta_degradation",0.);
    msg << prefix << "Lanczos beta degradation = " << beta_degradation << "\n";

    if(options.check_option("tb_basis")) tb_basis=options.get_option("tb_basis",string("sp3"));
    msg << prefix << "Lanczos tight-binding model = " << tb_basis << "\n";

    if(options.check_option("number_of_eigenvalues")) number_of_eigenvalues=options.get_option("number_of_eigenvalues",0);
    msg << prefix << "Lanczos number of eigenvalues = " << number_of_eigenvalues << "\n";

    number_of_eigenvalues2 = -1;
    if(options.check_option("number_of_eigenvalues_VB")) number_of_eigenvalues2=options.get_option("number_of_eigenvalues_VB",-1);
    msg << prefix << "Lanczos number of eigenvalues (VB) = " << number_of_eigenvalues2 << "\n";

    if(number_of_eigenvalues2 != -1)
    {
      num_requested_eigenvalues += number_of_eigenvalues2;
      number_of_eigenvalues_to_extract += number_of_eigenvalues2;
    }

    if(options.check_option("max_number_iterations")) maximum_number_of_iterations=options.get_option("max_number_iterations",0);
    msg << prefix << "Lanczos maximum number of iterations = " << maximum_number_of_iterations << "\n";

    if(options.check_option("check_every_steps_number")) check_every_steps_number=options.get_option("check_every_steps_number",0);
    msg << prefix << "Lanczos check every steps = " << check_every_steps_number << "\n";

    init_check_steps_number = check_every_steps_number;
    if(options.check_option("init_check_steps_number")) init_check_steps_number = options.get_option("init_check_steps_number",check_every_steps_number);
    msg << prefix << "Lanczos initial check steps = " << init_check_steps_number << "\n";

    if(options.check_option("energy_minimum")) energy_minimum=options.get_option("energy_minimum",0.0);
    msg << prefix << "Lanczos minimum value for eigenvalues = " << energy_minimum << "eV\n";

    if(options.check_option("energy_maximum")) energy_maximum=options.get_option("energy_maximum",0.0);
    msg << prefix << "Lanczos maximum value for eigenvalues = " << energy_maximum << "eV\n";

    if(options.check_option("energy_minimum_VB")) energy_minimum2=options.get_option("energy_minimum_VB",0.0);
    msg << prefix << "Lanczos minimum value for eigenvalues (VB) = " << energy_minimum << "eV\n";

    if(options.check_option("energy_maximum_VB")) energy_maximum2=options.get_option("energy_maximum_VB",0.0);
    msg << prefix << "Lanczos maximum value for eigenvalues (VB) = " << energy_maximum << "eV\n";

    if(options.check_option("convergence_method")) tmp=options.get_option("convergence_method",string(""));
    msg << prefix << "Lanczos convergence method for eigenvalues = " << tmp << "\n";
    //Michael - why are we using ints for options instead of strings?
    if(tmp=="full_convergence") convergence_method=1;
    else if(tmp=="partial_convergence") convergence_method=2;
    // else if(tmp=="binning") convergence_method=3;
    else
    {
      msg << "Unknown convergence method! Please, choose among full_convergence, partial_convergence or binning.\n";
      msg << "Default is full_convergence";
      convergence_method=1;
    }

    resolution=1.e-6;
    if(options.check_option("resolution")) resolution=options.get_option("resolution",0.0);
    msg << prefix << "Lanczos eigenvalues resolution = " << resolution << "\n";

    tolerance=1.e-6;
    if(options.check_option("tolerance")) tolerance=options.get_option("tolerance",0.0);
    msg << prefix << "Lanczos eigenvalues tolerance = " << tolerance << "\n";

    tridiag_output = false;
    if(options.check_option("tridiag_output")) tridiag_output=options.get_option("tridiag_output",false);

    trace_output = false;
    if(options.check_option("trace_output")) trace_output=options.get_option("trace_output",false);

    max_inv_iter = 20;
    if(options.check_option("max_inverse_iterations")) max_inv_iter=options.get_option("max_inverse_iterations",0);

    conv_bias = 1E-80;
    if(options.check_option("convergence_bias")) conv_bias =  options.get_option("convergence_bias", 0.0);

    conv_gain = 1E-8;
    if(options.check_option("convergence_gain")) conv_gain = options.get_option("convergence_gain",0.0);




    /*
       // assemble hamiltonian matrix
       if(!combine_H)
        this->assemble_hamiltonian();
       else
        combine_hamiltonian();
     */
    // definition of the eigensolver
    if(eigen_values_solver=="lanczos")
    {
      EigensolverLanczos& eig_solver = *(dynamic_cast<EigensolverLanczos*>(solver));
      eig_solver.set_matrix(hamiltonian);
      eig_solver.set_solver(tb_basis,number_of_eigenvalues,number_of_eigenvalues2,calculate_eigenfun,
          maximum_number_of_iterations,
          check_every_steps_number,
          init_check_steps_number,
          (double)(ISEED),
          energy_minimum,
          energy_maximum,
          energy_minimum2,
          energy_maximum2,
          convergence_method,
          resolution,
          tolerance,beta_degradation,
          tridiag_output, trace_output,
          max_inv_iter, conv_bias, conv_gain,get_name());
    }
    else if (eigen_values_solver == "lanczos_then_krylovschur")
    {
      EigensolverLanczosThenSlepc& eig_solver = *(dynamic_cast<EigensolverLanczosThenSlepc*>(solver));
      eig_solver.set_matrix(hamiltonian);

      //need to get krylovschur parameters from input deck
      std::string linsolvertype       = options.get_option("linear_solver", string(""));
      std::string preconditioner      = options.get_option("preconditioner", string("lu"));
      std::string transformation_type = options.get_option("solver_transformation_type",string("shift")); //should use shift for this eigensolver)
      int ncv = options.get_option("ncv",0);
      int mpd = options.get_option("mpd",0);
      double max_error = options.get_option("convergence_limit",1e-8);
      std::string monitor_convergence = options.get_option("monitor_convergence",string("false"));

      //How many eigenvalues should we ask krylovschur for? I presume that
      //this number should be larger than what we ask Lanczos for to get the
      //degenerate eigenvalues that Lanczos is discarding
      int num_eigv_ks = options.get_option("number_of_eigenvalues_ks",number_of_eigenvalues);
      //max iterations for K.S
      int max_iterations_ks = options.get_option("max_number_iterations_ks",maximum_number_of_iterations);

      std::string mumps_ordering = options.get_option("mumps_ordering",string(""));
      bool overlap = options.get_option("calculate_elastic_overlap",false);


      eig_solver.set_solver(tb_basis,number_of_eigenvalues,
          maximum_number_of_iterations,
          check_every_steps_number,       //lanczos solver parameter
          init_check_steps_number,
          (double) (ISEED),
          energy_minimum,
          energy_maximum,
          convergence_method,
          resolution,
          tolerance,beta_degradation,
          tridiag_output, trace_output,
          max_iterations_ks,   //krylovschur solver parameter
          num_eigv_ks,
          linsolvertype,preconditioner,
          transformation_type,
          ncv, mpd,
          max_error,
          monitor_convergence,
          mumps_ordering,
          overlap);
    }
    else
    {
      int block_p = 0;
      NEMO_ASSERT(options.check_option("p_parameter"),"[Schroedinger(\""+get_name()+"\")] please define p_parameter when using block Lanczos.");
      if(options.check_option("p_parameter")) block_p=(int)(options.get_option("p_parameter",2));
      msg << prefix << "Block Lanczos p parameter = " << block_p << "\n";

      EigensolverBlockLanczos& eig_solver = *(dynamic_cast<EigensolverBlockLanczos*>(solver));
      eig_solver.set_matrix(hamiltonian);
      eig_solver.set_solver(tb_basis,number_of_eigenvalues,calculate_eigenfun,
          maximum_number_of_iterations,
          check_every_steps_number,
          init_check_steps_number,
          (double)(ISEED),
          energy_minimum,
          energy_maximum,
          convergence_method,
          resolution,
          tolerance,beta_degradation,
          tridiag_output, trace_output,
          max_inv_iter, conv_bias, conv_gain,block_p);
    }
  }
  else if (eigen_values_solver=="feast")
  {
#ifndef FEAST_UNAVAILABLE
    EigensolverFeastComplex& eig_solver = *(dynamic_cast<EigensolverFeastComplex*>(solver));
    double Emin = options.get_option("lower_search_limit",0.0);
    double Emax = options.get_option("upper_search_limit",1.0);
    eig_solver.set_search_interval(Emin,Emax);
#endif
  }
  else
  {
    EigensolverSlepc& eig_solver = *(dynamic_cast<EigensolverSlepc*>(solver));

    string linsolvertype       = options.get_option("linear_solver", string(""));
    string preconditioner      = options.get_option("preconditioner", string("lu"));
    string transformation_type = options.get_option("solver_transformation_type",string("sinvert"));
    if (eigen_values_solver!="lapack")
    {
      num_requested_eigenvalues = ((int)num_requested_eigenvalues < (int)hamiltonian->get_num_rows())?num_requested_eigenvalues:hamiltonian->get_num_rows();
    }
    else
    {
      num_requested_eigenvalues=std::min(num_requested_eigenvalues, hamiltonian->get_num_rows());
      //   num_requested_eigenvalues = hamiltonian->get_num_rows();
      transformation_type = "none";//because we don't need any specral transformation for lapack
    }
    if (eigen_values_solver=="jd" || eigen_values_solver=="gd")
    {
      transformation_type = "precond";
    }
    double shift = options.get_option("shift",0.001);

    string monitor_convergence = options.get_option("monitor_convergence", string("false"));


    // -------------------------------------------------------------------------------------
    // set eigensolver (SLEPc) options:
    // - type of eigensolver
    // - number of evals to be computed
    // - spectral transformation (default: shift-and-invert)
    // - shift value (default: 0.001)
    // - type of preconditioner for linear solver used in shift-and-invert (default: LU) (includes mumps and superlu options)
    // - convergence parameters
    // - convergence monitoring
    // - spectral region (automatic, not set by user)
    // the linear solver type needed for shift-and-invert is not explicitly set.
    // the problem type is always Hermitian
    // -------------------------------------------------------------------------------------
    eig_solver.set_solver_type   (eigen_values_solver);
    eig_solver.set_num_evals     (num_requested_eigenvalues);
    if (overlap == NULL)
      eig_solver.set_problem_type  ("eps_hermitian");
    else
      eig_solver.set_problem_type  ("eps_gen_hermitian");

    eig_solver.set_linsolver_type(linsolvertype);
    eig_solver.set_precond_type  (preconditioner); // discrimination mumps,superlu this is now being done in the eigensolver class

    // determine thresholds below/above which a particle is regarded as an electron/hole
    // this is not connected to the eigensolver, but needed for the shift values
    if (_automatic_threshold)
    {
      double Ec_min = this->get_conduction_band_min(); // min(Ec-e*phi)
      double Ev_max = this->get_valence_band_max();    // max(Ev-e*phi) note: for strong band bending Ec_min<Ev_max is possible


      double cutoff_distance = options.get_option("cutoff_distance_to_bandedge", 0.2*(Ec_min-Ev_max));
      _electron_threshold_energy = Ec_min - cutoff_distance; //determine_electron_threshold();
      _hole_threshold_energy     = Ev_max + cutoff_distance; //determine_hole_threshold();

      msg.print_message2(MSG_INFO,"[Schroedinger] electron threshold at %.3g, hole threshold at %.3g (%s potential, Ecmin=%.3g, Evmax=%.3g)",
          _electron_threshold_energy, _hole_threshold_energy, (use_potential ? "includes" : "excludes"), Ec_min, Ev_max);
    }

    //set-up ordering (This block of code used to be below, but was moved up to this location by Bozidar)
    string charge_model=options.get_option("charge_model",string("electron_hole"));
    NEMO_ASSERT(charge_model == "electron_hole" || charge_model == "electron_core", this->get_name() + ": unknown charge model\n");

    if (electron_density)
    {
      eig_solver.sort_like_electrons = true;
      eig_solver.strict_threshold = true;
    }
    else if (hole_density)
    {
      eig_solver.sort_like_electrons = false;
      eig_solver.strict_threshold = true;
    }
    else if (calculate_band_structure)
    {
      eig_solver.sort_like_electrons = true;
      eig_solver.strict_threshold = false;
    }

    if (charge_model=="electron_core")
    {
      eig_solver.sort_like_electrons = true;
      eig_solver.strict_threshold = false;
    }

    eig_solver.sort_like_electrons = options.get_option("sort_like_electrons", eig_solver.sort_like_electrons);
    if(eig_solver.sort_like_electrons)
      eig_solver.sort_like = std::string("electrons");
    else
      eig_solver.sort_like = std::string("holes");
    eig_solver.strict_threshold = options.get_option("strict_threshold", eig_solver.strict_threshold);

    if (charge_model=="electron_core")
    {
      // in this charge model there are only electrons
      eig_solver.set_spectral_region(EigensolverSlepc::smallest_real);
      eig_solver.set_transformation_type(transformation_type);
      eig_solver.set_shift_sigma(shift);
    }
    else
    {
      eig_solver.set_spectral_region(EigensolverSlepc::default_region); // SLEPc>=3.1 required
      eig_solver.set_transformation_type(transformation_type);

      if (_automatic_threshold)
      {
        if ((electron_density && hole_density) || (calculate_band_structure && !eig_solver.strict_threshold))
          shift = 0.5 * (_electron_threshold_energy + _hole_threshold_energy);
        else if (calculate_band_structure && eig_solver.strict_threshold && (eig_solver.sort_like == "electrons")) //option added by Bozidar
        {
          shift = _electron_threshold_energy;
          if (eigen_values_solver == "arpack")
          {
            eig_solver.set_transformation_type("sinvert"); //for krylovschur this can be "none", but seems faster with "sinvert" (Bozidar)
            eig_solver.set_spectral_region(EigensolverSlepc::largest_real); // SLEPc>=3.1 required
          }
          else if (eigen_values_solver == "krylovschur")
          {
            eig_solver.set_transformation_type("sinvert"); //for krylovschur this can be "none", but seems faster with "sinvert" (Bozidar)
            eig_solver.set_spectral_region(EigensolverSlepc::target_largest_real); // SLEPc>=3.1 required
          }
        }
        else if (calculate_band_structure && eig_solver.strict_threshold && (eig_solver.sort_like == "holes")) //option added by Bozidar
        {
          shift = _hole_threshold_energy;
          eig_solver.set_transformation_type("sinvert");
          if (eigen_values_solver == "arpack")
            eig_solver.set_spectral_region(EigensolverSlepc::smallest_real); // SLEPc>=3.1 required
          else if (eigen_values_solver == "krylovschur")
            eig_solver.set_spectral_region(EigensolverSlepc::target_smallest_real); // SLEPc>=3.1 required
        }
        else if (electron_density)
        {
          shift = _electron_threshold_energy;
          //----------------------------------------------
          //Roza Kotlyar: 3/2/2011
          //convergence scheme based on eigenlevels from previous iteration
          if(do_esolvershift_fromsolution)
          {
            if((done_firstiter_esolvershift_fromsolution>0.5) && (eminecsub-_electron_threshold_energy)>enthresh_foresolvershift)
            {
              shift = enrate_foresolvershift*(eminecsub-_electron_threshold_energy);
            }
          }
          //----------------------------------------------
        }
        else if (hole_density)
        {
          shift = _hole_threshold_energy;
          //----------------------------------------------
          //Roza Kotlyar: 3/2/2011
          //convergence scheme based on eigenlevels from previous iteration
          if(do_esolvershift_fromsolution)
          {
            if((done_firstiter_esolvershift_fromsolution>0.5) && (_hole_threshold_energy-emaxevsub)>enthresh_foresolvershift)
            {
              shift = enrate_foresolvershift*(_hole_threshold_energy-emaxevsub);
            }
          }
          //----------------------------------------------
        }
      }
      if (options.check_option("fixed_shift"))
      {
        shift = options.get_option("fixed_shift",0.0);
      }
      //----------------------------------------------
      //Roza Kotlyar: 3/2/2011
      /*
        //M.P. this code gives problems for QD simulations
            if (ifinitialized_shift_foresolver)
            {
              get_ifinitialized_shift_foresolver(shift);
            }
            msg<<"Set automatic shift [Schrodinger]: "<<shift<<"\n";

            if(my_rank_sch == 0)
            {
              set_shift_foresolver(shift);
            }
       */
      //----------------------------------------------
      // shift *= -1;
      // <ss> no no, shift above is good. In shift-and-invert A --> (A-sigma*I)^-1, in shift A-->A+sigma*I


      eig_solver.set_shift_sigma(shift);
    }

    eig_solver.set_convergence_params(max_error, max_number_iterations);

    if (monitor_convergence=="true" || monitor_convergence=="1" || monitor_convergence=="yes")
    {
      eig_solver.monitor_convergence(true);
    }

    if (options.check_option("mpd"))
    {
      int mpd = options.get_option("mpd", 0);
      eig_solver.set_mpd(mpd);
    }

    if (options.check_option("ncv"))
    {
      int ncv = options.get_option("ncv", 0);
      eig_solver.set_ncv(ncv);
    }
    if (options.check_option("mumps_ordering"))
    {
      string mumps_ordering = options.get_option("mumps_ordering", string(""));
      eig_solver.set_mumps_ordering(mumps_ordering);
    }


    number_of_eigenvalues_to_extract = options.get_option("number_of_eigenvalues_to_use", num_requested_eigenvalues);
    if (options.check_option("number_of_eigenvalues_to_use"))
      number_of_eigenvalues_to_extract = std::min(number_of_eigenvalues_to_extract,num_requested_eigenvalues);
    eig_solver.set_num_evals_to_extract(number_of_eigenvalues_to_extract);

    //TODO:JC if ciss
    if (eigen_values_solver == "ciss")
    {
      //if using ciss there are several other options that need to be set.

      //1. set regions
      double center = options.get_option("region_center", 0.0);
      double radius = options.get_option("region_radius", 1.01);
      double vscale = options.get_option("region_vertical_scale", 1.0);
      double radius_inner = options.get_option("region_inner_radius", .99);
      eig_solver.set_ciss_regions(center, radius, vscale, radius_inner);

      //2. set sizes
      int ip = options.get_option("number_of_integration_points", 16);
      int bs = options.get_option("block_size", 16);
      int ms = options.get_option("moment_size", 16);
      int npart = options.get_option("real_space_partitions", 1);
      int bsmax = options.get_option("block_size_max", 16);
      int isreal = 0; //Petscbool is really an int in C
      std::string tb_kind = options.get_option("tb_basis", std::string("sp3"));
      if (tb_kind.find("SO") != std::string::npos)
        isreal = 1;
      int savelu = options.get_option("save_lu", bool(true));
      int isdense = options.get_option("isdense", bool(false));
      int isring = options.get_option("isring", bool(false));
      eig_solver.set_ciss_sizes(ip, bs, ms, npart, bsmax, isreal,
          savelu, isdense, isring);

      //3. set refinements
      int refine_inner = options.get_option("refine_inner_loop_iterations", 2);
      int refine_outer = options.get_option("refine_outer_loop_iterations", 0);
      int refine_blsize = options.get_option("refine_block_size_number", 0);
      eig_solver.set_ciss_refinements(refine_inner, refine_outer, refine_blsize);

      //4. set thresholds
      //
      //
    }


  }

  NemoUtils::toc(tic_toc_prefix);
}


//! Calculate the eigenvalues at a single k_point
void Schroedinger::solve_tb_single_kpoint(const vector<double>& k_point, vector<double>& EnergyV)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::solve_tb_single_kpoint";
  NemoUtils::tic(tic_toc_prefix);





  this->k_vector = k_point;

  double t1 = NemoUtils::get_time();
  if(!combine_H)
    this->assemble_hamiltonian();
  else
    combine_hamiltonian();
  if (overlap != NULL)
  {
    bool check_det = options.get_option("Check_determinant", bool(false) );
    if(check_det)
    {
      double deter;
      this->Check_determinant(deter);
      OUT_ERROR<<deter<<endl;
    }
  }
  set_eigensolver_options();
  // do NOT execute this in do_init() because shift values might change with potential and do_solve might be called several times!

  // Jun Huang: transform the Hamiltonian into mode space and solve the eigenvalue problem therein
  PetscMatrixParallelComplex* transformed_hamiltonian  = NULL;
  if (options.check_option("basis_generator"))
  {
    // get the basis generator, usually mode space solver
    std::string basis_generator_name = options.get_option("basis_generator",string(""));
    Simulation* basis_generator = find_simulation(basis_generator_name);

    // get the basis set as a vector of vector
    const std::string temp_string = std::string("basis_functions");
    const std::vector<NemoMeshPoint>* momentum_point = NULL; // not used now
    std::vector<std::vector<std::complex<double> > >* basis_set;
    basis_generator->get_data(temp_string, momentum_point, basis_set);

    // convert the vector of vector into a matrix
    unsigned int number_of_eigenvalues = basis_set->size();
    unsigned int number_of_vectors_size = (*basis_set)[0].size();
    PetscMatrixParallelComplex* transformation_matrix =
        new PetscMatrixParallelComplex(number_of_vectors_size,number_of_eigenvalues,get_simulation_domain()->get_communicator());
    transformation_matrix->set_num_owned_rows(number_of_vectors_size);//single cpu
    for(unsigned int i=0; i<number_of_vectors_size; i++)
      transformation_matrix->set_num_nonzeros(i,number_of_eigenvalues,0);
    transformation_matrix->consider_as_full();
    transformation_matrix->allocate_memory();
    transformation_matrix->set_to_zero();

    std::complex<double> temp_val(0.0,0.0);
    for(unsigned int i=0; i<number_of_eigenvalues; i++)
      for(unsigned int j=0; j<number_of_vectors_size; j++)
      {
        temp_val=(*basis_set)[i][j];
        if(abs(temp_val)>1e-8)
          transformation_matrix->set(j,i,temp_val);
      }

    transformation_matrix->assemble();

    // temp_transformed_hamiltonian = hamiltonian * transformation_matrix
    PetscMatrixParallelComplex* temp_transformed_hamiltonian = NULL;
    PetscMatrixParallelComplex::mult(*hamiltonian, *transformation_matrix, &temp_transformed_hamiltonian);

    // hermitian transpose of the transformation matrix
    PetscMatrixParallelComplex* transformation_matrix_dagger = NULL;
    transformation_matrix_dagger = new PetscMatrixParallelComplex(transformation_matrix->get_num_cols(), transformation_matrix->get_num_rows(),
        get_simulation_domain()->get_communicator());
    transformation_matrix_dagger->consider_as_full();
    transformation_matrix_dagger->allocate_memory();
    transformation_matrix->hermitian_transpose_matrix(*transformation_matrix_dagger, MAT_INITIAL_MATRIX);

    // transformed = transformation_matrix_dagger * temp_transformed
    PetscMatrixParallelComplex::mult(*transformation_matrix_dagger, *temp_transformed_hamiltonian, &transformed_hamiltonian);
    delete transformation_matrix;
    transformation_matrix = NULL;
    delete transformation_matrix_dagger;
    transformation_matrix_dagger = NULL;
    delete temp_transformed_hamiltonian;
    temp_transformed_hamiltonian = NULL;

    solver->set_matrix(transformed_hamiltonian);
  }
  else
  {
    solver->set_matrix(hamiltonian);
  }
  // Jun Huang: the mode space overlap matrix is not implemented so far
  if (overlap != NULL)
    solver->set_B_matrix(overlap);

  //  hamiltonian->save_to_matlab_file("hamiltonian.m");
  //  overlap->save_to_matlab_file("overlap.m");

  double t2 = NemoUtils::get_time();
  msg.print_message(MSG_LEVEL_2, "[Schroedinger(\""+get_name()+"\")] solving...");
  solver->solve();

  double t3 = NemoUtils::get_time();
  msg.print_message2(MSG_LEVEL_3, "    needed %.2gs  for assembly, %.2gs for solution", t2-t1, t3-t2);
  msg.print_message(MSG_LEVEL_3, "    post-processing...");
  unsigned int nconv = solver->get_num_found_eigenvalues();
  if (nconv!=EnergyV.size())
    EnergyV.resize(nconv); // can happen, when the user does not include full number of eigenvalues

  const vector<cplx>* evals = solver->get_eigenvalues();
  NEMO_ASSERT(evals->size()>=nconv, "[Schroedinger(\""+get_name()+"\")] evals.size()>=nconv failed.");


  for (unsigned int i=0; i<nconv; i++)
    EnergyV[i] = (*evals)[i].real();

  //those maps will dissapear
  {
    energy_to_vector_map.resize(nconv);
    vector_to_energy_map.resize(nconv);


    for (unsigned int i=0; i<nconv; i++)
    {
      energy_to_vector_map[i] = i;
      vector_to_energy_map[i] = i;
    }
  }

  if(transformed_hamiltonian != NULL)
  {
    delete transformed_hamiltonian;
    transformed_hamiltonian = NULL;
  }
  //<m.p.>
  //we do not need to order anything
  //eigensolver must give ordered eigenvalues
  /*
  {
    //ordering eigenvalues and eigenvectors
    std::multiset<EV,compareEV> numbered_EV;
    for (unsigned int i=0;i<nconv;i++)
    {
      EV temp;
      temp.eigen_value=((*evals)[i].real());
      temp.index=i;
      numbered_EV.insert(temp);
    }

    energy_to_vector_map.resize(nconv);
    vector_to_energy_map.resize(nconv);
    std::multiset<EV,compareEV>::const_iterator set_it=numbered_EV.begin();
    unsigned int is=0;
    for (;set_it!=numbered_EV.end();++set_it)
    {
      vector_to_energy_map[is]=(*set_it).index;
      energy_to_vector_map[(*set_it).index]=is;
      ++is;
    }





    std::multiset<double> ordered_eigenvalues;
    for (unsigned int i=0;i<nconv;i++)
    {
      ordered_eigenvalues.insert((*evals)[i].real());
    }

    unsigned int i = 0;

    if(has_output("DOS_dfermi"))
      DOS_times_fermi_derivative();
    if(has_output("EmEf_DOS_dfermi"))
      EmEf_DOS_times_fermi_derivative();

    if (msg.get_level()>4) msg << "[Schroedinger] found ";
    for (std::multiset<double>::iterator it=ordered_eigenvalues.begin();it!= ordered_eigenvalues.end(); ++it)
    {
      EnergyV[i]=*it;
      ++i;
      if (msg.get_level()>4) msg << *it << "  ";
    }
    if (msg.get_level()>4) msg << "\n";


    //now the eigenvalues must be ordered
    {
      unsigned int n = EnergyV.size();
      for (int i = 0; i < n ; i++)
      {

        if (energy_to_vector_map[i] != i)
          throw runtime_error("[Schroedinger]: energy_to_vector map\n");


        if (vector_to_energy_map[i] != i)
          throw runtime_error("[Schroedinger]: vector_to_energy map\n");


        if (i > 0)
          if (EnergyV[i] <  EnergyV[i-1])
            throw runtime_error("[Schroedinger]: EnergyV is not ordered\n");


      }


      if (msg.get_level() > 4)
        msg << "check ordering passed ok\n";

    }
  }
   */
  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::solve_tb_single_kpoint(const vector <double>& k_point, const string& name_of_outputfile)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::solve_tb_single_kpoint1";
  NemoUtils::tic(tic_toc_prefix);

  k_vector=k_point;

  if(!combine_H)
    this->assemble_hamiltonian();
  else
    combine_hamiltonian();

  solver->solve();

  int nconv = solver->get_num_found_eigenvalues();

  const vector<cplx>* evals = solver->get_eigenvalues();

  std::multiset<double> ordered_eigenvalues;
  std::multiset<double>::iterator it = ordered_eigenvalues.begin();
  for (int i=0; i<nconv; i++)
  {
    NEMO_ASSERT(std::abs((*evals)[i].imag())<1e-12,"[Schroedinger(\""+get_name()+"\")] solve_tb_single_kpoint Complex eigenvalue \n");
    ordered_eigenvalues.insert((*evals)[i].real());
  }

  // ----------------------------------------
  // open tb_eigenvalues.dat in append mode
  // create the file, if it does not exist
  // ----------------------------------------
  ofstream output_file;
  output_file.open(name_of_outputfile.c_str(),std::ios_base::out | std::ios_base::app);
  for (it=ordered_eigenvalues.begin(); it  != ordered_eigenvalues.end(); ++it)
    output_file << *it << "\t";
  output_file << "\n";
  output_file.close();

  NemoUtils::toc(tic_toc_prefix);
}



/*// Calculate parts of the bandstructure according to user input
void Schroedinger::get_dispersion(const vector <vector <double> > & k_pointsV,
                                  const vector <unsigned int> & number_of_nodesV,
                                  const unsigned int number_of_eigenvalues,
                                  vector <vector <double> > & EnergyV,
                                  vector <vector <double> > & k_vectorsV)
{


  if (number_of_nodesV.size()<k_pointsV.size()-1)
    throw invalid_argument("Schroedinger: vector number_of_nodesV has wrong size\n");

  //how many k-points are we going to calculates and store:
  unsigned int total_points=0;
  for (unsigned int i=0;i<k_pointsV.size()-1;i++)
    total_points+=number_of_nodesV[i];
  vector <double> temp_energyV (number_of_eigenvalues);
  EnergyV.resize(total_points, temp_energyV);
  vector<double> tempV(3);
  k_vectorsV.resize(total_points,tempV);

  full_energy_to_vector_map.clear();
  full_energy_to_vector_map.resize(k_pointsV.size());
  for (unsigned int i=1;i<k_pointsV.size();i++) // loop over the user-given k-points
  {
    if (number_of_nodesV[i-1]<2)
      throw invalid_argument("Schroedinger: too small number of nodes\n");


    for (unsigned int ii=0;ii<number_of_nodesV[i-1];ii++) //loop over the k-points in between two user-given ones
    {
      tempV[0]=k_pointsV[i-1][0]+(k_pointsV[i][0]-k_pointsV[i-1][0])*1.0*ii/(1.0*number_of_nodesV[i-1]-1.0);
      tempV[1]=k_pointsV[i-1][1]+(k_pointsV[i][1]-k_pointsV[i-1][1])*1.0*ii/(1.0*number_of_nodesV[i-1]-1.0);
      tempV[2]=k_pointsV[i-1][2]+(k_pointsV[i][2]-k_pointsV[i-1][2])*1.0*ii/(1.0*number_of_nodesV[i-1]-1.0);
      solve_tb_single_kpoint(tempV,EnergyV[ii]);
      k_vectorsV[ii][0]=tempV[0];
      k_vectorsV[ii][1]=tempV[1];
      k_vectorsV[ii][2]=tempV[2];
    }
  }
}*/

void Schroedinger::DOS_times_fermi_derivative(void)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::DOS_times_fermi_derivative";
  NemoUtils::tic(tic_toc_prefix);


  //sample the energy homogeneously between given maximum and minimum
  double min_energy=options.get_option("DOS_min_energy",-20.0);
  double max_energy=options.get_option("DOS_max_energy",20.0);
  int number_of_points=options.get_option("DOS_points",100);
  //std::map<double, double> result; //to be replaced by reference to an externally declared map
  double temperature=options.get_option("electron_temperature",NemoPhys::temperature);//[K]
  temperature*=NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
  double chem_pot=options.get_option("electron_chem_pot",0.0);//[eV]
  const std::vector<std::complex<double> >* eigenvals=solver->get_eigenvalues();
  unsigned int num_evals=eigenvals->size();


  //initialize the result map (if required)
  if(DOS_times_dfermi.size()<1)
  {
    for(unsigned int i=0; (int)i<number_of_points; i++)
    {
      double energy = (max_energy-min_energy)/(number_of_points-1.0)*i+min_energy;
      DOS_times_dfermi[energy]=0.0;
    }
  }
  std::map<double,double>::iterator it_low, it_up;
  for(unsigned int i=0; i<num_evals; i++)
  {
    double temp_eigenval=(*eigenvals)[i].real();
    double temp_result=-NemoMath::dfermi_distribution_over_dE(chem_pot, temperature, temp_eigenval);
    it_low=DOS_times_dfermi.lower_bound(temp_eigenval);
    it_up=DOS_times_dfermi.upper_bound(temp_eigenval);
    if(it_up==DOS_times_dfermi.end())
      it_up--;
    std::map<double,double>::iterator temp_it;
    if(abs(temp_eigenval-it_low->first)<abs(temp_eigenval-it_up->first))
      temp_it=it_low;
    else
      temp_it=it_up;

    //std::map<double,double>::const_iterator neighbor_it1,neighbor_it2;
    //neighbor_it1=temp_it;
    //if(neighbor_it1!=DOS_times_dfermi.begin())
    //  neighbor_it1--;
    //neighbor_it2=temp_it;
    //++neighbor_it2;
    //if(neighbor_it2==DOS_times_dfermi.end())
    //  neighbor_it2--;
    //double energy_interval=(neighbor_it2->first-neighbor_it1->first)/2;//(c_it2->first+temp_it->first)/2-(temp_it->first+c_it1->first)/2
    //double energy_interval=(max_energy-min_energy)/(number_of_points-1.0);
    //The sum runs over momenta
    temp_it->second+=temp_result; //*energy_interval;
  }

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::EmEf_DOS_times_fermi_derivative(void)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::EmEf_DOS_times_fermi_derivative";
  NemoUtils::tic(tic_toc_prefix);

  //sample the energy homogeneously between given maximum and minimum
  double min_energy=options.get_option("DOS_min_energy",-20.0);
  double max_energy=options.get_option("DOS_max_energy",20.0);
  int number_of_points=options.get_option("DOS_points",100);
  //std::map<double, double> result; //to be replaced by reference to an externally declared map
  double temperature=options.get_option("electron_temperature",NemoPhys::temperature);//[K]
  temperature*=NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
  double chem_pot=options.get_option("electron_chem_pot",0.0);//[eV]
  const std::vector<std::complex<double> >* eigenvals=solver->get_eigenvalues();
  unsigned int num_evals=eigenvals->size();
  //  const vector<vector<complex<double> > >* eigenvectors=solver.get_eigenvectors();

  //initialize the result map (if required)
  if(EmEf_DOS_times_dfermi.size()<1)
  {
    for(unsigned int i=0; (int)i<number_of_points; i++)
    {
      double energy = (max_energy-min_energy)/(number_of_points-1.0)*i+min_energy;
      EmEf_DOS_times_dfermi[energy]=0.0;
    }
  }
  std::map<double,double>::iterator it_low, it_up;
  for(unsigned int i=0; i<num_evals; i++)
  {
    double temp_eigenval=(*eigenvals)[i].real();
    double temp_result=-(temp_eigenval-chem_pot)*NemoMath::dfermi_distribution_over_dE(chem_pot, temperature, temp_eigenval);
    it_low=EmEf_DOS_times_dfermi.lower_bound(temp_eigenval);
    it_up=EmEf_DOS_times_dfermi.upper_bound(temp_eigenval);
    if(it_up==DOS_times_dfermi.end())
      it_up--;
    std::map<double,double>::iterator temp_it;
    if(abs(temp_eigenval-it_low->first)<abs(temp_eigenval-it_up->first))
      temp_it=it_low;
    else
      temp_it=it_up;

    //std::map<double,double>::const_iterator neighbor_it1,neighbor_it2;
    //neighbor_it1=temp_it;
    //if(neighbor_it1!=DOS_times_dfermi.begin())
    //  neighbor_it1--;
    //neighbor_it2=temp_it;
    //++neighbor_it2;
    //if(neighbor_it2==DOS_times_dfermi.end())
    //  neighbor_it2--;
    //double energy_interval=(neighbor_it2->first-neighbor_it1->first)/2;//(c_it2->first+temp_it->first)/2-(temp_it->first+c_it1->first)/2
    //double energy_interval=(max_energy-min_energy)/(number_of_points-1.0);
    //The sum runs over momenta
    temp_it->second+=temp_result; //*energy_interval;
  }

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_position_operator(const Domain* /*domain*/, std::string direction, PetscMatrixParallelComplex*& result_position_operator)
{
  assemble_position_operator(direction, result_position_operator);
}


// Hesam: Get spatial threshold or charge sign
void Schroedinger::get_data(const string& variable,const double& Energy, map<unsigned int, double>& result_map, map<unsigned int, double>& result_map_right)
{
  //Bozidar added
  MsgLevel msg_level = msg.get_level();
  msg.set_level(MsgLevel(3));

  // Hesam: Spatial sign according to spatial threshold
  std::string error_prefix =  "Schroedinger(\""+get_name()+"\")::get_data ";
  // variable=="spatial_charge_sign" || || variable=="hole_factor_dof_right_left"
  std::string Electron_Hole_Heuristics = options.get_option("electron_hole_heuristics",string("sharp"));
  msg << "Heuristics = " << Electron_Hole_Heuristics << "\n";

  //Bozidar added
  msg.set_level(msg_level);

  if(variable=="spatial_threshold" || variable=="spatial_threshold_dof" || variable=="hole_factor_dof")
  {
    if(&result_map!=NULL)
      if(result_map.size()>0)
        result_map.clear();
    if(&result_map_right!=NULL)
      if(result_map_right.size()>0)
        result_map_right.clear();
    //1. loop over all atoms
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    DOFmapInterface&                dof_map = get_dof_map();

    //m//    std::map<int,int> atom_id_to_index_map; //this map stores which atom_id corresponds to which iterator number (.begin()==0)
    //m//dof_map.build_atom_id_to_local_atom_index_map(&atom_id_to_index_map);

    const AtomicStructure& atoms   = domain->get_atoms();
    //   const map< unsigned int,AtomStructNode >& lattice = atoms.lattice();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    unsigned int number_of_atoms=0;
    for(; it!=end; ++it)
      number_of_atoms++;
    //allocate temp memory for result
    /// vector<double> result(number_of_atoms,0.0);
    /// map<unsigned int, bool> result(number_of_atoms,0.0);

    // Vector which stores the threshold
    vector<double> data_threshold;
    this->set_up_threshold("spatial_threshold", data_threshold);

    it = atoms.active_atoms_begin();
    int counter = 0;
    double Source_Threshold = data_threshold[0];
    double Drain_Threshold  = data_threshold[data_threshold.size()-1];
    double Hole_Factor_From_Left, Hole_Factor_From_Right;
    for ( ; it != end; ++it)
    {
      const AtomStructNode& nd        = it.node();
      const Atom*           atom      = nd.atom;
      const unsigned int    atom_id   = it.id();
      const Material*       material  = atom->get_material();
      HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*>(this->get_material_properties(material));
      double Spatial_threshold;
      double Ec = tb_ham->get_conduction_band_edge();
      double Ev = tb_ham->get_valence_band_edge();
      double Eg_Half = (Ec - Ev)/2.0;

      //3. get the Spatial_threshold for the atom indices
      ////TBSchroedinger* temp_schroedinger = dynamic_cast<TBSchroedinger*>(this);
      ////NEMO_ASSERT(temp_schroedinger!=NULL,error_prefix+"cast of this into TBSchroedinger failed\n");
      ////temp_schroedinger->get_data(std::string("spatial_threshold"),atom_id,&Spatial_threshold);
      Spatial_threshold = data_threshold[counter];
      //4. Set the atom index and spatial threshold
      if(variable=="spatial_threshold")
        result_map[atom_id] = Spatial_threshold;
      /*
      if(variable=="spatial_charge_sign")
      {
      if(Energy > Eg_Half + Spatial_threshold) result_map[atom_id] = -1.0;
      else if(Energy < -Eg_Half + Spatial_threshold) result_map[atom_id] = +1.0;
      else result_map[atom_id] = 2.0*(Spatial_threshold - Energy)/(Ec-Ev);
      }
       */
      if(variable=="spatial_threshold_dof")
      {
        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
        int n_orbital = tb_ham->get_number_of_orbitals(atom);
        // Loop over each atoms orbitals and set the threshold for each one
        for (unsigned short orb_i=0; orb_i<n_orbital; orb_i++)
        {
          map<short, unsigned int>::const_iterator it_dofs = atom_dofs->find(orb_i);
          unsigned int atom_id_dof = it_dofs->second;
          NEMO_ASSERT( it_dofs != atom_dofs->end(),error_prefix+
              "[Schroedinger(\""+get_name()+"\")] (Spatial Threshold): first orbital index reached end of orbitals\n");
          result_map[atom_id_dof] = Spatial_threshold;
        }
      }
      if((Electron_Hole_Heuristics == "omen_smooth") && (variable=="hole_factor_dof"))
      {
        // Left
        if(Energy <= Source_Threshold) // Hole injected at source
        {
          if(Energy <= Spatial_threshold) // is still Hole
            Hole_Factor_From_Left = +1.0;
          else if((Energy > Spatial_threshold) && (Energy <= Spatial_threshold+Eg_Half)) // transition
            Hole_Factor_From_Left = (Spatial_threshold+Eg_Half-Energy)/Eg_Half;
          else // is electron
            Hole_Factor_From_Left = 0.0;
        }
        else
        {
          if(Energy > Spatial_threshold) // electron
            Hole_Factor_From_Left = +0.0; // electron
          else if((Energy <= Spatial_threshold) && (Energy > Spatial_threshold-Eg_Half)) // transition
            Hole_Factor_From_Left = (Spatial_threshold-Energy)/Eg_Half;
          else // is hole
            Hole_Factor_From_Left = 1.0;
        }
        // Right
        if(Energy <= Drain_Threshold) // Hole injected at source
        {
          if(Energy <= Spatial_threshold) // is still Hole
            Hole_Factor_From_Right = +1.0;
          else if((Energy > Spatial_threshold) && (Energy <= Spatial_threshold+Eg_Half)) // transition
            Hole_Factor_From_Right = (Spatial_threshold+Eg_Half-Energy)/Eg_Half;
          else // is electron
            Hole_Factor_From_Right = 0.0;
        }
        else
        {
          if(Energy > Spatial_threshold) // electron
            Hole_Factor_From_Right = +0.0; // electron
          else if((Energy <= Spatial_threshold) && (Energy > Spatial_threshold-Eg_Half)) // transition
            Hole_Factor_From_Right = (Spatial_threshold-Energy)/Eg_Half;
          else // is hole
            Hole_Factor_From_Right = 1.0;
        }


        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
        int n_orbital = tb_ham->get_number_of_orbitals(atom);
        // Loop over each atoms orbitals and set the threshold for each one
        for (unsigned short orb_i=0; orb_i<n_orbital; orb_i++)
        {
          map<short, unsigned int>::const_iterator it_dofs = atom_dofs->find(orb_i);
          unsigned int atom_id_dof = it_dofs->second;
          NEMO_ASSERT( it_dofs != atom_dofs->end(),error_prefix+
              "[Schroedinger(\""+get_name()+"\")] Spatial Threshold): first orbital index reached end of orbitals\n");
          result_map[atom_id_dof] = Hole_Factor_From_Left;
          result_map_right[atom_id_dof] = Hole_Factor_From_Right;
        }

      }
      else if((Electron_Hole_Heuristics == "omen_smooth_scattering") && (variable=="hole_factor_dof"))
      {
        //data_threshold is Eg/2 + Vmiddle
        double scatt_threshold = Spatial_threshold;

        double hole_factor = 0.0;
        if(Energy <= scatt_threshold - Eg_Half/2.0) //is a hole
          hole_factor = 1.0;
        //hole_fact
        else if ((Energy > scatt_threshold-Eg_Half/2.0) && (Energy < scatt_threshold+Eg_Half/2.0) ) //transition
          hole_factor = (scatt_threshold + Eg_Half/2.0 - Energy)/Eg_Half;

        else
          hole_factor = 0.0;

        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
        int n_orbital = tb_ham->get_number_of_orbitals(atom);
        // Loop over each atoms orbitals and set the threshold for each one
        for (unsigned short orb_i = 0; orb_i < n_orbital; orb_i++)
        {
          map<short, unsigned int>::const_iterator it_dofs = atom_dofs->find(orb_i);
          unsigned int atom_id_dof = it_dofs->second;
          NEMO_ASSERT(it_dofs != atom_dofs->end(),
              error_prefix + "[Schroedinger(\"" + get_name()
              + "\")] Spatial Threshold): first orbital index reached end of orbitals\n");
          result_map[atom_id_dof] = hole_factor;
          result_map_right[atom_id_dof] = hole_factor;
        }
      }
      else if((Electron_Hole_Heuristics == "sharp") && (variable=="hole_factor_dof"))
      {
        // Left
        if(Energy <= Spatial_threshold) // is still Hole
          Hole_Factor_From_Left = +1.0;
        else // is electron
          Hole_Factor_From_Left = 0.0;

        // Right
        Hole_Factor_From_Right = Hole_Factor_From_Left;

        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
        int n_orbital = tb_ham->get_number_of_orbitals(atom);
        // Loop over each atoms orbitals and set the threshold for each one
        for (unsigned short orb_i=0; orb_i<n_orbital; orb_i++)
        {
          map<short, unsigned int>::const_iterator it_dofs = atom_dofs->find(orb_i);
          unsigned int atom_id_dof = it_dofs->second;
          NEMO_ASSERT( it_dofs != atom_dofs->end(),error_prefix+
              "[Schroedinger(\""+get_name()+"\")] (Spatial Threshold): first orbital index reached end of orbitals\n");
          result_map[atom_id_dof] = Hole_Factor_From_Left;
          result_map_right[atom_id_dof] = Hole_Factor_From_Right;
        }

      }

      counter++;
    }
  }
  else
    throw std::runtime_error(error_prefix+"called with unknown variable\n");
}

void Schroedinger::get_data(const std::string& variable, const double& input_double, std::vector<double>& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data";
  NemoUtils::tic(tic_toc_prefix);
  data.clear();
  if(variable=="averaged_hole_factor")
  {
    std::string temp_simulation_name=options.get_option("hole_factor_solver",get_name());
    Simulation* temp_simulation=find_simulation(temp_simulation_name);
    NEMO_ASSERT(temp_simulation!=NULL,tic_toc_prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
    std::map<unsigned int, double> threshold_map;
    std::map<unsigned int, double> threshold_map_right;
    temp_simulation->get_data(std::string("hole_factor_dof"),input_double, threshold_map, threshold_map_right);
    data.resize(2,0.0);
    std::map<unsigned int, double>::const_iterator temp_cit=threshold_map.begin();
    for(; temp_cit!=threshold_map.end(); ++temp_cit)
      data[0]+=temp_cit->second/threshold_map.size();
    temp_cit=threshold_map_right.begin();
    for(; temp_cit!=threshold_map_right.end(); ++temp_cit)
      data[1]+=temp_cit->second/threshold_map_right.size();
  }
  else if(variable=="boundary_hole_factors")
  {
    //1. get the spatial hole factor
    std::string temp_simulation_name=options.get_option("hole_factor_solver",get_name());
    Simulation* temp_simulation=find_simulation(temp_simulation_name);
    NEMO_ASSERT(temp_simulation!=NULL,tic_toc_prefix+"have not found simulation \""+temp_simulation_name+"\"\n");
    std::map<unsigned int, double> threshold_map;
    std::map<unsigned int, double> threshold_map_right;
    temp_simulation->get_data(std::string("hole_factor_dof"),input_double, threshold_map, threshold_map_right);
    data.resize(2,0.0);
    //2. find the atom_id's of those atoms coupled to one of the two lead domains (check that there are two lead domains)
    const std::vector<const Domain*>& all_leads=get_const_simulation_domain()->get_all_leads();
    const Domain* source_domain_or_left = NULL;
    const Domain* drain_domain_or_right = NULL;
    if(options.check_option("left_lead_domain")&&options.check_option("right_lead_domain"))
    {
      std::string left_name=options.get_option("left_lead_domain",std::string(""));
      source_domain_or_left=Domain::get_domain(left_name);
      NEMO_ASSERT(source_domain_or_left!=NULL,tic_toc_prefix+"have not found \""+left_name+"\"\n");
      std::string right_name=options.get_option("right_lead_domain",std::string(""));
      drain_domain_or_right=Domain::get_domain(right_name);
      NEMO_ASSERT(source_domain_or_left!=NULL,tic_toc_prefix+"have not found \""+right_name+"\"\n");
    }
    else
    {
      NEMO_ASSERT(all_leads.size()==2,tic_toc_prefix+"found more or less than 2 leads\n");
      for(unsigned int i=0; i<all_leads.size(); i++)
      {
        std::string temp_name=all_leads[i]->get_name();
        if(temp_name.find(std::string("source"))!=std::string::npos||temp_name.find(std::string("Source"))!=std::string::npos)
          source_domain_or_left=all_leads[i];
        else
          drain_domain_or_right=all_leads[i];
      }
    }
    NEMO_ASSERT(source_domain_or_left!=NULL&&drain_domain_or_right!=NULL,tic_toc_prefix+"one domain is not found\n");
    const AtomisticDomain* temp_domain=dynamic_cast<const AtomisticDomain*>(get_const_simulation_domain());
    NEMO_ASSERT(temp_domain!=NULL,tic_toc_prefix+"cast of simulation domain into AtomisticDomain failed\n");
    std::set<unsigned int> atoms_coupled_to_source=temp_domain->get_atoms_coupled_to_lead(source_domain_or_left);
    std::set<unsigned int> atoms_coupled_to_drain=temp_domain->get_atoms_coupled_to_lead(drain_domain_or_right);
    NEMO_ASSERT(atoms_coupled_to_source.size()>0,tic_toc_prefix+"no atom coupled to the source\n");
    NEMO_ASSERT(atoms_coupled_to_drain.size()>0,tic_toc_prefix+"no atom coupled to the drain\n");
    //3. store the appropriate hole factors in data
    //3.1 average over the atoms coupled to source and drain
    //source
    std::set<unsigned int>::iterator source_it=atoms_coupled_to_source.begin();
    for(; source_it!=atoms_coupled_to_source.end(); ++source_it)
    {
      //debug:SS
      cout << *source_it << "\n";
      std::map<unsigned int, double>::const_iterator temp_it=threshold_map.find(*source_it);
      NEMO_ASSERT(temp_it!=threshold_map.end(),tic_toc_prefix+"have not found atom coupled to the source\n");
      data[0]+=temp_it->second;
    }
    data[0]/=atoms_coupled_to_source.size();
    //drain
    std::set<unsigned int>::iterator drain_it=atoms_coupled_to_drain.begin();
    for(; drain_it!=atoms_coupled_to_drain.end(); ++drain_it)
    {
      std::map<unsigned int, double>::const_iterator temp_it=threshold_map.find(*drain_it);
      NEMO_ASSERT(temp_it!=threshold_map_right.end(),tic_toc_prefix+"have not found atom coupled to the drain\n");
      data[1]+=temp_it->second;
    }
    data[1]/=atoms_coupled_to_drain.size();
  }
  else if(variable=="averaged_threshold")
  {
    throw std::runtime_error(tic_toc_prefix+"not implemented yet\n");
  }
  else
    throw std::invalid_argument(tic_toc_prefix+"called with unknown variable \""+variable+"\"\n");


  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::get_data(const string& variable,const vector<unsigned int>& index,vector<double>& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data";
  NemoUtils::tic(tic_toc_prefix);


  std::string error_prefix =  "Schroedinger(\""+get_name()+"\")::get_data ";
  if (variable=="electron_density")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      data[i] = this->get_electron_density(index[i]);
    }
  }
  else if (variable=="hole_density")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      data[i] = this->get_hole_density(index[i]);
    }
  }
  else if (variable=="derivative_electron_density_over_potential")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      data[i] = this->get_derivative_density_over_potential(index[i],"electron");
    }
  }
  else if (variable=="derivative_hole_density_over_potential")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      data[i] = this->get_derivative_density_over_potential(index[i],"hole");
    }
  }
  else if(variable=="derivative_total_charge_density_over_potential")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    if(!hole_density)
    {
      for(unsigned int i=0; i<size_of_index; i++)
      {
        data[i] = -this->get_derivative_density_over_potential(index[i],"electron");
      }
    }
    else
    {
      for(unsigned int i=0; i<size_of_index; i++)
      {
        data[i] = -this->get_derivative_density_over_potential(index[i],"electron") + this->get_derivative_density_over_potential(index[i],"hole");
      }
    }
  }
  else if (variable=="ion_density")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      data[i] = this->get_ion_density(index[i]);
    }
  }
  else if (variable=="free_charge")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    for(unsigned int i=0; i<size_of_index; i++)
    {
      if (electron_density&&hole_density)
        data[i] =  this->get_hole_density(index[i]) - this->get_electron_density(index[i]);
      else if (electron_density)
        data[i] = -this->get_electron_density(index[i]);
      else if (hole_density)
        data[i] =  this->get_hole_density(index[i]);
      else
        data[i]=0;
    }
  }
  else if (variable=="total_density")
  {
    unsigned int size_of_index=index.size();
    data.resize(size_of_index);
    if (!hole_density)
    {
      for(unsigned int i=0; i<size_of_index; i++)
      {
        data[i] = this->get_ion_density(index[i]) - this->get_electron_density(index[i]);
      }
    }
    else
    {
      for(unsigned int i=0; i<size_of_index; i++)
      {
        data[i] = this->get_ion_density(index[i]) - this->get_electron_density(index[i]) + this->get_hole_density(index[i]);
      }
    }
  }
  else if(variable=="effective_mass")
  {
    if(&data!=NULL)
      if(data.size()>0)
        data.clear();
    //1. loop over all atoms
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    DOFmapInterface&                dof_map = get_dof_map();

    std::map<int,int> atom_id_to_index_map; //this map stores which atom_id corresponds to which iterator number (.begin()==0)
    dof_map.build_atom_id_to_local_atom_index_map(&atom_id_to_index_map);

    const AtomicStructure& atoms   = domain->get_atoms();
    //    const map< unsigned int,AtomStructNode >& lattice = atoms.lattice();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    unsigned int number_of_atoms=0;
    for(; it!=end; ++it)
      number_of_atoms++;
    //allocate temp memory for result
    vector<double> result(number_of_atoms,0.0);
    it = atoms.active_atoms_begin();
    for ( ; it != end; ++it)
    {
      //const AtomStructNode& nd        = it.node();
      //      const Atom*           atom      = nd.atom;
      const unsigned int    atom_id   = it.id();

      //2. translate atom_id into local index (as done for the basis functions)
      std::map<int,int>::const_iterator c_it=atom_id_to_index_map.find(atom_id);
      NEMO_ASSERT(c_it!=atom_id_to_index_map.end(),error_prefix+"have not found atom_id_to_index_map for given atom_id\n");
      const int translated_atom_id = c_it->second;
      NEMO_ASSERT(translated_atom_id<(int)result.size(),error_prefix+"translated index of atom_id is beyond vector boundaries\n");

      //3. get the effective masses for the atom indices
      double eff_mass;
      TBSchroedinger* temp_schroedinger = dynamic_cast<TBSchroedinger*>(this);
      NEMO_ASSERT(temp_schroedinger!=NULL,error_prefix+"cast of this into TBSchroedinger failed\n");
      temp_schroedinger->get_data(std::string("effective_mass"),atom_id,&eff_mass);
      //4. order the effective masses towards the translated indices
      result[translated_atom_id]=eff_mass;
    }
    //5. store result into data
    data=result;
  }
  else if (variable=="conduction_band")
  {
    this->set_up_bandedge("conduction_band", data);  // second get_data() argument is not used
  }
  else if (variable=="valence_band")
  {
    this->set_up_bandedge("valence_band", data);    // second get_data() argument is not used
  }
  else if (variable=="cb_threshold")
  {
    this->set_up_threshold("conduction_band", data); // second get_data() argument is not used
  }
  else if (variable=="vb_threshold")
  {
    this->set_up_threshold("valence_band", data);  // second get_data() argument is not used
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::get_data(const string& variable, vector<vector<cplx > >& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data1";
  NemoUtils::tic(tic_toc_prefix);


  std::string error_prefix = "Schroedinger(\""+get_name()+"\")::get_data: ";
  if (variable=="eigenfunctions")
  {
    unsigned int number_of_eigenvalues=solver->get_num_found_eigenvalues();
    data.resize(number_of_eigenvalues);
    for (unsigned int i=0; i<number_of_eigenvalues; i++)
    {
      //this->get_eigenfunction(i, data[i]); unordered output
      this->get_eigenfunction(energy_to_vector_map[i],data[i]);
    }
  }
  else if(variable=="subset_of_eigenvectors")
  {
    data.clear();
    //data.resize(subset_of_eigenvectors.size());

    //Gather the size of subset_of_momenta from all MPI processes
    int number_of_processes;
    MPI_Comm_size(k_space_communicator,&number_of_processes);
    std::vector<int> subset_of_eigenvectors_found_per_CPU(number_of_processes,0);
    int send_size=subset_of_eigenvectors.size();
    MPI_Allgather(&send_size, 1, MPI_INT, &(subset_of_eigenvectors_found_per_CPU[0]),1, MPI_INT, k_space_communicator);
    std::vector<int> displs(number_of_processes, 0);

    unsigned int dimension_of_hamiltonian=hamiltonian->get_num_cols();

    for(int i = 1; i < number_of_processes; i++)
      displs[i] = displs[i-1] + subset_of_eigenvectors_found_per_CPU[i-1]*dimension_of_hamiltonian;

    unsigned int total_number_of_found_eigenvectors=0;
    for(int i=0; i<number_of_processes; i++)
      total_number_of_found_eigenvectors+=subset_of_eigenvectors_found_per_CPU[i];

    data.resize(total_number_of_found_eigenvectors,std::vector<std::complex<double> >(dimension_of_hamiltonian,0.0));

    //call Allgatherv for each momentum direction individually
    std::vector<std::complex<double> > out_subset_of_eigenvectors(subset_of_eigenvectors.size(),0.0);
    std::vector<std::complex<double> > in_subset_of_eigenvectors(total_number_of_found_eigenvectors,0.0);
    for(unsigned int i=0; i<dimension_of_hamiltonian; i++)
    {
      for(unsigned int j=0; j<subset_of_eigenvectors.size(); j++)
        out_subset_of_eigenvectors[j]=subset_of_eigenvectors[j][i];

      //replace subset_of_eigenvectors_found_per_CPU with subset_of_eigenvectors_found_per_CPU*dimension_of_hamitonian
      MPI_Allgatherv(&(out_subset_of_eigenvectors[0]), subset_of_eigenvectors.size(), MPI_DOUBLE_COMPLEX,
          &(in_subset_of_eigenvectors[0]), &(subset_of_eigenvectors_found_per_CPU[0]),&(displs[0]),MPI_DOUBLE_COMPLEX,k_space_communicator);

      for(unsigned int j=0; j<total_number_of_found_eigenvectors; j++)
        data[j][i]=in_subset_of_eigenvectors[j];
    }

  }
  else if(variable=="basis_functions")
  {
    if(&data!=NULL)
      if(data.size()>0)
        data.clear();
    //1. loop over all atoms
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    DOFmapInterface&                dof_map = get_dof_map();

    std::map<int,int> atom_id_to_index_map; //this map stores which atom_id corresponds to which iterator number (.begin()==0)
    dof_map.build_atom_id_to_local_atom_index_map(&atom_id_to_index_map);

    const AtomicStructure& atoms   = domain->get_atoms();
    //    const map< unsigned int,AtomStructNode >& lattice = atoms.lattice();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    unsigned int number_of_atoms=0;
    for(; it!=end; ++it)
      number_of_atoms++;
    //allocate temp memory for result
    vector<vector<cplx > > result;

    it = atoms.active_atoms_begin();
    HamiltonConstructor* tb_ham = 0; // bulk TB model
    int old_number_of_orbitals=-1; //for checking that all atoms have the same number of orbitals...
    int counter=0;
    for ( ; it != end; ++it)
    {
      const AtomStructNode& nd        = it.node();
      const Atom*           atom      = nd.atom;
      const unsigned int    atom_id   = it.id();
      const Material*       material  = atom->get_material();
      const Crystal*        crystal   = material->get_crystal();

      const std::map<short,unsigned int>* atom_dof_map=dof_map.get_atom_dof_map(atom_id);

      tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(material));
      NEMO_ASSERT(tb_ham!=NULL, error_prefix +"dynamic cast of HamiltonConstructor failed\n");
      const unsigned int n_orbital = tb_ham->get_number_of_orbitals(atom);

      //2. get the volume that this atom is representing
      double cell_volume_per_atom = crystal->calculate_primitive_cell_volume()/crystal->get_number_of_prim_cell_atom();
      double sqrt_atomic_volume = std::sqrt(
          cell_volume_per_atom); //NOTE: if there are N orbitals per atom, they should be contracted to a atom of 4 states, i.e. no futher normalization is required
      double sqrt_periodic_space_volume=std::sqrt(domain->return_periodic_space_volume());
      if(!(domain->return_periodic_space_volume()>0.0))
        sqrt_periodic_space_volume=1.0;

      //checking that all atoms have the same number of orbitals - should become obsolete in later versions
      if(old_number_of_orbitals!=-1)
        NEMO_ASSERT(old_number_of_orbitals==(int)n_orbital,error_prefix+"inhomogeneous number of orbitals!\n");
      old_number_of_orbitals=n_orbital;

      //4. create a vector of the size = number_of_atoms*n_orbitals for each atom (each is one column of the transformation matrix)
      std::vector<std::complex<double> > temp_vector(n_orbital*number_of_atoms,(std::complex<double> (0.0,
          0.0)) ); //index of the atom, defines the column index of this vector
      //3. loop over the atomic orbitals and find the DOFmap index
      std::map<short,unsigned int>::const_iterator c_it=atom_dof_map->begin();
      //if(options.get_option("density_per_unit_volume",false))
      {
        for(; c_it!=atom_dof_map->end(); c_it++)
          temp_vector[c_it->second] = 1.0/sqrt_atomic_volume*sqrt_periodic_space_volume;
      }
      //else
      //{
      //  //for (unsigned int i=0;i<n_orbital;i++)
      //  //  temp_vector[atom_id*n_orbital+i] = 1.0;
      //  for(; c_it!=atom_dof_map->end(); c_it++)
      //    temp_vector[c_it->second] = 1.0;
      //}
      //4. if there is no spin in the Hamiltonian model, multiply the temp_vector with sqrt(2.0)
      std::string tb_kind = options.get_option("tb_basis", std::string("sp3"));
      if (tb_kind.find("SO") == std::string::npos)
        for(unsigned int i=0; i<temp_vector.size(); i++)
          temp_vector[i]*=std::sqrt(2.0);

      //5. store the result in data
      if(result.size()==0)
        result.resize(number_of_atoms,temp_vector);
      else
      {
        std::map<int,int>::const_iterator c_it=atom_id_to_index_map.find(atom_id);
        NEMO_ASSERT(c_it!=atom_id_to_index_map.end(),error_prefix+"have not found atom_id_to_index_map for given atom_id\n");
        NEMO_ASSERT(c_it->second<(int)result.size(),error_prefix+"translated index of atom_id is beyond vector boundaries\n");
        //result[counter]=temp_vector; //here we use the number of the atom iterator - to be replaced with atom_id_to_index_map of DOFmap...
        result[c_it->second]=temp_vector;
      }
      counter++;
    }
    data=result;
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::get_data(const string& variable, vector<vector<double> >& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data2";
  NemoUtils::tic(tic_toc_prefix);



  if (variable=="subset_of_momenta")
  {
    data.clear();

    //Gather the size of subset_of_momenta from all MPI processes
    int number_of_processes;
    MPI_Comm_size(k_space_communicator,&number_of_processes);
    std::vector<int> subset_of_momenta_found_per_CPU(number_of_processes,0);
    int send_size=subset_of_momenta.size();
    MPI_Allgather(&send_size, 1, MPI_INT, &(subset_of_momenta_found_per_CPU[0]),1, MPI_INT, k_space_communicator);
    std::vector<int> displs(number_of_processes, 0);
    for(int i = 1; i < number_of_processes; i++)
      displs[i] = displs[i-1] + subset_of_momenta_found_per_CPU[i-1];

    unsigned int total_number_of_found_momenta=0;
    for(unsigned int i=0; (int)i<number_of_processes; i++)
      total_number_of_found_momenta+=subset_of_momenta_found_per_CPU[i];


    data.resize(total_number_of_found_momenta,std::vector<double>(3,0.0));

    //call Allgatherv for each momentum direction individually
    std::vector<double> out_subset_of_momenta(subset_of_momenta.size(),0.0);
    std::vector<double> in_subset_of_momenta(total_number_of_found_momenta,0.0);
    for(unsigned int i=0; i<3; i++)
    {
      for(unsigned int j=0; j<subset_of_momenta.size(); j++)
        out_subset_of_momenta[j]=subset_of_momenta[j][i];

      MPI_Allgatherv(&(out_subset_of_momenta[0]), subset_of_momenta.size(), MPI_DOUBLE,
          &(in_subset_of_momenta[0]), &(subset_of_momenta_found_per_CPU[0]),&(displs[0]),MPI_DOUBLE,k_space_communicator);

      for(unsigned int j=0; j<total_number_of_found_momenta; j++)
        data[j][i]=in_subset_of_momenta[j];
    }
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_EOMMatrix(const std::vector<NemoMeshPoint>& input_momentum_tuple, const Domain* row_domain, const Domain* column_domain, const bool transfer_ownership,
    PetscMatrixParallelComplex*& output, DOFmapInterface*& neighbor_DOFmapInterface)
{
  //neighbor_DOFmapInterface=NULL; //needed to avoid deletion of dofmaps of other solvers. originally needed for EHSchroedinger
  if(row_domain==NULL&&column_domain==NULL)
    get_data(std::string("Hamiltonian"), output, input_momentum_tuple, transfer_ownership,get_const_simulation_domain());
  else if(row_domain==column_domain)
    get_data(std::string("Hamiltonian"), output, input_momentum_tuple, transfer_ownership,get_const_simulation_domain());
  else if(row_domain!=NULL || column_domain!=NULL)
  {
    NEMO_ASSERT(row_domain==get_const_simulation_domain()||column_domain==get_const_simulation_domain(),"Schroedinger(\""+get_name()+"\")::get_EOMMatrix called with 2 foreign domains\n");
    DOFmapInterface* coupling_DOFmap = neighbor_DOFmapInterface;
    if(row_domain==get_const_simulation_domain())
      get_data(std::string("Hamiltonian"), input_momentum_tuple, column_domain, output, coupling_DOFmap,get_const_simulation_domain());
    else
      get_data(std::string("Hamiltonian"), input_momentum_tuple, row_domain, output, coupling_DOFmap,get_const_simulation_domain());   
    if(coupling_DOFmap!=neighbor_DOFmapInterface)
    {
      delete coupling_DOFmap;
      coupling_DOFmap=NULL;
    }
  }
  else
    throw std::runtime_error("Schroedinger(\""+get_name()+"\")::get_EOMMatrix called configuration not implemented yet");
}

void Schroedinger::get_data(const string& variable, PetscMatrixParallelComplex*& matrix)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data3";
  NemoUtils::tic(tic_toc_prefix);

  if (variable=="Hamiltonian")
  {
    matrix=hamiltonian;
  }
  else if (variable=="overlap_matrix")
  {
    matrix=overlap;
  }
  else if (variable=="overlap_matrix_coupling")
  {
    matrix=overlap_coupling;
  }

  else if(variable=="real_space_trafo")
  {
    if(real_space_transformation==NULL)
      create_real_space_transformation();
    matrix=real_space_transformation;
    //throw std::runtime_error("Schroedinger::get_data real_space_trafo is not coded, yet\n");
  }
  else if(variable=="dimensional_S_matrix")
  {
    create_dimensional_S_Matrix(matrix);
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const string& variable, PetscMatrixParallelComplexBlock*& block_matrix)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data3";
  NemoUtils::tic(tic_toc_prefix);

  if (variable == "Hamiltonian")
  {
    block_matrix = dynamic_cast<PetscMatrixParallelComplexBlock*>(hamiltonian);
  }
  else if (variable == "overlap_matrix")
  {
    block_matrix = dynamic_cast<PetscMatrixParallelComplexBlock*>(overlap);
  }
  else if(variable == "real_space_trafo")
  {
    if(real_space_transformation == NULL)
      create_real_space_transformation();
    block_matrix = dynamic_cast<PetscMatrixParallelComplexBlock*>(real_space_transformation);
    //throw std::runtime_error("Schroedinger::get_data real_space_trafo is not coded, yet\n");
  }
  else
  {
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");
  }

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data (const string& variable, vector<unsigned int>& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data4";
  NemoUtils::tic(tic_toc_prefix);



  if (variable=="energy_number_map")
  {
    data.resize(vector_to_energy_map.size());
    data=vector_to_energy_map;
  }
  else if (variable=="inverse_energy_number_map")
  {
    data.resize(energy_to_vector_map.size());
    data=energy_to_vector_map;
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const string& variable, vector<double>& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data5";
  NemoUtils::tic(tic_toc_prefix);


  if(variable=="subset_of_eigenvalues")
  {
    data.clear();

    //Gather the size of subset_of_momenta from all MPI processes
    int number_of_processes;
    MPI_Comm_size(k_space_communicator,&number_of_processes);
    std::vector<int> subset_of_eigenvalues_found_per_CPU(number_of_processes,0);
    int send_size=subset_of_eigenvalues.size();
    MPI_Allgather(&send_size, 1, MPI_INT, &(subset_of_eigenvalues_found_per_CPU[0]),1, MPI_INT, k_space_communicator);
    std::vector<int> displs(number_of_processes, 0);
    for(int i = 1; i < number_of_processes; i++)
      displs[i] = displs[i-1] + subset_of_eigenvalues_found_per_CPU[i-1];

    unsigned int total_number_of_found_eigenvalues=0;
    for(unsigned int i=0; (int)i<number_of_processes; i++)
      total_number_of_found_eigenvalues+=subset_of_eigenvalues_found_per_CPU[i];

    data.resize(total_number_of_found_eigenvalues,0.0);

    //call Allgatherv
    std::vector<double> out_subset_of_eigenvalues(subset_of_eigenvalues.size(),0.0);
    std::vector<double> in_subset_of_eigenvalues(total_number_of_found_eigenvalues,0.0);

    for(unsigned int j=0; j<subset_of_eigenvalues.size(); j++)
      out_subset_of_eigenvalues[j]=subset_of_eigenvalues[j];

    MPI_Allgatherv(&(out_subset_of_eigenvalues[0]), subset_of_eigenvalues.size(), MPI_DOUBLE,
        &(in_subset_of_eigenvalues[0]), &(subset_of_eigenvalues_found_per_CPU[0]),&(displs[0]),MPI_DOUBLE,k_space_communicator);

    for(unsigned int j=0; j<total_number_of_found_eigenvalues; j++)
      data[j]=in_subset_of_eigenvalues[j];

  }
  else if(variable=="E_k_data")
  {
    int Nk = EnergyV.size();
    int NE = EnergyV[0].size();

    data.resize(Nk*NE);
    for (int k = 0; k < Nk; k++)
      for (int e = 0; e < NE; e++)
        data[k*NE + e] = EnergyV[k][e];
  }
  else if(variable=="k_vector")
  {
    data = k_vector;
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const string&, const vector<double>&,vector<double>&)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data6";
  NemoUtils::tic(tic_toc_prefix);


  throw std::runtime_error("Schroedinger::get_data(const string&, const vector<double>&,vector<double>&) not implemented\n");

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::get_data(const string& variable, unsigned int point, vector<double>& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data7";
  NemoUtils::tic(tic_toc_prefix);

  if (variable=="eigenvalues")
  {
    data.resize(EnergyV.size());
    data=EnergyV[point];
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const string& variable, unsigned int point, vector<cplx >& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data8";
  NemoUtils::tic(tic_toc_prefix);

  if (variable=="eigenfunction")
  {
    unsigned int index=energy_to_vector_map[point];
    this->get_eigenfunction(index, data);
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const string& variable, unsigned int point, PetscVectorNemo<cplx>& data)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data8a";
  NemoUtils::tic(tic_toc_prefix);

  if (variable=="eigenfunction")
  {
    unsigned int index=energy_to_vector_map[point];
    data = solver->get_eigenvector(index);
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");

  NemoUtils::toc(tic_toc_prefix);

}


void Schroedinger::get_data(const string& variable, NemoMesh*& Mesh_pointer)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data9";
  NemoUtils::tic(tic_toc_prefix);

  if (variable.find("k_space")!=std::string::npos)
    Mesh_pointer=kspace;
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string \""+variable+"\"\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const string& variable, double& data)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data10";
  NemoUtils::tic(tic_toc_prefix);

  if (variable=="number_of_orbitals")
  {
    const AtomisticDomain*  domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    //DOFmap&                 dof_map = get_dof_map();
    const AtomicStructure&  atoms   = domain->get_atoms();
    ConstActiveAtomIterator it      = atoms.active_atoms_begin();
    const AtomStructNode& nd = it.node();
    //    const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(it.id());
    const Atom*           atom      = nd.atom;
    HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(atom->get_material()));
    unsigned short n_orbital = tb_ham->get_number_of_orbitals(atom);
    data = n_orbital*1.0;
  }
  else if (variable=="fermi_level")
    data = Ef;
  else if (variable == "conduction_band_edge")
    data = get_conduction_band_min();
  else if (variable == "valence_band_edge")
    data = get_valence_band_max();
  else if (variable.find("averaged_effective_mass")!=std::string::npos)
  {
    bool this_is_for_electrons=true;
    NEMO_ASSERT(variable.find("conduction_band")!=std::string::npos||variable.find("valence_band")!=std::string::npos,
        tic_toc_prefix+"called with unknown variable \""+variable+"\"\n");
    if(variable.find("valence_band")!=std::string::npos)
      this_is_for_electrons=false;

    const AtomisticDomain* atomdomain  = dynamic_cast<const AtomisticDomain*> (get_const_simulation_domain()); //obtain this domain pointer
    const AtomicStructure& atoms = atomdomain->get_atoms(); //obtain all atoms of this domain
    ConstActiveAtomIterator it  = atoms.active_atoms_begin(); //first active atom of this domain
    ConstActiveAtomIterator end = atoms.active_atoms_end(); //last active atom
    unsigned int n_atoms=0;
    double summed_mass=0.0;
    map<string, double> mstar_c_dos_list;
    map<string, double> mstar_v_dos_list;
    map<string, double>::iterator it_mstar;
    std::vector<std::string> group(2);
    group[0] = "Bands";
    group[1] = "BandEdge";
    for(; it!=end; ++it)
    {
      //1. get the atom pointer of this atom iterator
      const AtomStructNode& atom_i_nd = it.node();
      const Atom* atom_pointer = atom_i_nd.atom;

      //2. get the material properties according to that atom
      const Material* material  = atom_pointer->get_material();
      //MaterialProperties* material_properties = get_material_properties(material);

      //HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*>(material_properties);
      //NEMO_ASSERT(tb_ham!=NULL, tic_toc_prefix+"Hamilton constructor pointer is NULL\n");

      //3. query database for the effective mass of that material; result=result+effective mass
      std::string parameter_name;
      if(this_is_for_electrons)
      {
        double mstar_c_dos;
        it_mstar = mstar_c_dos_list.find(material->get_name());
        if (it_mstar == mstar_c_dos_list.end())
        {
          mstar_c_dos = MaterialProperties::query_database_for_material(material->get_name(),group,"mstar_c_dos");
          mstar_c_dos_list[material->get_name()] = mstar_c_dos;
        }
        else
        {
          mstar_c_dos = it_mstar->second;
        }
        summed_mass+=mstar_c_dos;
      }
      else
      {
        double mstar_v_dos;
        it_mstar = mstar_v_dos_list.find(material->get_name());
        if (it_mstar == mstar_v_dos_list.end())
        {
          mstar_v_dos = MaterialProperties::query_database_for_material(material->get_name(),group,"mstar_v_dos");
          mstar_v_dos_list[material->get_name()] = mstar_v_dos;
        }
        else
        {
          mstar_v_dos = it_mstar->second;
        }
        summed_mass+=mstar_v_dos;
      }
      //4. increment the active atom counter
      n_atoms++;
    }
    data = summed_mass/n_atoms;
    data*=NemoPhys::electron_mass;
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string \""+variable+"\"\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const std::string& variable, const NemoMeshPoint& momentum, PetscMatrixParallelComplex*& Matrix, bool transfer_ownership,
    const Domain*)
{
  std::string tic_toc_prefix = options.get_option("tic_toc_name","Schroedinger(\""+tic_toc_name+"\")::get_data(Hamiltonian)");
  NemoUtils::tic(tic_toc_prefix);
  if (variable=="Hamiltonian")
  {
    // check whether the NemoMeshPoint has the correct dimensionality
    std::vector<double> temp_momentum;
    temp_momentum=momentum.get_coords();
    /*std::cerr<<"in Hamiltonian:\n"<<"size of temp_momentum is "<<temp_momentum.size()<<std::endl;
    for (unsigned int i=0;i<temp_momentum.size();++i)
      std::cerr<<"dimension "<<i<<" : "<<temp_momentum[i]<<std::endl;*/

    string valley_name;
    HamiltonConstructor* tb_ham = NULL;
    if (options.check_option("valleys") )
    {
      //valley_name = options.get_option("set_valley", string("DEFAULT"));
      const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
      const AtomicStructure& atoms   = domain->get_atoms();
      const AtomStructNode& nd        = atoms.active_atoms_begin().node();
      const Atom*           atom      = nd.atom;
      const Material*       material  = atom->get_material();
      tb_ham = dynamic_cast<HamiltonConstructor*> (get_material_properties(material));
      tb_ham->set_valley(this->valley);
    }
    if (temp_momentum.size()==k_vector.size())
      k_vector=temp_momentum;
    else
      throw std::runtime_error("Schroedinger::get_data mismatch in the dimensionality of momentum vectors\n");

    //get the Hamiltonian for the given momentum
    if(!combine_H)
      assemble_hamiltonian();
    else
      combine_hamiltonian();

    if(!transfer_ownership)
    {
      Matrix=hamiltonian;
    }
    else
    {
      Matrix = new PetscMatrixParallelComplex (*(hamiltonian->get_petsc_matrix()),hamiltonian->get_communicator(),hamiltonian->get_num_rows(),
          hamiltonian->get_num_cols(),true);
      hamiltonian->loose_ownership();
      delete hamiltonian;
      hamiltonian = NULL;
    }
  }
  else if (variable == std::string("overlap_matrix"))
  {
    std::string tb_basis=options.get_option("tb_basis", std::string("sp3"));
    //if(tb_basis.find("non_orthogonal")==std::string::npos)
    {
      if (!transfer_ownership)
      {
        Matrix=overlap;
      }
      else
      {
        Matrix = new PetscMatrixParallelComplex (*(overlap->get_petsc_matrix()),overlap->get_communicator(),overlap->get_num_rows(),
            overlap->get_num_cols(),true);
        overlap->loose_ownership();
        delete overlap;
        overlap = NULL;
      }
    }
    //else
    //{
    //  //create the overlap and let Matrix point to it
    //  throw std::runtime_error("Schroedinger::get_data not implemented yet\n");
    //}
  }
  else if(variable.find(std::string("velocity_operator"))!=std::string::npos)
  {
    PetscMatrixParallelComplex* position_operator=NULL;
    assemble_position_operator(variable,position_operator);
    if(variable.find(std::string("_x"))!=std::string::npos)
      position_operator->save_to_matlab_file("x.m");
    if(variable.find(std::string("_y"))!=std::string::npos)
      position_operator->save_to_matlab_file("y.m");
    if(variable.find(std::string("_z"))!=std::string::npos)
      position_operator->save_to_matlab_file("z.m");
    assemble_velocity_operator(momentum,position_operator,Matrix);
    delete position_operator;
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");
  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::get_data(const std::string& variable, PetscMatrixParallelComplex*& Matrix, const vector< NemoMeshPoint >& momentum, bool transfer,
    const Domain* this_domain)
{
  //author:Fabio
  std::string prefix="Schroedinger(\""+get_name()+"\")::get_data() ";
  std::string tic_toc_prefix = options.get_option("tic_toc_name","Schroedinger(\""+tic_toc_name+"\")::get_data(Hamiltonian, with vector of NemoMeshPoint)");
  NemoUtils::tic(tic_toc_prefix);
  //momentum can have size 0, 1 or 2
  //momentum has size 0 for case rectangular energy
  //momentum has size 1 for case valley or momentum
  //momentum has size 2 for case valley and momentum
  vector<string> quantum_number_order;
  NEMO_ASSERT(options.check_option("quantum_number_order"),prefix+"option quantum_number_order not defined");
  options.get_option("quantum_number_order",quantum_number_order);
  if(quantum_number_order[0]=="none")
  {
    //rectangular energy
    std::vector<double> K_Temp(3,0.0);  //set k point at gamma since there is no k dependence
    NemoMeshPoint temp_momentum_k_point(0,K_Temp);
    this->valley = "Default";
    get_data(variable,temp_momentum_k_point,Matrix,transfer,this_domain);  //call the get_data that handles a single K point
  }
  else if(momentum.size()==1)
  {
    if(momentum[0].get_dim()==1)
    {
      //momentum contains valley
      int valley_index = 0;   //used to store the index in momentum that is a valley NemoMeshPoint
      NEMO_ASSERT(options.check_option("valleys"),prefix+"valleys are not defined\n");  //check that valleys are defined in the input deck
      std::vector<string> valley_list;  //used to store the list of all valley names defined in the input deck
      options.get_option("valleys",valley_list);  //copy valley names to vlaley_list
      NEMO_ASSERT(momentum[valley_index].get_idx()<int(valley_list.size()),prefix+"valley index is out of scope\n"); //check that the id of valley is within the scope
      this->valley = valley_list[momentum[valley_index].get_idx()];  //valley=the name of the valley
      std::vector<double> K_Temp(3,0.0);  //set k point at gamma since there is no k dependence
      NemoMeshPoint temp_momentum_k_point(0,K_Temp);
      get_data(variable,temp_momentum_k_point,Matrix,transfer,this_domain);  //call the get_data that handles a single K point
    }
    else if(momentum[0].get_dim()==3)
    {
      //momentum contains k
      int k_index = 0;
      NemoMeshPoint temp_momentum_k_point = momentum[k_index];  //set k point value in temp_momentum_k_point
      get_data(variable,temp_momentum_k_point,Matrix,transfer,this_domain);  //call the get_data that handles a single K point
    }
    else
      throw std::runtime_error("Schroedinger::get_data called with unknown momentum with size=1, should be valley or K\n");
  }
  else if(momentum.size()==2)
  {
    //momentum contains valley and K
    int valley_index=0;
    int k_index=1;
    if(quantum_number_order[0]!="valley")
    {
      valley_index = 1;
      k_index = 0;
    }
    NEMO_ASSERT(options.check_option("valleys"),prefix+"valleys are not defined\n");
    std::vector<string> valley_list;  //used to store the list of all valley names defined in the input deck
    options.get_option("valleys",valley_list);  //copy valley names to vlaley_list
    NEMO_ASSERT(momentum[valley_index].get_idx()<int(valley_list.size()),prefix+"valley index is out of scope\n");
    this->valley = valley_list[momentum[valley_index].get_idx()];  //valley=the name of the valley
    NemoMeshPoint temp_momentum_k_point = momentum[k_index];  //set k point value in temp+momentum_k_point
    get_data(variable,temp_momentum_k_point,Matrix,transfer,this_domain);  //call the get_data that handles a single K point
    /*
    else if(quantum_number_order[0] == "angular_momentum" || quantum_number_order[1] == "angular_momentum")
    {
      int theta_index = 0;
      int k_index=1;
      if(quantum_number_order[0]!="angular_momentum")
      {
        theta_index = 1;
        k_index = 0;
      }
      NemoMeshPoint temp_momentum_k_point = momentum[k_index];  //set k point value in temp+momentum_k_point
      get_data(variable,temp_momentum_k_point,Matrix,transfer,this_domain);  //call the get_data that handles a single K point
    }
    */
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with incorrect NemoMeshPoint vector size, allowed size are 0, 1 or 2\n");
  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::get_data(const std::string& variable, const NemoMeshPoint& momentum, double& result)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::get_data11";
  NemoUtils::tic(tic_toc_prefix);

  if(variable=="integration_weight")
  {
    if(integrator.mesh_is_set())
    {
      //For custom and OMEN like k grid. Works with single periodic direction for now (UTB). (Bozidar)
      double kxmin = options.get_option("kxmin", 0.0);
      double kxmax = options.get_option("kxmax", 1.0);
      //For benchmarking with OMEN (Bozidar)
      if (options.get_option("OMEN_style_transverse_k_points", false))
      {
        const vector< vector<double> >& rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
        bool k_rotational_symmetry = options.get_option("k_rotational_symmetry",false);

        NEMO_ASSERT(rec_lattice.size()==1 ||  (k_rotational_symmetry && rec_lattice.size()==2),
            "[Schroedinger(\""+get_name()
            +"\")]::get_data()]: OMEN_style_transverse_k_points supports only single periodic direction for now (due to integration weights).");

        int number_of_k_points = options.get_option("number_of_k_points",10);

        if(momentum.get_idx() == 0 || momentum.get_idx() == options.get_option("number_of_kx_points",number_of_k_points)-1)
          result = (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1)/2;
        else
          result = (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1);
      }
      //Custom inhomogeneous transverse k grid (Pengyu, modified by Bozidar)
      else if (options.get_option("custom_transverse_k_points", false))
      {
        const vector< vector<double> >& rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
        bool k_rotational_symmetry = options.get_option("k_rotational_symmetry",false);

        NEMO_ASSERT(rec_lattice.size()==1 || (k_rotational_symmetry && rec_lattice.size()==2),
            "[Schroedinger(\""+get_name()+"\")] get_data()]: custom_transverse_k_points supports only single periodic direction for now (due to integration weights).");

        double dense_grid_ratio = options.get_option("dense_grid_ratio",0.2); //the ratio of total k that will have a denser grid (Pengyu)
        double grid_density = options.get_option("grid_density",1); //how many times denser will the grid be (Pengyu)
        int number_of_k_points = options.get_option("number_of_k_points",10);

        if (momentum.get_idx() == 0) //first point
          result = (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1)/(2*grid_density);
        else if(momentum.get_idx() > 0 && momentum.get_idx() < ((int)(options.get_option("number_of_kx_points",number_of_k_points)*dense_grid_ratio + 0.5))*
            grid_density) //dense grid
          result = (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1)/grid_density;
        else if(momentum.get_idx() == ((int)(options.get_option("number_of_kx_points",number_of_k_points)*dense_grid_ratio + 0.5))*grid_density) //transition point
          result = (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1)/grid_density/2 + 
          (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1)/2;
        else if(momentum.get_idx() == options.get_option("number_of_kx_points",number_of_k_points)-1) //last
          result = (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1)/2;
        else
          result = (kxmax-kxmin)/(options.get_option("number_of_kx_points",number_of_k_points)-1);
      }
      else
        result=integrator.get_weights()[momentum.get_idx()];

      /***
          NemoUtils::MsgLevel prev_level = msg.get_level();
          NemoUtils::msg.set_level(NemoUtils::MsgLevel(3));
          //msg <<"temp_domain->return_periodic_space_volume()*NemoPhys::nm_in_m = " << temp_domain->return_periodic_space_volume()*NemoPhys::nm_in_m << "\n";
          msg <<"kxmin kxmax integration weight = " << kxmin <<"\t" << kxmax <<"\t" << result <<"\t" << "\n";
       ***/
      //two cases are possible:
      //we want to get a density: divide by the integration volume
      //we want to get quantity in the unit volume: do not divide

      //adding this back after it was removed in 16627 MP,JF
      //if (options.get_option("integration_to_get_density", false))
      {
        const Domain* temp_domain=get_const_simulation_domain();
        result/=temp_domain->return_periodic_space_volume();
        //msg <<"integration_to_get_density is true: periodic_volume integration weight = " << temp_domain->return_periodic_space_volume()<<"\t" << result <<"\t" << "\n";

      }


      //      const Domain* temp_domain=get_const_simulation_domain();
      //      result/=temp_domain->return_periodic_space_volume();

      /*std::cerr<<get_name()<<" 1:"<<result<<"\n";*/
      result *= k_degeneracy;
      result *= pow((2*M_PI),k_space_dimensionality);//needed for convention with formulas in the Propagation class
      //msg << "k_degeneracy k_space_dimensionality integration weight = " << k_degeneracy <<"\t" << k_space_dimensionality << "\t" << result << "\n";
      //NemoUtils::msg.set_level(prev_level);
    }
    else
      result=1.0;

  }
  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");
  //std::cerr<<get_name()<<" 2:"<<result<<"\n";

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::get_data(const std::string& variable, const NemoMeshPoint& momentum, const Domain* neighbor_domain,
    PetscMatrixParallelComplex*& Matrix, DOFmapInterface*& resulting_DOFmap, const Domain*)
{
  std::string tic_toc_prefix = options.get_option("tic_toc_name","Schroedinger(\""+tic_toc_name+"\")::get_data(interdomain-Hamiltonian)");
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Schroedinger(\""+get_name()+"\")::get_data ";
  if (variable=="Hamiltonian")
  {
    //std::cerr << prefix << "\n";
    // check whether the NemoMeshPoint has the correct dimensionality
    std::vector<double> temp_momentum=momentum.get_coords();
    //string valley_name;
    //    HamiltonConstructor* tb_ham = NULL;
    if (options.check_option("valleys") )
    {
      //string valley_name = options.get_option("set_valley", string("DEFAULT"));
      const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
      const AtomicStructure& atoms   = domain->get_atoms();
      ConstActiveAtomIterator it  = atoms.active_atoms_begin();
      ConstActiveAtomIterator end  = atoms.active_atoms_end();
      for( ; it != end; ++it)
      {
        const AtomStructNode& nd        = it.node();
        const Atom*           atom      = nd.atom;
        const Material*       material  = atom->get_material();
        HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (get_material_properties(material));
        tb_ham->set_valley(this->valley);
      }
    }
    if (temp_momentum.size()==k_vector.size())
    {
      k_vector=temp_momentum;
      /*std::cerr<<k_vector[0]<<","<<k_vector[1]<<","<<k_vector[2]<<"\n";
      NEMO_ASSERT(k_vector[2]>0.0,prefix+"here\n");*/
    }
    else
    {
      OUT_ERROR<< "[Schroedinger(\""+get_name()+"\")]" << "expected dimensionality of momentum vector: "<<k_vector.size()
                   <<"\ndimensionality of received momentum vector: "<<temp_momentum.size()
             <<"\n";
      throw std::runtime_error("[Schroedinger(\""+get_name()+"\")]::get_data mismatch in the dimensionality of momentum vectors\n");
    }
    //cerr << "neighbour " << get_name() << "   " <<  k_vector[0] << "  " << k_vector[1] << "  " <<  k_vector[2] << "\n";
    assemble_hamiltonian(neighbor_domain, Matrix, resulting_DOFmap);
    //Matrix=hamiltonian;
  }
  else if(variable=="overlap_matrix_coupling")
  {
    std::string tb_basis=options.get_option("tb_basis", std::string("sp3"));
    //if(tb_basis.find("non_orthogonal")==std::string::npos)
    {
      Matrix = overlap_coupling; //new PetscMatrixParallelComplex(*overlap_coupling);  //NULL;
      resulting_DOFmap=NULL;
    }
    //else
    //{
    //  //create the overlap_coupling and let Matrix point to it
    //  throw std::runtime_error(prefix+"not implemented yet\n");
    //}
  }
  else if(variable =="DOFmap")
  {
    //code copied from assemble_hamiltonian(nonvoid)
    //DOFmap initialization
    {
      const HamiltonConstructor* neighbor_tb_ham = dynamic_cast<const HamiltonConstructor*> ( material_properties.begin()->second);
      NEMO_ASSERT(neighbor_tb_ham!=NULL, "[Schroedinger(\""+get_name()
          +"\")]::assemble_hamiltonian: tried to cast neighbor_tb_ham into HamiltonConstructor but did not succeed.");
      std::vector<string> temp_vars = neighbor_tb_ham->get_orbital_names() ;
      for (unsigned int i = 0; i < temp_vars.size(); i++)
        resulting_DOFmap->add_variable(temp_vars[i]);

      //use smart DOF ordering: find repartitioner for the neighbour and use it
      if (options.check_option("atom_order_from"))
      {
        string partitioner_solver_name = options.get_option("atom_order_from",string(""));
        Simulation* partitioner = find_simulation(partitioner_solver_name);
        if (partitioner == NULL)
          throw invalid_argument("TBSchroedinger: wrong atom_order_from option for solver " + get_name() + "\n");
        else
        {
          resulting_DOFmap->attach_repartitioner(partitioner);
        }
      }

      if(options.check_option("use_neighbour_repartitioner_"+neighbor_domain->get_name()))
      {

        string partitioner_solver_name = options.get_option("use_neighbour_repartitioner_"+neighbor_domain->get_name(),string(""));
        Simulation* partitioner = find_simulation(partitioner_solver_name);
        DOFmap* temp_dofmap=dynamic_cast<DOFmap*>(resulting_DOFmap);
        NEMO_ASSERT(temp_dofmap!=NULL,"Schroedinger(\""+get_name()+"\")::get_data resulting_DOFmap is not a DOFmap\n");
        if (partitioner != NULL)
          temp_dofmap->attach_repartitioner_for_neighbour(partitioner);

        bool order = options.get_option("direct_order_of_partitions_"+neighbor_domain->get_name(), true);
        if (order)
          temp_dofmap->set_order_of_partitions("direct");
        else
          temp_dofmap->set_order_of_partitions("reversed");
      }
      const AtomisticDomain* temp_domain_pointer=dynamic_cast<const AtomisticDomain*>(neighbor_domain);
      resulting_DOFmap->init(dynamic_cast<const AtomisticDomain*>(get_simulation_domain()),temp_domain_pointer);  //DOFMap of the inter-domain Hamiltonian
    }
  }



  else
    throw std::runtime_error("Schroedinger::get_data called with unknown data-string\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const std::string& variable, const vector< NemoMeshPoint >& momentum, const Domain* neighbor_domain,
    PetscMatrixParallelComplex*& Matrix,
    DOFmapInterface*& resulting_DOFmap,const Domain* this_domain)
{
  //author:Fabio
  std::string prefix="Schroedinger(\""+get_name()+"\")::get_data() ";
  std::string tic_toc_prefix = options.get_option("tic_toc_name",
      "Schroedinger(\""+tic_toc_name+"\")::get_data(interdomain-Hamiltonian, with vector of NemoMeshPoint)");
  NemoUtils::tic(tic_toc_prefix);
  vector<string> quantum_number_order;
  NEMO_ASSERT(options.check_option("quantum_number_order"),prefix+"option quantum_number_order not defined");
  options.get_option("quantum_number_order",quantum_number_order);
  //momentum can have size 0, 1 or 2
  //momentum has size 0 for case rectangular energy
  //momentum has size 1 for case valley or momentum
  //momentum has size 2 for case valley and momentum
  if(quantum_number_order[0]=="none")
  {
    //nonrectangular energy
    std::vector<double> K_Temp(3,0.0);  //set k point at gamma since there is no k dependence
    NemoMeshPoint temp_momentum_k_point(0,K_Temp);
    this->valley = "Default";
    get_data(variable,temp_momentum_k_point,neighbor_domain,Matrix,resulting_DOFmap,this_domain);  //call the get_data that handles a single K point
  }
  else if(momentum.size()==1)
  {
    if(momentum[0].get_dim()==1)
    {
      //momentum contains valley
      int valley_index = 0;   //used to store the index in momentum that is a valley NemoMeshPoint
      NEMO_ASSERT(options.check_option("valleys"),tic_toc_prefix+"valleys are not defined\n");  //check that valleys are defined in the input deck
      std::vector<string> valley_list;  //used to store the list of all valley names defined in the input deck
      options.get_option("valleys",valley_list);  //copy valley names to vlaley_list
      NEMO_ASSERT(momentum[valley_index].get_idx()<int(valley_list.size()),
          tic_toc_prefix+"valley index is out of scope\n"); //check that the id of valley is within the scope
      this->valley = valley_list[momentum[valley_index].get_idx()];  //valley=the name of the valley
      std::vector<double> K_Temp(3,0.0);  //set k point at gamma since there is no k dependence
      NemoMeshPoint temp_momentum_k_point(0,K_Temp);
      get_data(variable,temp_momentum_k_point,neighbor_domain,Matrix,resulting_DOFmap,this_domain);  //call the get_data that handles a single K point
    }
    else if(momentum[0].get_dim()==3)
    {
      //momentum contains k
      int k_index = 0;
      NemoMeshPoint temp_momentum_k_point = momentum[k_index];  //set k point value in temp_momentum_k_point
      get_data(variable,temp_momentum_k_point,neighbor_domain,Matrix,resulting_DOFmap,this_domain);  //call the get_data that handles a single K point
    }
    else
      throw std::runtime_error("Schroedinger::get_data called with unknown momentum with size=1, should be valley+energy or K+energy\n");
  }
  else if(momentum.size()==2)
  {
    //momentum contains valley and k
    int k_index = 1;
    int valley_index = 0;
    if(quantum_number_order[0]!="valley")
    {
      valley_index = 1;
      k_index = 0;
    }
    NEMO_ASSERT(options.check_option("valleys"),tic_toc_prefix+"valleys are not defined\n");
    std::vector<string> valley_list;  //used to store the list of all valley names defined in the input deck
    options.get_option("valleys",valley_list);  //copy valley names to vlaley_list
    NEMO_ASSERT(momentum[valley_index].get_idx()<int(valley_list.size()),tic_toc_prefix+"valley index is out of scope\n");
    this->valley = valley_list[momentum[valley_index].get_idx()];  //valley=the name of the valley
    NemoMeshPoint temp_momentum_k_point = momentum[k_index];  //set k point value in temp+momentum_k_point
    get_data(variable,temp_momentum_k_point,neighbor_domain,Matrix,resulting_DOFmap,this_domain);  //call the get_data that handles a single K point
    /*
    else if(quantum_number_order[0] == "angular_momentum" || quantum_number_order[1] == "angular_momentum")
    {
      int theta_index = 0;
      int k_index=1;
      if(quantum_number_order[0]!="angular_momentum")
      {
        theta_index = 1;
        k_index = 0;
      }
      NemoMeshPoint temp_momentum_k_point = momentum[k_index];  //set k point value in temp+momentum_k_point
      get_data(variable,temp_momentum_k_point,neighbor_domain,Matrix,resulting_DOFmap,this_domain);  //call the get_data that handles a single K point
    }
    */
  }
  else
    throw std::runtime_error("Schroedinger::get_data called with incorrect NemoMeshPoint vector size, allowed size is 0, 1 or 2\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::set_em_valley(const string valley_name)
{
  //author:Fabio
  std::string prefix="Schroedinger(\""+get_name()+"\")::set_valley ";
  std::string tic_toc_prefix = options.get_option("tic_toc_name","Schroedinger(\""+tic_toc_name+"\")::set_valley");
  NemoUtils::tic(tic_toc_prefix);
  this->valley = valley_name;
  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::create_k_space_object (void)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::create_k_space_object";
  NemoUtils::tic(tic_toc_prefix);
  /*
  {
    const vector< vector<double> >& rec_vecs = this->get_simulation_domain()->get_reciprocal_vectors();
    cerr << "&&&&&&&&&&\n";
    for (int i = 0; i < rec_vecs.size(); i++)
    {
      for (int j = 0; j < 3; j++)
        cerr << rec_vecs[i][j] << "   ";

      cerr << "\n";
    }
  }
   */
  msg.print_message(MSG_LEVEL_1,"[Schroedinger(\""+get_name()+"\")] creating k-space object");
  k_space_dimensionality = get_simulation_domain()->get_reciprocal_vectors().size();

  // consistency check:
  string k_space_basis = options.get_option("k_space_basis",string("reciprocal"));
  if (k_space_basis=="reciprocal" && get_simulation_domain()->get_reciprocal_vectors().size()<1)
    throw invalid_argument("Schroedinger:create_k_space_object:\"reciprocal\" and setting all periodicity direction to \"false\" are in contradiction!\n");

  //Roza Kotlyar; 3/11/2011
  //change reciprocal space from unstrained to strained if constant_strain option is requested
  if (constant_strain)
  {
    //    this->get_simulation_domain()->calculate_reciprocal_basis_under_constant_strain(strain_matrix);
  }

  vector< vector<double> > k_pointsV;
  vector<unsigned int>    number_of_nodesV;
  // band structure case due to job_list
  if (calculate_band_structure || output_H_all_k)
  {
    if (options.get_option("calculate_brillouin_zone",false))
    {
      // calculate entire brillouin zone
      const vector< vector<double> >& rec_vecs = this->get_simulation_domain()->get_reciprocal_vectors();


      kspace->calculate_brillouin_zone(rec_vecs);

      double dx = options.get_option("brillouin_zone_meshsize", 0.1);

      // set mesh boundary (=BZ), meshsize
      msg << "[Schroedinger(\""+get_name()+"\")] Setting k-space mesh boundary...\n";
      vector< vector<double> > bz;
      vector< vector<int> > bz_facets;
      kspace->get_brillouin_zone(bz);
      kspace->get_brillouin_zone_facets(bz_facets);
      kspace->set_mesh_boundary(bz);
      kspace->set_mesh_boundary_facets(bz_facets);

      msg << "[Schroedinger(\""+get_name()+"\")] Creating k-points using dx=" << dx << "...\n";
      if (rec_vecs.size()==1)
      {
        kspace->set_segmental_mesh(true);
        double bz_length = kspace->get_bz_volume();
        int num_nodes = (int)(bz_length / dx) + 1;
        kspace->set_segmental_num_points(0, num_nodes);
      }
      else
      {
        if (rec_vecs.size()==2)
        {
          bool found = false;
          for (unsigned int ii=1; ii<bz.size(); ii++)
          {
            double cos_angle = NemoMath::vector_scalar_product_3d(&(bz[0][0]), &(bz[ii][0]))
            / (NemoMath::vector_norm_3d(&bz[0][0]) * NemoMath::vector_norm_3d(&bz[ii][0]) );
            if (NemoMath::abs(cos_angle) < 0.99999)
            {
              kspace->set_2D_mesh_of_3D_points(true, bz[0], bz[ii]);
              found = true;
              break;
            }
          }
          NEMO_ASSERT(found, "[Schroedinger(\""+get_name()+"\")] 2D Brillouin zone corner points all seemed to be collinear.");
          kspace->set_tensorial_mesh_refinement(0, -1.0, 1.0, dx);
          kspace->set_tensorial_mesh_refinement(1, -1.0, 1.0, dx);
        }
        kspace->set_meshsize(dx);
      }
      kspace->calculate_mesh();
    }
    else if (options.check_option("k_points"))
    {
      msg<< "entered K-points\n";//rv

      // calculate multiple k-points along some path in k-space
      // k_points parameter may contain:
      // - a single point (only 1 k-point is solved)
      // - two points A,B (the segment from A to B is solved)
      // - more than two points A,B,C,D,... (the segments A-B, B-C, C-D etc. are solved)
      options.get_option("k_points",k_pointsV);
      NEMO_ASSERT(k_pointsV.size()>0, "[Schroedinger(\""+get_name()+"\")] k_points parameter seemed to be empty.");

      // discretization of segments: number_of_nodesV
      if(options.check_option("number_of_nodes"))
      {
        options.get_option("number_of_nodes", number_of_nodesV);
        NEMO_ASSERT((number_of_nodesV.size()==k_pointsV.size()-1)||(number_of_nodesV.size()==1 && number_of_nodesV[0]==1 &&
            k_pointsV.size()==1), "[Schroedinger(\""+get_name()+"\")] need to have 1 number_of_nodes entry for each k-segment.\n");
        if (number_of_nodesV.size()==1 && number_of_nodesV[0]==1 && k_pointsV.size()==1)
          number_of_nodesV.clear();
      }
      else if (k_pointsV.size()!=1)
      {
        msg << "[Schroedinger(\""+get_name()+"\")] did not find parameter number_of_nodes in input deck - assume  points per k-segment.\n";
        number_of_nodesV.resize(k_pointsV.size()-1, 11);
      }

      kspace->set_mesh_boundary(k_pointsV);
      kspace->set_segmental_mesh(true);
      for (unsigned int ii=0; ii<number_of_nodesV.size(); ii++)
      {
        kspace->set_segmental_num_points(ii, number_of_nodesV[ii]);
      }
      kspace->calculate_mesh();
      msg << "K-mesh done\n";//rv

      // transform k_space into cartesian coordinates (if necessary)
      string k_space_basis = options.get_option("k_space_basis",string("reciprocal"));
      if (k_space_basis!="cartesian")
      {
        msg<<"transform k_space into cartesian coordinates (if necessary)\n";//rv
        vector <vector<double> > transformation_matrix;
        unsigned int number_of_rec_vectors=get_simulation_domain()->get_reciprocal_vectors().size();
        transformation_matrix.resize(3);
        for (unsigned int i=0; i<3; i++)
        {
          transformation_matrix[i].resize(number_of_rec_vectors,0.0);
          for (unsigned int j=0; j<number_of_rec_vectors; j++)
            /*
            // rvedula if statements to correct k-space for constant strain
              if(constant_strain)
              transformation_matrix[i][j]=
                get_simulation_domain()->get_reciprocal_vectors_strained()[j][i]; //transpose transformation==inverse transformation for orthonormal vectors
            else
             */
            //M.P. "strained" reciprocal space is returned 
            transformation_matrix[i][j]=
                get_simulation_domain()->get_reciprocal_vectors()[j][i]; //transpose transformation==inverse transformation for orthonormal vectors
        }
        kspace->transform_basis(transformation_matrix);
      }
    }
    //RK for bandstructure optimization
    //-------------RK reading and solving on a list of k-points from file in units (2pi/ax,2pi/ay,2pi/az)
    else if (options.check_option("k_points_from_file"))
    {
      //x,y,z are in principal crystal coordinates
      //msg<< "entered K-points from file in units (2pi/ax,2pi/ay,2pi/az) \n";//rv

      // arbitrary list in a file
      //----------------------------------------------
      string fileoutname = options.get_option("input_filename_for_k_points", string(""));

      if(fileoutname == ""){
        throw runtime_error("Schroedinger:: file name with input k-points is not provided for the k_points_from_file option\n");
      }
      //--------------------------------------------------------------------
      this->get_lattice_constants(lattice_a);
      //--------------------------------------------------------------------
      int places=15;
      int numk, itmp;
      double kvt[3];
      std::ifstream kinp_file(fileoutname.c_str(),std::ios_base::in);
      if ( kinp_file.is_open())
      {
        kinp_file.precision(places);

        kinp_file>>numk;

        for (int ik=0; ik<numk; ik++)
        {
          kinp_file>>itmp >> kvt[0] >> kvt[1] >> kvt[2];
          //-----------------------------------------------------
          vector<double> k_pointV(3,0.0);
          for (unsigned int i2=0;i2<3;i2++)
          {
            k_pointV[i2]=kvt[i2]*2.0*NemoMath::pi/fabs(lattice_a[i2]);
          }
          kspace->add_point(k_pointV);
          //-----------------------------------------------------
        }
        kinp_file.close();
        //----------------------------------------------
      }
      else
      {
        throw runtime_error("[Schroedinger] input_filename_for_k_points cannot be opened\n");
      }
    }
    //--------------------------------------------------------------------------------------------------
    //RK for bandstructure optimization


    else if (options.get_option("k_mesh",false))
    {
      //Allow reading a vector of k_points directly (no LIBMESH grid generation) (Bozidar)
      if (!options.check_option("k_points_vector"))
      {
        int number_of_k_points = options.get_option("number_of_k_points",10);

        //Allow different number of points in different directions (Bozidar)
        int number_of_kx_points = options.get_option("number_of_kx_points",number_of_k_points);
        int number_of_ky_points = options.get_option("number_of_ky_points",number_of_k_points);
        int number_of_kz_points = options.get_option("number_of_kz_points",number_of_k_points);

        double kxmin;
        double kymin;
        double kzmin;
        double kxmax;
        double kymax;
        double kzmax;

        string k_space_basis = options.get_option("k_space_basis",string("reciprocal"));
        const vector< vector<double> >& rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
#ifndef NOLIBMESH
	Mesh mesh(Parallel::Communicator(MPI_COMM_SELF));

        //Allow input in Cartesian coordinates (Bozidar)
        if (k_space_basis!="cartesian")
        {
          kxmin = options.get_option("kxmin", 0.0);
          kymin = options.get_option("kymin", 0.0);
          kzmin = options.get_option("kzmin", 0.0);
          kxmax = options.get_option("kxmax", 1.0);
          kymax = options.get_option("kymax", 1.0);
          kzmax = options.get_option("kzmax", 1.0);

          mesh.set_mesh_dimension(rec_lattice.size());
#if LIBMESH_MAJOR_VERSION < 1 
          mesh.partitioner() =  AutoPtr<Partitioner>(NULL);  //no mesh partitioning!
#endif
          if (rec_lattice.size()==1)
          {
            MeshTools::Generation::build_line(mesh, number_of_kx_points, kxmin, kxmax, EDGE2);
          }
          else if (rec_lattice.size()==2)
          {
            MeshTools::Generation::build_square(mesh, number_of_kx_points, number_of_ky_points, kxmin, kxmax, kymin, kymax,  QUAD4);
          }
          else if (rec_lattice.size()==3)
          {
            MeshTools::Generation::build_cube(mesh, number_of_kx_points, number_of_ky_points, number_of_kz_points, kxmin, kxmax, kymin, kymax, kzmin, kzmax,  HEX8);
          }
          else throw invalid_argument("[Schroedinger(\""+get_name()+"\"):create_k_space_object] wrong dimensionality of reciprocal_lattice!\n");
        }
        else
        {
          kxmin = options.get_option("kxmin", 0.0);
          kymin = options.get_option("kymin", 0.0);
          kzmin = options.get_option("kzmin", 0.0);
          kxmax = options.get_option("kxmax", 0.0);
          kymax = options.get_option("kymax", 0.0);
          kzmax = options.get_option("kzmax", 0.0);

          std::vector<double> kminmax;
          std::vector<double> num_k_points;
          if (kxmin != kxmax)
          {
            kminmax.push_back(kxmin);
            kminmax.push_back(kxmax);
            num_k_points.push_back(number_of_kx_points);
          }
          if (kymin != kymax)
          {
            kminmax.push_back(kymin);
            kminmax.push_back(kymax);
            num_k_points.push_back(number_of_ky_points);
          }
          if (kzmin != kzmax)
          {
            kminmax.push_back(kzmin);
            kminmax.push_back(kzmax);
            num_k_points.push_back(number_of_kz_points);
          }

          NEMO_ASSERT(kminmax.size()/2 > 0, "[Schroedinger(\""+get_name()+"\"):create_k_space_object()] k-mesh must at leat be one-dimensional.\n");

          mesh.set_mesh_dimension(kminmax.size()/2);
#if (LIBMESH_MAJOR_VERSION < 1)
          mesh.partitioner() =  AutoPtr<Partitioner>(NULL);  //no mesh partitioning!
#endif
          if (kminmax.size()/2==1)
          {
            MeshTools::Generation::build_line(mesh, num_k_points[0], kminmax[0], kminmax[1], EDGE2);
          }
          else if (kminmax.size()/2==2)
          {
            MeshTools::Generation::build_square(mesh, num_k_points[0], num_k_points[1], kminmax[0], kminmax[1], kminmax[2], kminmax[3],  QUAD4);
          }
          else if (kminmax.size()/2==3)
          {
            MeshTools::Generation::build_cube(mesh, num_k_points[0], num_k_points[1], num_k_points[2], kminmax[0], kminmax[1],
                kminmax[2], kminmax[3], kminmax[4], kminmax[5],  HEX8);
          }
        }

        integrator.set_mesh(&mesh);
        int integration_order  = options.get_option("integration_order", 1);
        integrator.set_order(integration_order);
        integrator.build_integral_sum();
        const vector<vector<double> >& x = integrator.get_points();

        int n = x.size();
        for (int i=0; i<n; i++)
        {
          vector<double> k_pointV(3,0.0);
          if (k_space_basis!="cartesian") //If cartesian is input, no need to transform (Bozidar)
          {
            for (unsigned int i1=0; i1<rec_lattice.size(); i1++)
            {
              for (unsigned int i2=0; i2<3; i2++)
              {
                k_pointV[i2]+=x[i][i1]*rec_lattice[i1][i2];
              }
            }
          }
          else
          {
            double dim = 0;
            if (kxmin == kxmax)
              k_pointV[0] = kxmin;
            else
              k_pointV[0] = x[i][dim++];
            if (kymin == kymax)
              k_pointV[1] = kymin;
            else
              k_pointV[1] = x[i][dim++];
            if (kzmin == kzmax)
              k_pointV[2] = kzmin;
            else
              k_pointV[2] = x[i][dim];
          }
          kspace->add_point(k_pointV);
        }

#else
        NEMO_EXCEPTION("[Schroedinger(\""+get_name()+"\")] NEMO5 was compiled without libmesh.");
#endif
      }
      else
      {
        std::vector<std::vector<double> > k_points;
        options.get_option("k_points_vector", k_points);

        NEMO_ASSERT(options.get_option("k_space_basis",string("reciprocal")) == string("cartesian"), "[Schroedinger(\""+get_name()+"\")] with k_points_vector option" +
            " cartesian basis is required for now.\n");
        NEMO_ASSERT(k_points.size() >= 1, "[Schroedinger(\""+get_name()+"\")] k_points_vector must have at least one point.\n");

        for (unsigned int i = 0; i < k_points.size(); i++)
        {
          NEMO_ASSERT(k_points[i].size() == 3, "[Schroedinger(\""+get_name()+"\")] k_points_vector must contain 3-dimensional k-points.\n");
          kspace->add_point(k_points[i]);
        }
      }

    }
    else
    {
      throw invalid_argument("Schroedinger:create_k_space_object: \"k_points\" have to be set in the input deck!\n");
    }
  } // end if (calculate_band_structure)
  // set up the k-space due to integrator; required for k-space integration
  // density calculation due to job_list
  else if (electron_density || hole_density || density_of_states || spatial_density_of_states)
  {
    vector< vector<double> >  rec_lattice;
    /*
    if (constant_strain)
    {
      rec_lattice = get_simulation_domain()->get_reciprocal_vectors_strained();
    }
    else
    {
      rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
    }
     */
    //M.P.get_reciprocal_vectors() returns that actual i.e. strained reciprocal vectors
    rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
    int number_of_k_points = options.get_option("number_of_k_points",10);
    int number_of_kx_points = options.get_option("number_of_kx_points",number_of_k_points);
    int number_of_ky_points = options.get_option("number_of_ky_points",number_of_k_points);
    int number_of_kz_points = options.get_option("number_of_kz_points",number_of_k_points);
    //BN: variable number_of_k_points is not used below this line. Use only number_of_kx_points, number_of_ky_points, number_of_kz_points.

    if(number_of_kx_points*number_of_ky_points*number_of_kz_points == 0)
    {
      // analytical k-point integration
      // --> calculate only k=0
      msg << "[Schroedinger(\""+get_name()+"\")] detected analytical (effective mass) k-integration - setting up k=0 only.\n";
      this->analytical_k = true;
      vector<double> zeros(3,0.0);
      kspace->add_point(zeros);
    }
    else
    {
      // <ss 26.8.11> so far basis=reciprocal was implicitly assumed. for 3D-periodicity we might wanna have cartesian!
      string basis = "reciprocal";
      if (rec_lattice.size()==3)
      {
        basis = options.get_option("k_space_basis", string("reciprocal"));
      }
      //cout << "SCHROEDINGER BASIS: " << basis << endl;

      // possibility for the user to define the maximum k-coordinates by himself
      // it is up to the user to then define k_degeneracy in order to account for the incomplete k-space
      double kxmin = options.get_option("kxmin", 0.0);
      double kymin = options.get_option("kymin", 0.0);
      double kzmin = options.get_option("kzmin", 0.0);
      double kxmax = options.get_option("kxmax", 1.0);
      double kymax = options.get_option("kymax", 1.0);
      double kzmax = options.get_option("kzmax", 1.0);



#ifndef NOLIBMESH
      SerialMesh mesh(Parallel::Communicator(MPI_COMM_SELF),rec_lattice.size());

       //no mesh partitioning!
      if (rec_lattice.size()==0)
      {
        vector<double> zeroV(3,0.0);
        kspace->add_point(zeroV);
      }
      else
      {
        //Use high order Gaussian quadrature with only one k-space element for total number of k points<=22 (libmesh limitation), after that use simple 
        //rectangular with multi-element. Keep backward compatibility (Bozidar).
        int integration_order;
        int number_of_kx_points_temp = number_of_kx_points, number_of_ky_points_temp = number_of_ky_points, number_of_kz_points_temp = number_of_kz_points;
        if (options.check_option("integration_order"))
        {
          integration_order = options.get_option("integration_order", 1);
        }
        else
        {
          if (options.get_option("gaussian_quadrature_order_automatic",false))
          {
            if (number_of_kx_points > 22)
            {
              integration_order = 1;
            }
            else
            {
              integration_order = 2*number_of_kx_points - 1;
              number_of_kx_points_temp = 1;
              number_of_ky_points_temp = 1;
              number_of_kz_points_temp = 1;
            }
          }
          else
          {
            integration_order = 1;
          }
        }

        bool k_rotational_symmetry = options.get_option("k_rotational_symmetry", false);

        //std::cerr<<"in create_k_space_object: number_of_k_points is: " <<number_of_k_points<<"\n";
        if (rec_lattice.size()==1 ||  (k_rotational_symmetry && rec_lattice.size()==2) )
          MeshTools::Generation::build_line(mesh, number_of_kx_points_temp, kxmin, kxmax, EDGE2);
        else if (rec_lattice.size()==2)
          MeshTools::Generation::build_square(mesh, number_of_kx_points_temp, number_of_ky_points_temp, kxmin, kxmax, kymin, kymax,  QUAD4);
        else if (rec_lattice.size()==3)
          MeshTools::Generation::build_cube(mesh, number_of_kx_points_temp, number_of_ky_points_temp, number_of_kz_points_temp, kxmin, kxmax, kymin, kymax, kzmin, kzmax,
              HEX8);
        else
          throw invalid_argument("Schroedinger:create_k_space_object: do_init wrong dimensionality of reciprocal_lattice!\n");

        if (k_rotational_symmetry)
          integrator.rotation_symmetry() = true;

        integrator.set_mesh(&mesh);

        //add the subset_of_momenta of the previous simulation
        if(scattering_probability)
        {
          if(options.check_option("extra_points"))
          {
            //get the user given points from the input deck
            std::vector<std::vector<double> > extra_points;
            options.get_option("extra_points",extra_points);
            integrator.add_points_to_existing_mesh(extra_points);
            integrator.set_order(1);
            integrator.set_rule(std::string("trapezoidal"));
          }
        }


        integrator.set_order(integration_order);
        integrator.build_integral_sum(); //Gaussian quadrature by default. First order Gaussain is just simple rectangular (Bozidar).

        vector<vector<double> > x = integrator.get_points();

        //If required avoid using libmesh generation with reciprocal space and generate k points like done below for cartesian space (Bozidar).
        if (options.get_option("OMEN_style_transverse_k_points", false))
        {
          int temp_size = 0;
          if(rec_lattice.size()==1 ||  (k_rotational_symmetry && rec_lattice.size()==2))
            temp_size = number_of_kx_points;
          else if(rec_lattice.size()==2)
            temp_size = number_of_kx_points*number_of_ky_points;
          else if(rec_lattice.size()==3)
            temp_size = number_of_kx_points*number_of_ky_points*number_of_kz_points;

          //These loops go over all three possible k directions, even nonexisting in our device, but due to "continue" statement the nonexisting will be skipped.
          vector<double> tmp(rec_lattice.size());
          vector<vector<double> > x_temp(temp_size, tmp);
          for (int kx=0; kx<number_of_kx_points; kx++)
          {
            tmp[0] = kx * (kxmax-kxmin)/(number_of_kx_points-1.0) + kxmin;
            if(rec_lattice.size() == 1 || (k_rotational_symmetry && rec_lattice.size()==2))
            {
              x_temp[kx] = tmp;
              msg << "adding point (" << tmp[0] << ")..." << endl;
              continue;
            }
            for (int ky=0; ky<number_of_ky_points; ky++)
            {
              tmp[1] = ky * (kymax-kymin)/(number_of_ky_points-1.0) + kymin;
              if (rec_lattice.size() == 2)
              {
                x_temp[kx*number_of_ky_points+ky] = tmp;
                msg << "adding point (" << tmp[0] << "," << tmp[1] << ")..." << endl;
                continue;
              }
              for (int kz=0; kz<number_of_kz_points; kz++)
              {
                tmp[2] = kz * (kzmax-kzmin)/(number_of_kz_points-1.0) + kzmin;
                if (rec_lattice.size() == 3)
                {
                  x_temp[(kx*number_of_ky_points+ky)*number_of_kz_points+kz] = tmp;
                  msg << "adding point (" << tmp[0] << "," << tmp[1] << "," << tmp[2] << ")..." << endl;
                  continue;
                }
              }
            }
          }
          x = x_temp;
        }
        //Custom inhomogeneous transverse k grid (Pengyu, modified by Bozidar)
        else if (options.get_option("custom_transverse_k_points", false))
        {
          double dense_grid_ratio = options.get_option("dense_grid_ratio",0.2); //the ratio of total k that will have a denser grid
          double grid_density = options.get_option("grid_density",1); //how many times denser will the grid be;
          int kx_break_count = ((int) (number_of_kx_points*dense_grid_ratio + 0.5))*grid_density;
          int ky_break_count = ((int) (number_of_ky_points*dense_grid_ratio + 0.5))*grid_density;
          int kz_break_count = ((int) (number_of_kz_points*dense_grid_ratio + 0.5))*grid_density;

          //These loops go over all three possible k directions, even nonexisting in our device, but due to "continue" statement the nonexisting will be skipped.
          int custom_number_of_kx_points = ((int) (number_of_kx_points*dense_grid_ratio + 0.5))*grid_density + (int) (number_of_kx_points*(1-dense_grid_ratio) + 0.5);
          int custom_number_of_ky_points = ((int)(number_of_ky_points*dense_grid_ratio + 0.5))*grid_density + (int)(number_of_ky_points*(1-dense_grid_ratio) + 0.5);
          int custom_number_of_kz_points = ((int)(number_of_kz_points*dense_grid_ratio + 0.5))*grid_density + (int)(number_of_kz_points*(1-dense_grid_ratio) + 0.5);
          int temp_size = 0;
          if(rec_lattice.size()==1 ||  (k_rotational_symmetry && rec_lattice.size()==2))
            temp_size = custom_number_of_kx_points;
          else if(rec_lattice.size()==2)
            temp_size = custom_number_of_kx_points*custom_number_of_ky_points;
          else if(rec_lattice.size()==3)
            temp_size = custom_number_of_kx_points*custom_number_of_ky_points*custom_number_of_kz_points;

          vector<double> tmp(rec_lattice.size());
          vector<vector<double> > x_temp(temp_size, tmp);

          for (int kx = 0; kx < custom_number_of_kx_points; kx++)
          {
            if (kx < kx_break_count)
            {
              tmp[0] = kx * (kxmax-kxmin)/(number_of_kx_points-1.0)/grid_density;
              if(rec_lattice.size() == 1 || (k_rotational_symmetry && rec_lattice.size()==2))
              {
                x_temp[kx]=tmp; //store grid into x_temp
                msg << "kx loop adding point (" << tmp[0] << ")..." << endl;
                continue;
              }
            }
            else
            {
              tmp[0] = (kx-kx_break_count) * (kxmax-kxmin)/(number_of_kx_points-1.0) + kx_break_count*(kxmax-kxmin)/(number_of_kx_points-1.0)/grid_density;
              if(rec_lattice.size() == 1 || (k_rotational_symmetry && rec_lattice.size()==2))
              {
                x_temp[kx] = tmp;
                msg << "adding point (" << tmp[0] << ")..." << endl;
                continue;
              }
            }
            for (int ky=0; ky < custom_number_of_ky_points; ky++)
            {
              if (ky < ky_break_count)
              {
                tmp[1] = ky * (kymax-kymin)/(number_of_ky_points-1.0)/grid_density;
                if (rec_lattice.size() == 2)
                {
                  x_temp[kx*custom_number_of_ky_points+ky]=tmp; //store grid into x_temp
                  msg << "adding point (" << tmp[0] << "," << tmp[1] << ")..." << endl;
                  continue;
                }
              }
              else
              {
                tmp[1] = (ky-ky_break_count) * (kymax-kymin)/(number_of_ky_points-1.0) + ky_break_count*(kymax-kymin)/(number_of_ky_points-1.0)/grid_density;
                if (rec_lattice.size() == 2)
                {
                  x_temp[kx*custom_number_of_ky_points+ky] = tmp;
                  msg << "adding point (" << tmp[0] << "," << tmp[1] << ")..." << endl;
                  continue;
                }
              }
              for (int kz=0; kz < custom_number_of_kz_points; kz++)
              {
                if (kz < kz_break_count)
                {
                  tmp[2] = kz * (kzmax-kzmin)/(number_of_kz_points-1.0)/grid_density;
                  if (rec_lattice.size() == 3)
                  {
                    x_temp[(kx*custom_number_of_ky_points+ky)*custom_number_of_kz_points+kz]=tmp; //store grid into x_temp
                    msg << "adding point (" << tmp[0] << "," << tmp[1] << "," << tmp[2] << ")..." << endl;
                    continue;
                  }
                }
                else
                {
                  tmp[2] = (kz-kz_break_count) * (kzmax-kzmin)/(number_of_kz_points-1.0) + kz_break_count*(kzmax-kzmin)/(number_of_kz_points-1.0)/grid_density;
                  if (rec_lattice.size() == 3)
                  {
                    x_temp[(kx*custom_number_of_ky_points+ky)*custom_number_of_kz_points+kz] = tmp;
                    msg << "adding point (" << tmp[0] << "," << tmp[1] << "," << tmp[2] << ")..." << endl;
                    continue;
                  }
                }
              }
            }
          }
          x = x_temp;
        }
        if ((basis=="reciprocal")&!(options.check_option("K_points_file") ) )
        {
          int n = x.size();

          for (int i=0; i<n; i++)
          {
            vector<double> k_pointV(3,0.0);
            for (unsigned int i1=0; i1<rec_lattice.size(); i1++)
            {
              for (unsigned int i2=0; i2<3; i2++)
              {
                k_pointV[i2] += x[i][i1]*rec_lattice[i1][i2];
              }
            }
            kspace->add_point(k_pointV);
            //msg << "added k=(" << k_pointV[0] << "," << k_pointV[1] << "," << k_pointV[2] << ")\n";
          }
        }
        //define K mesh from file
        else if(options.check_option("K_points_file"))
        {
          std::string Kfile = options.get_option("K_points_file",std::string(""));
          vector< vector<double> > k_point;
          //vector<double> weightk;
          Compute_k_mesh_from_file(Kfile, k_point, k_weight_file);
          //for( unsigned i=0; i<k_weight_file.size(); i++)
          //kspace->add_point(k_pointV[i]);

          for (unsigned int i=0; i<k_weight_file.size(); i++)
          {
            vector<double> k_pointV(3,0.0);
            for (unsigned int i1=0; i1<rec_lattice.size(); i1++)
            {
              for (unsigned int i2=0; i2<3; i2++)
              {
                k_pointV[i2] += k_point[i][i1]*rec_lattice[i1][i2];
              }
            }
            kspace->add_point(k_pointV);
          }
        }
        else
        {
          NEMO_ASSERT(rec_lattice.size()==3, "[Schroedinger(\""+get_name()+"\")] cartesian basis only works for 3D-periodicity.");
          vector<double> tmp(3);
          for (int kx=0; kx<number_of_kx_points; kx++)
          {
            tmp[0] = kx * (kxmax-kxmin)/(number_of_kx_points-1.0) + kxmin;
            for (int ky=0; ky<number_of_ky_points; ky++)
            {
              tmp[1] = ky * (kymax-kymin)/(number_of_ky_points-1.0) + kymin;
              for (int kz=0; kz<number_of_kz_points; kz++)
              {
                tmp[2] = kz * (kzmax-kzmin)/(number_of_kz_points-1.0) + kzmin;
                msg << "adding point (" << tmp[0] << "," << tmp[1] << "," << tmp[2] << ")..." << endl;
                kspace->add_point(tmp);
              }
            }
          }
        }

      }

#else
      NEMO_EXCEPTION("[Schroedinger(\""+get_name()+"\")] NEMO5 was compiled without libmesh.");
#endif
    } // number_of_k_points!=0

    // just for fun
    if (rec_lattice.size()>0)
    {
      kspace->calculate_brillouin_zone(rec_lattice);
    }
  }
  else if(assemble_H) //just one k_point
  {
    options.get_option("k_points",k_pointsV);
    NEMO_ASSERT(k_pointsV.size()==1, "[Schroedinger(\""+get_name()+"\")]::create_k_space_object: k_points parameter has to contain exactly one k-point");

    /*  if(options.check_option("number_of_nodes"))
      {
         options.get_option("number_of_nodes", number_of_nodesV);
         NEMO_ASSERT((number_of_nodesV.size()==k_pointsV.size()-1)||(number_of_nodesV.size()==1 && number_of_nodesV[0]==1 && k_pointsV.size()==1), "need to have 1 number_of_nodes entry for each k-segment.\n");
         if (number_of_nodesV.size()==1 && number_of_nodesV[0]==1 && k_pointsV.size()==1)
     number_of_nodesV.clear();
      }
      else if (k_pointsV.size()!=1)
      {
         msg << "[Schroedinger(\""+get_name()+"\")] did not find parameter number_of_nodes in input deck - assume  points per k-segment.\n";
         number_of_nodesV.resize(k_pointsV.size()-1, 11);
      }*/
    kspace->set_mesh_boundary(k_pointsV);
    kspace->set_segmental_mesh(true);

    for (unsigned int ii=0; ii<number_of_nodesV.size(); ii++)
      kspace->set_segmental_num_points(ii, number_of_nodesV[ii]);
    kspace->calculate_mesh();

    // transform k_space into cartesian coordinates (if necessary)
    string k_space_basis = options.get_option("k_space_basis",string("reciprocal"));
    if (k_space_basis!="cartesian")
    {
      vector <vector<double> > transformation_matrix;
      unsigned int number_of_rec_vectors=get_simulation_domain()->get_reciprocal_vectors().size();

      transformation_matrix.resize(3);
      for (unsigned int i=0; i<3; i++)
      {
        transformation_matrix[i].resize(number_of_rec_vectors,0.0);
        for (unsigned int j=0; j<number_of_rec_vectors; j++)

          transformation_matrix[i][j]=
              get_simulation_domain()->get_reciprocal_vectors()[j][i];
        /*
              // rvedula if statements to correct k-space for constant strain
          if(constant_strain)
            transformation_matrix[i][j]=
              get_simulation_domain()->get_reciprocal_vectors_strained()[j][i]; //transpose transformation==inverse transformation for orthonormal vectors
          else
            transformation_matrix[i][j]=
              get_simulation_domain()->get_reciprocal_vectors()[j][i]; //transpose transformation==inverse transformation for orthonormal vectors
         */
      }
      msg << "K-space object Transformation matrix" << transformation_matrix[0][0]<<transformation_matrix[0][1]<<transformation_matrix[0][2]<<"\n";
      kspace->transform_basis(transformation_matrix);
    }


  }
  else // default setting for k_space object
  {
    options.get_option("k_points",k_pointsV);
    if (k_pointsV.size()!=1) msg <<"Schroedinger::create_k_space_object: using k_point=(0,0,0) as default\n";
    vector<double> tempV(3,0.0);
    k_pointsV.resize(1,tempV);
    k_pointsV[0]=tempV;
    kspace->set_mesh_boundary(k_pointsV);
    kspace->set_segmental_mesh(true);
    kspace->calculate_mesh();
  }

  //Roza Kotlyar: 3/2/2011
  //the conversion factor for k integration to get integrated density in 1/cm^dim
  vector< vector<double> >  rec_lattice;
  rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
  /*
  if (constant_strain)
  {
    rec_lattice = get_simulation_domain()->get_reciprocal_vectors_strained();
  }
  else
  {
    rec_lattice = get_simulation_domain()->get_reciprocal_vectors();
  }
   */
  if (rec_lattice.size()>0 && rec_lattice.size()<3)
  {
    do_electron_integrated_den = options.get_option("calculate_integrated_electron_density", false);
    do_hole_integrated_den = options.get_option("calculate_integrated_hole_density", false);
    double smallnumber=1.e-6;
    rspace_multfact=1.0;

    if (do_electron_integrated_den || do_hole_integrated_den)
    {
      for (unsigned int ic=0; ic<rec_lattice.size(); ic++)
      {
        for (unsigned int ic1=0; ic1<3; ic1++)
        {
          if (fabs(rec_lattice[ic][ic1])>smallnumber)
          {
            //getting essentially a_x * a_y  in dim=2
            //getting essentially a_x  in dim=1
            //where a's are structure (wire, well) unit cell lattice constants
            rspace_multfact*=(2.0*NemoMath::pi/fabs(rec_lattice[ic][ic1]))*1.e-7; //a in cm from nm
          }
        }
      }
    }
  }


  //k_space_dimensionality = get_simulation_domain()->get_reciprocal_vectors().size();

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::read_job_list(void)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::read_job_list";
  NemoUtils::tic(tic_toc_prefix);



  // -----------------------------------
  // read in the parameters and job list
  // -----------------------------------
  vector<string> job_list;
  options.get_option("job_list",job_list);

  if (job_list.size()==0)
  {
    msg << "[Schroedinger(\""+get_name()+"\")] nothing to be done; quit Schroedinger\n";
    NemoUtils::toc(tic_toc_prefix);
    return;
  }

  for (unsigned int loop_jobs=0; loop_jobs<job_list.size(); loop_jobs++)
  {
    string job=job_list[loop_jobs];

    if(job=="assemble_H")
    {
      assemble_H=true;
    }
    else if (job=="output_H_all_k")
    {
      output_H_all_k=true;
    }
    else if (job=="passivate_H")
    {
      assemble_H=true;
      passivate_H=true;
    }
    else if (job=="calculate_band_structure")
    {
      assemble_H=true;

      calculate_band_structure=true;
    }
    else if (job=="electron_density")
    {
      assemble_H=true;
      electron_density=true;
    }
    else if (job=="hole_density")
    {
      assemble_H=true;
      hole_density=true;
    }
    else if (job == "scattering_probability")
    {
      assemble_H=true;
      electron_density=true;
      scattering_probability = true;
    }
    else if (job=="derivative_electron_density_over_potential")
    {
      assemble_H=true;
      electron_density=true;
      derive_electron_density_over_potential=true;
    }
    else if (job=="derivative_hole_density_over_potential")
    {
      assemble_H=true;
      hole_density=true;
      derive_hole_density_over_potential=true;
    }
    else if (job=="ion_density")
    {
      ion_density=true;
      determine_ion_density();
    }
    else if (job=="include_strain_H")
    {
      assemble_H=true;
      include_strain_H=true;
    }
    else if (job=="include_shear_strain_H")
    {
      assemble_H=true;
      include_strain_H=true;
      include_shear_strain_H=true;
    }
    else if (job=="constant_strain")
    {
      constant_strain=true;
    }
    else if (job=="DOS")
    {
      density_of_states=true;
      if (options.get_option("project_DOS", false))
        project_density_of_states = true;
      //M.P. we need it because if we don't do it will produce a garbage
      //spatial_density_of_states=true;
    }
    else if (job=="DOS_spatial")
    {
      spatial_density_of_states=true;
    }
    else if (job=="energy_gradient")
    {
      energy_gradient=true;
    }
    else if(job=="block_matrix_vector")
    {
      block_matrix_vector_mult_time=true;
    }
    else if (job=="matrix_vector")
    {
      matrix_vector_mult_time = true;
    }
    else if (job == "spin")
    {
      spin_projection = true;
    }
    else if( job == "calculate_Fermi_level")
    {
      calculate_Fermi_level = true;
    }
    else if( job =="combine_H")
    {
      combine_H = true;
    }
    else
    {
      throw invalid_argument("[Schroedinger(\""+get_name()+"\")] unknown job: " + job + "\n");
    }
  }

  // consistency checks
  if (calculate_band_structure&&electron_density)
    throw invalid_argument("[Schroedinger(\""+get_name()
        +"\"):read_job_list] \"calculate_band_structure=true\" and \"electron_density=true\" are in contradiction!\n");
  if (calculate_band_structure&&hole_density)
    throw invalid_argument("[Schroedinger(\""+get_name()
        +"\"):read_job_list] \"calculate_band_structure=true\" and \"hole_density=true\" are in contradiction!\n");
  if (calculate_band_structure && (density_of_states || spatial_density_of_states))
    throw invalid_argument("[Schroedinger(\""+get_name()
        +"\"):read_job_list] \"calculate_band_structure=true\" and \"(spatial_)density_of_states=true\" are in contradiction!\n");
  if (electron_current && !electron_density)
    throw invalid_argument("[Schroedinger(\""+get_name()
        +"\"):output_list] \"electron_current=true\" and \"electron_density=false\" are in contradiction!\n");
  if (hole_current && !hole_density)
    throw invalid_argument("[Schroedinger(\""+get_name()+"\"):output_list] \"hole_current=true\" and \"hole_density=false\" are in contradiction!\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::output_density(const string& density_type)
{
  if (has_output(density_type+"_VTK")) print_density(get_name()+"_"+density_type+"_vtk.dat", density_type, "VTK");
  if (has_output(density_type+"_VTP")) print_density(get_name()+"_"+density_type+"_vtk.dat", density_type, "VTP");
  if (has_output(density_type+"_VTU")) print_density(get_name()+"_"+density_type+"_vtk.dat", density_type, "VTU");
  if (has_output(density_type+"_DX"))  print_density(get_name()+"_"+density_type+"_dx.dat", density_type, "DX");
  if (has_output(density_type))        print_density(get_name()+"_"+density_type+"_xyz.dat", density_type);
}

void Schroedinger::output_hamiltonian(bool binary)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::output_hamiltonian";
  string ext = ".m";
  if (binary)
	  ext = ".bin";
  string temp = get_name()+get_output_suffix()+"_Ham" + ext;
  NEMO_ASSERT(hamiltonian!=NULL,tic_toc_prefix+"cannot assemble Hamiltonian, since it is NULL."+
      +" Please make sure to run \""+get_name()+"\" before its output\n");
  hamiltonian->assemble();
  if (binary)
    hamiltonian->save_to_binary_file (temp.c_str());
  else
    hamiltonian->save_to_matlab_file (temp.c_str());
  if (overlap != NULL)
  {
    temp = get_name()+get_output_suffix()+"_S" + ext;
    if (binary)
      overlap->save_to_binary_file (temp.c_str());
    else
      overlap->save_to_matlab_file (temp.c_str());
  }
}

void Schroedinger::output_energies(void)
{
  // output k-points and energies in a simulation where density is calculated
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::output_energies";
  string temp = get_name()+ get_output_suffix() + "_energies.dat";
  msg << "[Schroedinger(\""+get_name()+"\")] saving k-points and energies to " << temp << "...\n";
  ofstream file(temp.c_str(), std::ios_base::out | std::ios_base::trunc);
  file.precision(output_precision);
  file << "% first 3 columns are k-vector coordinates, followed be energy eigenvalues\n";
  for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
  {
    const NemoMeshPoint& ki = kspace->get_point(ii);
    file << ki[0] << " " << ki[1] << " " << ki[2] << "\t";
    for (unsigned int jj=0; jj<EnergyV[ii].size(); jj++)
    {
      file << EnergyV[ii][jj] << "\t";
    }
    file << "\n";
  }
  file.close();
}

void Schroedinger::output_dos_fermi(const string& file_name, const std::map<double,double>& DOS_dfermi)
{
  ofstream file;
  file.open (file_name.c_str());
  std::map<double,double>::const_iterator c_it=DOS_dfermi.begin();
  for(; c_it!=DOS_dfermi.end(); ++c_it)
  {
    file << c_it->first << "\t  \t" << c_it->second << "\n";
  }
  file.close();
}

void Schroedinger::do_output(void)
{
#ifdef USE_PAPI
  MPI_Comm krpcomm = MPI_COMM_WORLD;
  krp_rpt_init_(&krp_rank, &krpcomm, &krp_hw_counters, &krp_rcy, &krp_rus, &krp_ucy, &krp_uus);
#endif

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::do_output";
  NemoUtils::tic(tic_toc_prefix);
  // geometry_replica = index which replica of the geometry the current process has
  // suppose there is no geometry parallelization and you're running the simulation using N CPUs (and Nk>=N).
  // then there will be N geometry replica
  if (get_simulation_domain()->get_geometry_replica() != 0  || output_list.size()==0)
  {
    NemoUtils::toc(tic_toc_prefix);
    return;
  }


  int my_rank;
  MPI_Comm_rank(get_simulation_domain()->get_communicator(), &my_rank);
  int num_geom_ranks;
  MPI_Comm_size(get_simulation_domain()->get_communicator(), &num_geom_ranks);

  bool eigfun_output    =     has_output("eigenfunctions")      || has_output("eigenfunctions_VTK")
                                  || has_output("eigenfunctions_VTP")   || has_output("eigenfunctions_VTU")
                    || has_output("eigenfunctions_DX")   || has_output("eigenfunctions_XYZ")
                    || has_output("eigenfunctions_Silo") || has_output("eigenfunctions_1D") || has_output("eigenfunctions_Point3D");

  bool eigfun_k0_output =     has_output("eigenfunctions_k0")      || has_output("eigenfunctions_VTK_k0")
                                  || has_output("eigenfunctions_VTP_k0")   || has_output("eigenfunctions_VTU_k0")
                    || has_output("eigenfunctions_DX_k0")   || has_output("eigenfunctions_XYZ_k0")
                    || has_output("eigenfunctions_Silo_k0") || has_output("eigenfunctions_1D_k0" ) || has_output("eigenfunctions_Point3D");

  bool hole_current_out = output_list.find("hole_current")!=output_list.end();
  bool electron_current_out = output_list.find("electron_current")!=output_list.end();
  bool energies_out = output_list.find("energies")!=output_list.end();
  bool relation_xyz_H_index = output_list.find("relation_xyz_H_index") != output_list.end();
  bool eigfun_out = eigfun_output || eigfun_k0_output;

  bool density_out = hole_density||electron_density||derive_electron_density_over_potential||derive_hole_density_over_potential||density_of_states;
  if(assemble_H)
    if (has_output("Hamiltonian_binary"))
      output_hamiltonian( true );
    if (has_output("Hamiltonian"))
      output_hamiltonian( false );
  if (density_out)
  {
    if(my_rank==0){
      if (electron_density)
        output_density("electron_density");
      if (hole_density)
        output_density("hole_density");
      if (derive_electron_density_over_potential)
        output_density("derivative_electron_density_over_potential");
      if (derive_hole_density_over_potential)
        output_density("derivative_hole_density_over_potential");
      if (ion_density)
        output_density("ion_density");
      if (hole_current_out)
    	output_current_modes(HOLE, _hole_threshold_energy );
      if (electron_current_out)
    	output_current_modes(ELECTRON, _electron_threshold_energy );
      if (energies_out)
        output_energies();
    }
    if (eigfun_out)
    {
      if (eigfun_k0_output && NemoMath::vector_norm_3d(&this->k_vector[0])>1e-10)
      {
        msg << "[Schroedinger(\""+get_name()+"\")] solving k=0 one more time to get eigenfunctions...\n";
        vector<double> k0(3, 0.0);
        vector<double> tmp_energies;
        this->solve_tb_single_kpoint(k0, tmp_energies);
      }

      if (my_rank==0) boost::filesystem::create_directory("eigenfunctions");
      MPI_Barrier(get_simulation_domain()->get_communicator());

      const vector<cplx>* evals = solver->get_eigenvalues();
      unsigned int num_evals = evals->size();
      msg << "[Schroedinger(\""+get_name()+"\")] saving " << num_evals << " eigenfunctions to folder eigenfunctions/ ...\n";
      NEMO_ASSERT( num_evals==vector_to_energy_map.size(),
          "[Schroedinger(\""+get_name()+"\"):do_output] number_of_eigenvalues does not equal size of energy_number_map\n");

      msg << "[Schroedinger(\""+get_name()+"\")] eigenvalues: ";
      vector<double> energies_k0(num_evals, -888.888);

      for (unsigned int i=0; i<num_evals; i++)
      {
        unsigned int idx = vector_to_energy_map[i];
        if (eigfun_k0_output)
        {
          // how many eigenvalues are smaller?
          idx = 0;
          for (unsigned int jj=0; jj<num_evals; jj++)
          {
            if ((*evals)[jj].real() < (*evals)[i].real()) idx++;
          }
        }
        msg << i << "=" << (*evals)[idx] << "   ";
        energies_k0[idx] = (*evals)[i].real();
        output_eigenfunction(idx, (num_geom_ranks==1));

      }
      msg << "\n";

      if (my_rank== (int)0)
      {
        string energies_k0_filename = get_name() +get_output_suffix()+ "_energies_k0.dat";
        ofstream energies_k0_file(energies_k0_filename.c_str());
        energies_k0_file << "% eigenenergies at k=0:\n";
        for (unsigned int i=0; i<num_evals; i++)
        {
          energies_k0_file << std::setprecision(10) << energies_k0[i] << "    ";
        }
        energies_k0_file << "\n";
        energies_k0_file.close();
      }
    } // eigenfunction output
    if(scattering_probability && has_output("scattering_probability"))
    {

      if(my_rank==0)
      {
        string scatteringrate_filename = get_name() + "_scatteringrates.dat";
        ofstream scatteringrate_file(scatteringrate_filename.c_str());
        std::map<unsigned int, double>::const_iterator c_it=scattering_rates.begin();
        for(; c_it!=scattering_rates.end(); ++c_it)
        {
          scatteringrate_file <<
              "state # "<<c_it->first<<":\t[("<<subset_of_momenta[c_it->first][0]<<","<<subset_of_momenta[c_it->first][1]<<","<<subset_of_momenta[c_it->first][2]<<"), ";
          scatteringrate_file << subset_of_eigenvalues[c_it->first] << "]\t" << c_it->second<<"\n";
        }
        scatteringrate_file.close();
      }
    }
  } // density jobs
  else if (calculate_band_structure || output_H_all_k)
  {
    if (has_output("k-points") && my_rank==0) // <ss Mar 3 2011> my_rank==0 is new
    {
      string temp = get_name()+"_k_points.dat";
      ofstream kfile(temp.c_str(), std::ios_base::out | std::ios_base::trunc);
      temp = get_name()+"_k_distance.dat";
      ofstream kdfile(temp.c_str(), std::ios_base::out | std::ios_base::trunc);

      for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
      {
        const NemoMeshPoint& ki = kspace->get_point(ii);
        kfile << ki[0] << "," << ki[1] << "," << ki[2] << "\n";
        kdfile<< k_distanceV[ii]<<"\n";
      }
      kfile.close();
      kdfile.close();
    }
    //RK for bandstructure optimization
    if (has_output("k-points-scaled-precise") && my_rank==0) //
    {
      this->get_lattice_constants(lattice_a);
      vector< vector<double> > rec_vecs;

      //                if (constant_strain) { //just for information
      //                       rec_vecs = this->get_simulation_domain()->get_reciprocal_vectors_strained();
      //                } else {
      rec_vecs = this->get_simulation_domain()->get_reciprocal_vectors();
      //                }
      unsigned int number_of_rec_vectors=rec_vecs.size();

      double ksc[3];
      int places=15;
      string temp = get_name()+"_k_points_scaled_precise.dat";
      ofstream kfile(temp.c_str(), std::ios_base::out | std::ios_base::trunc);
      kfile.precision(places);
      //--------------------------------------------------------
      //cout<<"RECIPROCAL VECTORS:"<<endl;
      for (unsigned int ic=0; ic<number_of_rec_vectors; ic++) {
        //cout<<ic;
        for (unsigned int ic1=0; ic1<3; ic1++) {
          //cout<<" "<<rec_vecs[ic][ic1];
        }
        //cout<<endl;
      }
      //--------------------------------------------------------
      for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
      {
        const NemoMeshPoint & ki = kspace->get_point(ii);
        for (int ic1=0; ic1<3; ic1++) {
          ksc[ic1]=ki[ic1]/(2.0*NemoMath::pi/fabs(lattice_a[ic1]));
        }
        //--------------------------------------------------------
        kfile << ii<< " " <<ksc[0] << " " << ksc[1] << " " << ksc[2] << "\n";
      }
      kfile.close();
    }
    //--------------------------------------------------------
    //RK target reading and calculating
    if (has_output("do_target_postprocessing") && my_rank==0) //
    {
      do_target_postprocessing();
    }
    //--------------------------------------------------------

    if (has_output("energies") && my_rank==0) // <ss Mar 3 2011> my_rank==0 is new
    {
      string temp = get_name()+get_output_suffix()+"_energies.dat";
      ofstream Efile(temp.c_str(), std::ios_base::out | std::ios_base::trunc);
      Efile.precision(output_precision);
      for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
      {
        for (unsigned int jj=0; jj<EnergyV[ii].size(); jj++)
        {
          Efile << EnergyV[ii][jj] << "\t";
        }
        Efile << "\n";
      }
      Efile.close();
    }

    if(options.get_option("compute_edges_masses",false))
    {
      string temp = get_name()+get_output_suffix()+"_effective_masses.dat";
      ofstream Efile(temp.c_str(), std::ios_base::out | std::ios_base::trunc);
      if( k_space_dimensionality  == 3)
      {
        for (unsigned int i=0 ; i< eff_mass_string_info.size(); i++)
        {
          Efile<<eff_mass_string_info[i];
        }
      }
      else
      {
        Efile <<"Band Number\t Valley Number \t Direction \t Kposition/pi \t Effective Mass \t Energy \n";
        for (unsigned int i = 0; i <eff_masses_info.size() ; i++)
        {
          Efile.precision(output_precision);
          Efile <<eff_masses_info[i]->band_number<<"\t|"<<eff_masses_info[i]->no_valley<<"\t|"<<"<"<<eff_masses_info[i]->mdir[0]<<","<<eff_masses_info[i]->mdir[1]<<","<<eff_masses_info[i]->mdir[2]
                                                                                                      <<">\t|["<<eff_masses_info[i]->kpos[0]<<","<<eff_masses_info[i]->kpos[1]<<","<<eff_masses_info[i]->kpos[2]<<"]\t|"
                                                                                                      <<eff_masses_info[i]->eff_mass<<"\t|"<<eff_masses_info[i]->Ene<<endl;
        }
      }
      Efile.close();
    }

    if (spin_projection)
      if (has_output("spin") && my_rank==0)
      {
        string temp = get_name()+"_spin_projection.dat";
        ofstream Spinfile(temp.c_str(), std::ios_base::out | std::ios_base::trunc);
        Spinfile.precision(output_precision);
        for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
        {
          for (unsigned int jj=0; jj<spin_projection_matrix[ii].size(); jj++)
          {
            for (short p = 0; p < 3; p++)
            {
              Spinfile << real(spin_projection_matrix[ii][jj][p]);

              double im = imag( spin_projection_matrix[ii][jj][p] );

              if ( fabs(im) > 1e-12 )
              {
                Spinfile <<   std::showpos << im <<"i";

              }
              Spinfile<< "\t";
            }
            Spinfile << "\t";
          }
          Spinfile << "\n";
        }
        Spinfile.close();
      }


    if (eigfun_output || eigfun_k0_output)
    {
      if (eigfun_k0_output)
      {
        vector<double> k0(3, 0.0);
        options.get_option("k0", k0);

        if (k0 != k_vector)
        {
          vector<double> tmp_energies;
          this->solve_tb_single_kpoint(k0, tmp_energies);
        }
      }

      /*
      if (eigfun_k0_output && NemoMath::vector_norm_3d(&this->k_vector[0])>1e-10) {
        msg << "[Schroedinger] solving k=0 one more time to get eigenfunctions...\n";
        vector<double> k0(3, 0.0);
        vector<double> tmp_energies;
        this->solve_tb_single_kpoint(k0, tmp_energies);
      }
       */

      if (my_rank==0) boost::filesystem::create_directory("eigenfunctions");
      MPI_Barrier(get_simulation_domain()->get_communicator());

      const vector<cplx>* evals  = solver->get_eigenvalues();
      unsigned int num_evals = evals->size();
      NEMO_ASSERT( num_evals==vector_to_energy_map.size(),
          "[Schroedinger(\""+get_name()+"\")] do_output number_of_eigenvalues does not equal size of energy_number_map\n");

      msg << "[Schroedinger(\""+get_name()+"\")] eigenvalues: ";
      for (unsigned int i=0; i<num_evals; i++)
      {
        unsigned int idx = vector_to_energy_map[i];

        /* M.P. I think this 'if'  block is not needed because vector_to_energy_map is tested now */
        if (eigfun_k0_output)
        {
          // how many eigenvalues are smaller?
          idx = 0;
          for (unsigned int jj=0; jj<num_evals; jj++)
          {
            if ((*evals)[jj].real() < (*evals)[i].real()) idx++;
          }
        }
        msg << i << "=" << (*evals)[idx] << "   ";
        output_eigenfunction(idx, (num_geom_ranks==1));
      }
      msg << "\n";
    } // eigenfunction output
  } // calculate_band_structure

  if (relation_xyz_H_index && my_rank == 0)
    output_relation_xyz_H_index();
   
  if (has_output("band_diagram")) print_band_diagram(get_name() + "_band_diagram");

  if(has_output("DOS_dfermi")) output_dos_fermi(this->get_name() + "DOS_dfermi.dat", DOS_times_dfermi);

  if(has_output("EmEf_DOS_dfermi")) output_dos_fermi(this->get_name() + "DOS_dfermi.dat", EmEf_DOS_times_dfermi);

  if (has_output("DOS") && (density_of_states) && my_rank==0 &&!(project_density_of_states)) // <ss Mar 3 2011> my_rank==0 is new
  {
    // we want to output density in the units 1/(eV*cm^n), where n is the periodic space dimensionality
    double volume_factor = 1.0;
    const Domain* domain = get_simulation_domain();
    int n = domain->get_reciprocal_vectors().size();

    if (n > 0)
      volume_factor = 1.0 /(domain->return_periodic_space_volume() * std::pow(1e-7, n));

    string filename = this->get_name() + "_DOS.dat";
    ofstream file;
    file.open (filename.c_str());
    file << "% energy[eV] density of states [1/(eV*cm^n)]\n";
    int n_energy = DOS_energy_grid.size();
    for (int i = 0; i < n_energy; i++)
    {
      file << std::setw(20) << std::setprecision(10) << DOS_energy_grid[i] <<"        "
          << DOS[i]* volume_factor  << "\n";
    }
    file.close();
  }
  //test
  if (project_density_of_states)
  {
    string filename = this->get_name() + "_PDOS.dat";
    ofstream file;
    file.open (filename.c_str());
    file << "% energy[eV] density of states   [1/(eV)] \t S_contribution \t P_contribution \t D_contribution \n";
    int n_energy = DOS_energy_grid.size();
    for (int i = 0; i < n_energy; i++)
    {
      file << std::setw(20) << std::setprecision(10) << DOS_energy_grid[i] <<"\t"
          << PDOS[3][i] <<"\t" << PDOS[0][i] <<"\t"<< PDOS[1][i] <<"\t"
          << PDOS[2][i] <<"\n";
    }
    file.close();
  }

  if (has_output("spatial_DOS") && (spatial_density_of_states) && my_rank==0)
  {
    //spatially resolved density of states
    //double volume_factor = 1.0;
    const AtomisticDomain*  domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    const AtomicStructure&  atoms   = domain->get_atoms();
    //    int n = domain->get_reciprocal_vectors().size();

    //if (n > 0)
    //  volume_factor = 1.0 /(domain->return_periodic_space_volume() * std::pow(1e-7, n));

    string filename = this->get_name() + "spatial_DOS.dat";
    ofstream file;
    file.open (filename.c_str());
    file << "% energy[eV], x[nm], y[nm], z[nm], density of states [1/(eV*cm^n)]\n";

    int n_energy = DOS_energy_grid.size();
    for (int i = 0; i < n_energy; i++)
    {
      double energy = DOS_energy_grid[i];
      const map<unsigned int, double>& dos_at_energy = spatially_resolved_DOS[i];
      for (map<unsigned int, double>::const_iterator it = dos_at_energy.begin();
          it != dos_at_energy.end(); ++it)
      {
        unsigned int atom_id = it->first;

        const AtomStructNode& nd = atoms.lattice().find(atom_id)->second;

        double dos_value = it->second;

        file << energy <<"   " << nd.position[0] << "  " <<
            nd.position[1] << "   " << nd.position[2] <<"   " << dos_value << "\n";
      }
    }
    file.close();
  }


  if (has_output("single_eigenfunction"))
  {
    vector<double> k0(3, 0.0);
    options.get_option("output_k_point", k0);
    msg << "(" << k0[0] << "  " << k0[1] << "   " << k0[2] << ")\n";

    string k_space_basis = options.get_option("k_space_basis",string("reciprocal"));

    if (k_space_basis!="cartesian")
    {
      vector <vector<double> > transformation_matrix;
      unsigned int number_of_rec_vectors=get_simulation_domain()->get_reciprocal_vectors().size();
      vector<double> k0_temp = k0;
      for (unsigned int j=0; j<number_of_rec_vectors; j++)
        for (short i = 0; i < 3; i++)
          k0[i] = get_simulation_domain()->get_reciprocal_vectors()[j][i]*k0_temp[j];
    }

    msg << "(" << k0[0] << "  " << k0[1] << "   " << k0[2] << ")\n";

    unsigned int eigenstate_number = 0;
    double eigenstate_energy = options.get_option("output_eigenstate_energy", 0.0);
    vector<double> tmp_energies;
    solve_tb_single_kpoint(k0, tmp_energies);
    int m = tmp_energies.size();

    for (int i = 1; i < m; i++)
    {
      if ( fabs(tmp_energies[i] - eigenstate_energy) < fabs(tmp_energies[eigenstate_number] - eigenstate_energy) )
      {
        eigenstate_number = i;
      }
    }

    if (my_rank==0) boost::filesystem::create_directory("eigenfunctions");
    MPI_Barrier(get_simulation_domain()->get_communicator());
    const vector<cplx>* evals  = solver->get_eigenvalues();
    msg << get_name() << " eigenstate number : " << eigenstate_number << "\n";
    unsigned int idx = vector_to_energy_map[eigenstate_number];
    msg << "Eigenstate number " << eigenstate_number << "=" << (*evals)[idx] << "   ";
    output_eigenfunction(idx, (num_geom_ranks==1));
  }
  //----------------------------------------------------------------------------
  //Roza Kotlyar: 3/2/2011
  //to output shifts for later restart
  bool do_output_nlp_restartinfo  = options.get_option("output_shift_subbandbased_for_restart",false);

  if(do_output_nlp_restartinfo && my_rank==0)
  {
    string suffix1=".nlp_restartinfo";
    string fileoutname=get_name() + suffix1;
    msg << "[Schroedinger(\""+get_name()+"\")] outputting restart info to a file " << fileoutname <<  ".\n";
    write_restartinfooutputs_in_af(fileoutname);
  }

  //----------------------------------------------------------------------------
  // M.P. I do not want any barrier here
  //  MPI_Barrier(holder.total_communicator);
  msg << "[Schroedinger(\""+get_name()+"\")::do_output] done." << endl;

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::output_eigenfunction(int eigenvalue_number, bool one_rank_output)
{
  stringstream filename;
  filename << "eigenfunctions/" << get_name().c_str() << "_f" << eigenvalue_number;
  if (one_rank_output)
  {
    if (has_output("eigenfunctions") || has_output("eigenfunctions_k0") )
      this->print_eigenfunction(filename.str(), "default", eigenvalue_number, false); // ASCII text file
    if ( has_output("eigenfunctions_1D") || has_output("eigenfunctions_1D_k0") )
      this->print_eigenfunction(filename.str(), "default", eigenvalue_number, true); // ASCII text file, 1D interpolation
    if ( has_output("eigenfunctions_VTK") || has_output("eigenfunctions_VTK_k0") )
      this->print_eigenfunction(filename.str(), "VTK", eigenvalue_number); // last index = entry in solver->get_eigenfunctions() array
    if ( has_output("eigenfunctions_VTP") || has_output("eigenfunctions_VTP_k0") )
      this->print_eigenfunction(filename.str(), "VTP", eigenvalue_number); // last index = entry in solver->get_eigenfunctions() array
    if ( has_output("eigenfunctions_VTU") || has_output("eigenfunctions_VTU_k0") )
      this->print_eigenfunction(filename.str(), "VTU", eigenvalue_number); // last index = entry in solver->get_eigenfunctions() array
    if ( has_output("eigenfunctions_DX") || has_output("eigenfunctions_DX_k0") )
      this->print_eigenfunction(filename.str(), "DX", eigenvalue_number);
    if ( has_output("eigenfunctions_XYZ") || has_output("eigenfunctions_XYZ_k0") )
      this->print_eigenfunction(filename.str(), "XYZ", eigenvalue_number);
    if( has_output("eigenfunctions_Point3D"))
      this->print_eigenfunction(filename.str(), "Point3D", eigenvalue_number);
  }
  else
  {
    if ( has_output("eigenfunctions_Silo") || has_output("eigenfunctions_Silo_k0") )
      this->print_eigenfunction(filename.str(), "Silo", eigenvalue_number);
    if ( has_output("eigenfunctions_VTK") || has_output("eigenfunctions_VTK_k0") )
      this->print_eigenfunction(filename.str(), "VTK", eigenvalue_number); // last index = entry in solver->get_eigenfunctions() array
    if ( has_output("eigenfunctions_VTP") || has_output("eigenfunctions_VTP_k0") )
      this->print_eigenfunction(filename.str(), "VTP", eigenvalue_number); // last index = entry in solver->get_eigenfunctions() array
    if ( has_output("eigenfunctions_VTU") || has_output("eigenfunctions_VTU_k0") )
      this->print_eigenfunction(filename.str(), "VTU", eigenvalue_number); // last index = entry in solver->get_eigenfunctions() array
  }
}

//----------------------------------------------------------------------------
//Roza Kotlyar: 3/2/2011
//to output shifts for later restart
void Schroedinger::write_restartinfooutputs_in_af(string& fileoutname)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::write_restartinfooutputs_in_af";
  NemoUtils::tic(tic_toc_prefix);

  int places=15;
  std::ofstream nlpaf_file(fileoutname.c_str(),std::ios_base::out);

  nlpaf_file.precision(places);



  nlpaf_file << shift_foresolver << "\n";
  nlpaf_file << rspace_multfact << "\n";
  nlpaf_file << hole_integrated_den << "\n";
  nlpaf_file << electron_integrated_den << "\n";

  if (constant_strain)
  {
    for (int i=0; i<3; i++)
    {
      for (int j=0; j<3; j++)
      {
        nlpaf_file <<"\t"<<strain_matrix[i][j];
      }
      nlpaf_file << "\n";
    }
  }
  nlpaf_file.close();

  NemoUtils::toc(tic_toc_prefix);
}



//----------------------------------------------------------------------------
double Schroedinger::get_electron_density (unsigned int atom_id) const
{
  map<unsigned int, double>::const_iterator it=electron_charge.find(atom_id);
  if(it==electron_charge.end()) throw std::runtime_error("Schroedinger::get_electron_density reached end of electron_charge\n");
  return it->second;
}


double Schroedinger::get_hole_density (unsigned int atom_id) const
{
  map<unsigned int, double>::const_iterator it=hole_charge.find(atom_id);
  if(it==hole_charge.end()) throw std::runtime_error("Schroedinger::get_hole_density reached end of hole_charge\n");
  return it->second;
}


void  Schroedinger::calculate_overlap_for_elastic_scattering(double Energy,double dE)
{

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::calculate_overlap_for_elastic_scattering";
  NemoUtils::tic(tic_toc_prefix);

  bool take_all_states_and_no_overlap=false;
  options.get_option("take_all_states_and_no_overlap",false);

  msg << "Calculating overlap for elastic scattering ...\n";
  //find k-point and eigenstaes closed to desired energy

  ofstream out;
  ofstream out_f;
  if (get_simulation_domain()->get_geometry_rank() == 0)
  {
    std::string file =   get_name() +  "_overlap.dat";
    out.open(file.c_str());


  }
  vector<unsigned int> k_index;

  vector<unsigned int> eig_index;

  unsigned int total_num_k_points = EnergyV.size();

  for (unsigned int i = 1; i < total_num_k_points - 1; i++) //do not consider boundary k-points
  {
    unsigned int num_eigs = EnergyV[i].size();

    for (unsigned int j = 0; j < num_eigs; j++)
    {
      double energy = EnergyV[i][j];

      if(!take_all_states_and_no_overlap)
      {
        if (std::fabs(Energy - energy) < dE)
          if (Energy >= energy)
            if ( (Energy < EnergyV[i][j+1]) ||  (( EnergyV[i][j+1] - EnergyV[i][j] < 1e-8 ) &&  (Energy < EnergyV[i][j+2])  ) )
              if ( (EnergyV[i+1][j] > Energy) || (EnergyV[i-1][j] > Energy) )
              {
                k_index.push_back(i);
                eig_index.push_back(j);
                out << j << "\n";
              }
      }
      else
      {
        k_index.push_back(i);
        eig_index.push_back(j);
      }
    }
  }
  //ofstream out_EV;
  //if (get_simulation_domain()->get_geometry_rank() == 0 && get_simulation_domain()->get_geometry_replica() =0)
  //{
  //  std::string file =   get_name() +  "_EV.dat";
  //  out_EV.open(file.c_str());
  //  for(unsigned int i=0;i<EnergyV.size();i++)
  //  {
  //    for(unsigned int j=0;j<EnergyV[i].size();j++)
  //      out_EV<<EnergyV[i][j] << " ";
  //    out_EV<<"\n";
  //  }
  //}


  //get eigenvectors

  unsigned int m = k_index.size();

  if ( m > 0 )
  {

    std::vector<std::vector<std::complex<double> > > eigen_states;
    eigen_states.resize(m);

    for (unsigned int i = 0; i < m; i++)
    {

      vector< double >  Energy;

      const NemoMeshPoint& point = kspace->get_point(k_index[i]);

      const vector< double >& k_point =  point.get_coords();

      //subset_of_momenta.clear();
      subset_of_momenta.push_back(point.get_coords());

      if (get_simulation_domain()->get_geometry_rank() == 0)
        out << "state # " << i << " Schroedinger index: " << full_energy_to_vector_map[k_index[i]][eig_index[i]]
                                                       << "  k  "  << k_point[0] << "   " << k_point[1] <<"   " << k_point[2]
                                                                                         << "  E = " << EnergyV[k_index[i]][eig_index[i]] <<  "\n";



      solve_tb_single_kpoint(k_point, Energy);

      //const vector<vector<complex<double> > >* eig_vectors = solver->get_eigenvectors();




      // eigen_states[i] = (*eig_vectors)[full_energy_to_vector_map[k_index[i]][eig_index[i]]];


      solver->get_eigenvector(full_energy_to_vector_map[k_index[i]][eig_index[i]]).
          get_local_part(eigen_states[i]);


      //subset_of_eigenvalues.clear();
      //subset_of_eigenvalues.push_back(Energy[energy_to_vector_map[eig_index[i]]]);
      subset_of_eigenvalues.push_back(EnergyV[k_index[i]][eig_index[i]]);

      //subset_of_eigenvectors.clear();
      //subset_of_eigenvectors.push_back(eigen_states[energy_to_vector_map[eig_index[i]]]);
      subset_of_eigenvectors.push_back(eigen_states[i]);
      if (get_simulation_domain()->get_geometry_rank() == 0)
      {
        stringstream name;
        name << get_name() + "_wavefunction_" << i ;
        /*print_eigenfunction (name.str(), "VTK",energy_to_vector_map[eig_index[i]]);
        print_eigenfunction (name.str(), "XYZ",energy_to_vector_map[eig_index[i]]);*/
        print_eigenfunction (name.str(), "VTK",full_energy_to_vector_map[k_index[i]][eig_index[i]]);
        print_eigenfunction (name.str(), "XYZ",full_energy_to_vector_map[k_index[i]][eig_index[i]]);
      }
    }


    // calculate overlap
    if(!take_all_states_and_no_overlap)
    {

      int local_vector_length = eigen_states[0].size();

      vector<double> prob(m,0.0);


      for (unsigned int i = 0; i < m; i++)
        for (unsigned int j = 0; j < m; j++)
        {

          complex<double> s(0.0, 0.0);

          complex<double> scal_prod;
          for (int k = 0; k < local_vector_length ; k++)
            s += conj(eigen_states[i][k])*eigen_states[j][k];


          MPI_Reduce ( &s, &scal_prod, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, get_simulation_domain()->get_communicator() );

          if (get_simulation_domain()->get_geometry_rank() == 0)
            out << i << "   " << j << "   " << scal_prod << "\n";

          if ( i != j)
            prob[i] += abs(scal_prod * scal_prod);
        }

      if (get_simulation_domain()->get_geometry_rank() == 0)
      {
        ofstream out_s;

        stringstream name;

        name << get_name() + "_scattering.dat";

        out_s.open(name.str().c_str());

        for (unsigned int i = 0; i < m; i++)
          out_s << prob[i] << "   ";

        out_s << "\n";

        out_s.close();
      }
    }
  }

  if (get_simulation_domain()->get_geometry_rank() == 0)
    out.close();

  NemoUtils::toc(tic_toc_prefix);

}

void Schroedinger::calculate_scattering_rate(const double weight, const std::vector<double>& k_point)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::calculate_scattering_rate";
  NemoUtils::tic(tic_toc_prefix);


  //msg<<"Schroedinger::calculate_scattering_rate\n";
  ////get the external state solver
  //NEMO_ASSERT(options.check_option("State_solver"),"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate define \"State_solver\"\n");
  //std::string state_solver_name = options.get_option("State_solver",std::string(""));
  //Simulation * state_solver = this->find_simulation(state_solver_name);
  //NEMO_ASSERT(state_solver!=NULL,"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate have not found simulation \""+state_solver_name+"\"\n");
  ////check:
  ////external state solver is defined on the same domain as this solver
  //const Domain* external_domain=state_solver->get_const_simulation_domain();
  //NEMO_ASSERT(external_domain->get_name()==get_const_simulation_domain()->get_name(),"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate"
  //  +" mismatch with domain of simulation \""+state_solver_name+"\"\n");
  //get eigenstates, eigenenergies and k_0 points from external state solver
  //std::vector<double> initial_energies;
  //std::vector<std::vector<std::complex<double> > > initial_states;
  //std::vector<std::vector<double> > initial_momenta;

  //MPI_Barrier(k_space_communicator);
  /*std::cerr<<"before get data\n";
  state_solver->get_data(std::string("subset_of_momenta"),subset_of_momenta);
  state_solver->get_data(std::string("subset_of_eigenvalues"),subset_of_eigenvalues);
  state_solver->get_data(std::string("subset_of_eigenvectors"),subset_of_eigenvectors);
  std::cerr<<"after get data\n";*/

  //scheme: similar to density calculation - for the MPI-parallelization
  // the only difference is the external wave functions...
  /*scattering_rates.clear();
  scattering_rates.resize(subset_of_eigenvectors.size());*/
  //get states and energies
  // const vector<vector<complex<double> > >* eig_vectors  = solver->get_eigenvectors();
  const vector<std::complex<double> >*           evals        = solver->get_eigenvalues();

  unsigned int number_of_ev = solver->get_num_found_eigenvalues();

  double angle;
  if(all_angles_scat_contributions.size()==0)
    all_angles_scat_contributions.resize(subset_of_eigenvectors.size());
  if(all_momenta_scat_contributions.size()==0)
    all_momenta_scat_contributions.resize(subset_of_eigenvectors.size());
  if(all_full_resolved_scat_contributions.size()==0)
    all_full_resolved_scat_contributions.resize(subset_of_eigenvectors.size());
  for(unsigned int i=0; i<subset_of_eigenvectors.size(); i++)
  {
    //std::cerr<<"subset_of_eigenvectors "<<i<<"\n";

    //calculate the scalar products, multiply with Lorentzian, update the scattering_rates
    for(unsigned int j=0; j<number_of_ev; j++)
    {

      vector<complex<double> > eig_vector;
      solver->get_eigenvector(j).get_local_part(eig_vector);

      //debugging: additional output to compare the wavefunctions of different simulations
      /*if((std::abs(k_point[0]-subset_of_momenta[i][0])<= std::abs(k_point[0])*1e-2 || k_point[0]==0.0 && subset_of_momenta[i][0]==0.0)
        && (std::abs(k_point[1]-subset_of_momenta[i][1])<= std::abs(k_point[1])*1e-2 || k_point[1]==0.0 && subset_of_momenta[i][1]==0.0)
        && (std::abs(k_point[2]-subset_of_momenta[i][2])<= std::abs(k_point[2])*1e-2 || k_point[2]==0.0 && subset_of_momenta[i][2]==0.0)
        && std::abs(subset_of_eigenvalues[i]-(*evals)[j])<=std::abs((*evals)[j])*1e-2)*/

      //if(i==0 && std::abs((*evals)[j])<1e-3 && std::abs(k_point[0])<1e-3)
      //{
      //  msg<<"printing extra output\n";
      //  stringstream filename; filename << "debug_WF_" <<k_point[0]<<"_"<<k_point[1]<<"_"<<k_point[2]<<"_"<< get_name().c_str() << "_f" << j;
      //  this->print_eigenfunction(filename.str(), "VTK", j);
      //}

      {
        //determine the angle between k_point and the original momentum
        double dot_product = NemoMath::vector_dot(k_point,subset_of_momenta[i]);
        double norms = NemoMath::vector_norm_3d(&(k_point[0]))*NemoMath::vector_norm_3d(&(subset_of_momenta[i][0]));
        bool parallel_vectors=true;
        //check if the vectors are parallel
        double temp_factor=0.0;
        for(unsigned int ii=0; ii<k_point.size()&&parallel_vectors; ii++)
        {
          if(k_point[ii]==0.0)
          {
            if(subset_of_momenta[i][ii]!=0.0)
              parallel_vectors=false;
          }
          else if(temp_factor==0.0)
            temp_factor=subset_of_momenta[i][ii]/k_point[ii];
          else
          {
            if(std::abs(subset_of_momenta[i][ii]-k_point[ii]*temp_factor)>std::abs(1e-12*subset_of_momenta[i][ii]))
              parallel_vectors=false;
          }
        }
        if(parallel_vectors)
          angle = 0.0;
        else
        {
          NEMO_ASSERT(std::abs(dot_product)<=std::abs(norms),"[Schroedinger(\""+get_name()+"\")]::calculate_scattering_rate error in the scalar product\n");
          angle = std::abs(NemoMath::acos(dot_product/norms)*180.0/NemoMath::pi);
        }
        //store the angle
        std::map<unsigned int, std::set<double> >::iterator angle_it=all_scattering_angles.find(i);
        if(angle_it==all_scattering_angles.end())
          all_scattering_angles[i].insert(angle);
        else
          angle_it->second.insert(angle);

        //determine the absolute difference between k_point and the original momentum
        std::vector<double> delta_k(3,0.0);
        for(unsigned int ii=0; ii<3; ii++)
          delta_k[ii]=k_point[ii]-subset_of_momenta[i][ii];
        double abs_delta_k=NemoMath::vector_norm_3d(&(delta_k[0]));

        double min_angle = options.get_option("minimum_scattering_angle",0.0);
        double max_angle = options.get_option("maximum_scattering_angle",180.0);
        double min_delta_k = options.get_option("minimum_delta_k",0.0); //[1/nm]
        double max_delta_k = options.get_option("maximum_delta_k",1e80); //[1/nm]
        if(angle<max_angle && angle>min_angle && abs_delta_k<max_delta_k && abs_delta_k>min_delta_k)
        {
          std::complex<double> temp=std::complex<double> (0.0,0.0);
          for(unsigned int k=0; k<subset_of_eigenvectors[i].size(); k++)
          {

            temp+=std::conj(eig_vector[k])*subset_of_eigenvectors[i][k];
          }
          //temp*=temp;
          double abs_temp=std::abs(temp*temp);

          double temp_width=options.get_option("Dirac_gaussian_width",1e-3);
          double temp_energy=(*evals)[j].real();
          abs_temp*=NemoMath::exp(-std::pow(std::abs(subset_of_eigenvalues[i]-temp_energy)/temp_width,2));

          std::map<unsigned int, double>::iterator it=scattering_rates.find(i);
          if(it==scattering_rates.end())
            scattering_rates[i]=weight*abs_temp;
          else
            it->second+=weight*abs_temp;
          //store the scattering contribution and the abs_delta_k
          std::map<double,double>::iterator delta_k_it=all_momenta_scat_contributions[i].find(abs_delta_k);
          if(delta_k_it==all_momenta_scat_contributions[i].end())
            all_momenta_scat_contributions[i][abs_delta_k]=weight*abs_temp;
          else
            delta_k_it->second+=weight*abs_temp;
          //store the scattering contribution and the angle
          std::map<double,double>::iterator angle_it2=all_angles_scat_contributions[i].find(angle);
          if(angle_it2==all_angles_scat_contributions[i].end())
            all_angles_scat_contributions[i][angle]=weight*abs_temp;
          else
            angle_it2->second+=weight*abs_temp;
          //store the scattering contribution and the final k-vector
          std::map<std::vector<double>,double>::iterator full_it=all_full_resolved_scat_contributions[i].find(k_point);
          if(full_it==all_full_resolved_scat_contributions[i].end())
            all_full_resolved_scat_contributions[i][k_point]=weight*abs_temp;
          else
            full_it->second+=weight*abs_temp;
        }
        else
        {
          std::map<unsigned int, double>::iterator it=scattering_rates.find(i);
          if(it==scattering_rates.end())
            scattering_rates[i]=0.0;
        }
      }
    }
  }

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::create_real_space_transformation(void)
{

  /*
    std::map<std::pair<unsigned int,unsigned int>, std::complex<double> > map_result;

    //the transformation is a combination of the inverse square root of the volume of each atom
    //and the amplitude of the wave functions at each atom and orbital
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    DOFmap&                dof_map = get_dof_map();
    const AtomicStructure& atoms   = domain->get_atoms();
    HamiltonConstructor* tb_ham = NULL;
    HamiltonConstructor* tb_ham2 = NULL;

    const map< unsigned int,AtomStructNode >& lattice = atoms.lattice();

    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    //1. loop over all atoms1
    for(;it!=end;it++)
    {
      const AtomStructNode& nd1        = it.node();
      const Atom*           atom1      = nd1.atom;
      const unsigned int    atom_id1   = it.id();
      const Material*       material1  = atom1->get_material();
      const Crystal *       crystal1   = material1->get_crystal();

      //2. get the local volume of the unit cell around the atom and devide by number of atoms in the unit cell
      double cell_volume_per_atom = crystal1->calculate_primitive_cell_volume() / crystal1->get_number_of_prim_cell_atom();


      //3. get the Hamiltonian constructor for this atom
      tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(material1));
      unsigned short n_orbital1 = tb_ham->get_number_of_orbitals();
      const map<short, unsigned int>* atom_dofs1 = dof_map.get_atom_dof_map(atom_id1);
      std::map<short, unsigned int>::const_iterator it_dofs1;
      std::vector<unsigned int> orbital_dof1(n_orbital1);
      ConstActiveAtomIterator it2  = atoms.active_atoms_begin();
      //4. loop over all atoms2
      for(;it2!=end;it2++)
      {
        const AtomStructNode& nd2        = it2.node();
        const Atom*           atom2      = nd.atom;
        const unsigned int    atom_id2   = it2.id();
        const Material*       material2  = atom2->get_material();
        const Crystal *       crystal2   = material2->get_crystal();

        tb_ham2 = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(material2));
        unsigned short n_orbital2 = tb_ham2->get_number_of_orbitals();
        std::map<short, unsigned int>::const_iterator it_dofs2;
        std::vector<unsigned int> orbital_dof2(n_orbital2);
        //4. loop over all orbitals1
        for (unsigned short i = 0; i < n_orbital1; i++)
        {
          it_dofs1 = atom_dofs1->find(i);
          NEMO_ASSERT( it_dofs1 != atom_dofs1->end(), "Schroedinger::create_real_space_transformation have not found it_dofs1\n");

          orbital_dof1[i] = it_dofs1->second;
          //5. loop over all orbitals2
          for(unsigned int ii=0;ii<n_orbital2;ii++)
          {
          }

        }


        //6. query get_diagonal and get_off for the transformation matrix
        //7. multiply the inverse square root of the volume with the amplitude of the wave functions


    }
   */
}

void Schroedinger::obtain_initial_scat_states()
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::obtain_initial_scat_states";
  NemoUtils::tic(tic_toc_prefix);

  //get the external state solver
  NEMO_ASSERT(options.check_option("State_solver"),"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate define \"State_solver\"\n");
  std::string state_solver_name = options.get_option("State_solver",std::string(""));
  Simulation* state_solver = this->find_simulation(state_solver_name);
  NEMO_ASSERT(state_solver!=NULL,"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate have not found simulation \""+state_solver_name
      +"\"\n");
  //check:
  //external state solver is defined on the same domain as this solver
  const Domain* external_domain=state_solver->get_const_simulation_domain();
  NEMO_ASSERT(external_domain->get_name()==get_const_simulation_domain()->get_name(),"Schroedinger(\""+this->get_name()+"\")::calculate_scattering_rate"
      +" mismatch with domain of simulation \""+state_solver_name+"\"\n");
  state_solver->get_data(std::string("subset_of_momenta"),subset_of_momenta);
  state_solver->get_data(std::string("subset_of_eigenvalues"),subset_of_eigenvalues);
  state_solver->get_data(std::string("subset_of_eigenvectors"),subset_of_eigenvectors);

  NemoUtils::toc(tic_toc_prefix);
}


void Schroedinger::clean_before_reinit(void)
{
  DOFmapBase*& dof_map = get_dof_map_pointer();
  if (dof_map!=NULL)
    dof_map->clear();
  //DOFmap& dof_map = get_dof_map(); //DM, Removed, it does not check if the dofmap was created before

  delete hamiltonian;
  hamiltonian = NULL;

  delete overlap;
  overlap = NULL;

  delete solver;
  solver = NULL;

  if(!PDOS.empty())
  {
    for(unsigned int j=0; j<PDOS.size(); j++)
      std::fill(PDOS[j].begin(), PDOS[j].end(), 0.0);
  }

}


void Schroedinger::assemble_velocity_operator(const NemoMeshPoint& momentum, PetscMatrixParallelComplex*& position_operator,
    PetscMatrixParallelComplex*& v)
{
  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::assemble_velocity_operator";
  NemoUtils::tic(tic_toc_prefix);

  PetscMatrixParallelComplex* HX = NULL;
  PetscMatrixParallelComplex* XH = NULL;
  PetscMatrixParallelComplex* Hamiltonian = NULL;
  get_data(std::string("Hamiltonian"),momentum,Hamiltonian);//Hamiltonian unit eV
  PetscMatrixParallelComplex::mult(*Hamiltonian,*position_operator, &HX);//position unit nm
  PetscMatrixParallelComplex::mult(*position_operator,*Hamiltonian, &XH);
  HX->add_matrix(*XH,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0)); // HX-XH
  HX->assemble();
  //*HX *=std::complex<double>(NemoPhys::elementary_charge/NemoPhys::h,0.0); // i/h*(HX-XH) unit m/s
  v=new PetscMatrixParallelComplex(*HX);

  delete HX;
  delete XH;
  NemoUtils::toc(tic_toc_prefix);
}

// Commented EVmax, ECmin to avoid a warning - D.L.
void Schroedinger::fermi_energy(double Ndop,double rho_sign,double spin_factor,double Temp,double kmin,double kmax,double dk,int N1,double kymin,double kymax,
  int N2,double kzmin,double kzmax,int N3,/*double EVmax,*/ /*double ECmin,*/ double cell_area,int /*no_task*/,int /*index*/)
{//author: Bozidar N., Zhengping J.

  std::string tic_toc_prefix = "Schroedinger(\""+tic_toc_name+"\")::fermi_energy";
  NemoUtils::tic(tic_toc_prefix);

  Fermi FF;
  //  double midgap = (ECmin+EVmax)/2;
  int n_of_modes = 0;

  /*
  for (unsigned int jj=0; jj<EnergyV[0].size(); jj++)
    if(EnergyV[0][jj]>midgap)
      n_of_modes++;
   */

  n_of_modes = EnergyV[0].size();

  msg << "[Schroedinger(\""+get_name()+"\")][fermi_energy]: number of modes is " << n_of_modes << endl;

  //double *Ek = new double [kspace->get_num_points() * n_of_modes];

  /*
  int conduction_band_number = 0;

  for (unsigned int jj=0; jj<EnergyV[0].size(); jj++)  {
    if(EnergyV[0][jj]>midgap){
      for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)  {
        Ek[conduction_band_number*kspace->get_num_points() + ii] = EnergyV[ii][jj];
        }
      conduction_band_number++;
    }
  }
   */

  std::list<double> kx_v, ky_v, kz_v;

  for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
  {
    const NemoMeshPoint& ki = kspace->get_point(ii);
    kx_v.push_back(ki[0]);
    ky_v.push_back(ki[1]);
    kz_v.push_back(ki[2]);
  }
  kx_v.sort();
  ky_v.sort();
  kz_v.sort();
  kx_v.unique(NemoMath::CompareDoubleEqual());
  ky_v.unique(NemoMath::CompareDoubleEqual());
  kz_v.unique(NemoMath::CompareDoubleEqual());

  vector<double> Ek(N1*N2*N3*n_of_modes);

  for (unsigned int ii = 0; ii < kspace->get_num_points(); ii++)
  {

    std::list<double>::iterator it = kx_v.begin();
    int indx = -1;
    unsigned int counter = 0;
    for (; it != kx_v.end(); it++)
    {
      if (NemoMath::CompareDoubleEqual()(kspace->get_point(ii).get_x(),(*it)))
      {
        indx = counter;
        break;
      }
      counter = counter + 1;
    }

    it = ky_v.begin();
    int indy = -1;
    counter = 0;
    for (; it != ky_v.end(); it++)
    {
      if (NemoMath::CompareDoubleEqual()(kspace->get_point(ii).get_y(),(*it)))
      {
        indy = counter;
        break;
      }
      counter = counter + 1;
    }

    it = kz_v.begin();
    int indz = -1;
    counter = 0;
    for (; it != kz_v.end(); it++)
    {
      if (NemoMath::CompareDoubleEqual()(kspace->get_point(ii).get_z(),(*it)))
      {
        indz = counter;
        break;
      }
      counter = counter + 1;
    }

    if ((indx == -1) || (indy == -1) || (indz == -1))
      throw runtime_error("[Schroedinger::fermi_energy]: problem in reordering the band structure vector for input into the function for finding the fermi energy.");

    for (int jj = 0; jj < n_of_modes; jj++)
    {
      Ek[indy*N3*N1*n_of_modes + indz*N1*n_of_modes + indx*n_of_modes  + jj] = EnergyV[ii][jj];
      //cerr << indx << " " << indy << " " << indz << " " << jj << " " << Ek[indy*N3*N1*n_of_modes + indz*N1*n_of_modes + indx*n_of_modes  + jj] << endl;
    }
  }

  /*
  int conduction_band_number = 0;
   for (unsigned int jj=0; jj < EnergyV[0].size(); jj++)
     if(EnergyV[0][jj] > midgap)
       conduction_band_number++;

   for (unsigned int jj = 0; jj < EnergyV[0].size(); jj++)
   {
     if(EnergyV[0][jj] > midgap)
     {
       for (unsigned int ii = 0; ii < kspace->get_num_points(); ii++)
       {
         Ek[conduction_band_number * ii + jj] = EnergyV[ii][jj];
       }
     }
   }
   */
  msg<<"Ek reordered into 1D vector..."<<endl;

  vector<double> dky(abs(N2)*abs(N3));
  vector<double> dkz(abs(N2)*abs(N3));

  for (unsigned int i = 0; i < dky.size(); i++)
  {
    if(N2>1)
      dky[i] = (kymax-kymin)/(N2-1);
    else dky[i] = 1;
    if(N3>1)
      dkz[i] = (kzmax-kzmin)/(N3-1);
    else dkz[i] = 1;
  }
  msg<<"kmin="<<kmin<<"; kmax="<<kmax<<"; dk="<<dk<<endl;
  msg<<"kymin="<<kymin<<"; kymax="<<kymax<<"; dky="<<dky[0]<<endl;
  msg<<"kzmin="<<kzmin<<"; kzmax="<<kzmax<<"; dkz="<<dkz[0]<<endl;

  Ef = FF.find_fermi(fabs(Ndop), rho_sign , &(Ek[0]), dk, N1, &(dky[0]), abs(N2), &(dkz[0]) ,abs(N3), n_of_modes, spin_factor, Temp, cell_area);
  msg<<"Ef = "<<Ef<<endl;



  //  delete[] Ek;
  //  delete[] FF;
  //  delete[] dky;
  //  delete[] dkz;

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_data(const std::string& variable, unsigned int& value)
{

  if (variable == "number_of_k_points")
  {
    value = EnergyV.size();
  }
  else if (variable == "number_of_energy_points")
  {
    value = EnergyV[0].size();
  }
  else if (variable == "spin_degeneracy")
  {
    value = (dynamic_cast<HamiltonConstructor*> (material_properties.begin()->second))->get_spin_degeneracy();
  }
  else if (variable == "global_size")
  {
    if (!(get_dof_map().ready()))
      init_dof_map();

    value = get_dof_map().get_global_dof_number();
  }
  else if (variable == "local_size")
  {
    if (!(get_dof_map().ready()))
      init_dof_map();

    value = get_dof_map().get_number_of_dofs();
  }
  else if (variable == "ref_domain_atom_4matprop") //Aryan add
  {

    value = options.get_option("ref_domain_atom_4matprop",0); //default 1st atom of this domain as before

  }
  else
    throw runtime_error("[Schroedinger(\""+get_name()+"\")::get_data] wrong variable " + variable + "\n" + get_name() + "\n");
}


/*
void Schroedinger::get_data(std::vector<double> &Ek)
{
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    const AtomicStructure& atoms   = domain->get_atoms();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    const AtomStructNode& nd        = it.node();
    const Atom*           atom      = nd.atom;
    const Material*       material  = atom->get_material();

//  vector<double> Ek(kspace->get_num_points() * n_of_modes);
    HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(material));
  double ECmin = tb_ham->get_conduction_band_edge();
  double EVmax = tb_ham->get_valence_band_edge();
  double midgap = (ECmin + EVmax) / 2;

  int conduction_band_number = 0;
  for (unsigned int jj=0; jj < EnergyV[0].size(); jj++)
    if(EnergyV[0][jj] > midgap)
      conduction_band_number++;

  for (unsigned int jj = 0; jj < EnergyV[0].size(); jj++)
  {
    if(EnergyV[0][jj] > midgap)
    {
      for (unsigned int ii = 0; ii < kspace->get_num_points(); ii++)
      {
        Ek[conduction_band_number * ii + jj] = EnergyV[ii][jj];
      }
    }
  }
  fermi_level = Ef;
}
 */
void Schroedinger::combine_hamiltonian(void)
{
  throw runtime_error("Schroedinger(\""+get_name()+"\")::combine_hamiltonian is not implemented\n");
}

void Schroedinger::create_dimensional_S_Matrix(PetscMatrixParallelComplex*& result)
{
  std::string prefix="Schroedinger(\""+get_name()+"\")::create_dimensional_S_Matrix ";
  //1.loop over all active atoms
  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
  DOFmapInterface&                dof_map = get_dof_map();
  const AtomicStructure& atoms   = domain->get_atoms();
  ConstActiveAtomIterator it  = atoms.active_atoms_begin();
  ConstActiveAtomIterator end = atoms.active_atoms_end();

  delete result;

  std::map<unsigned int,double> volume_per_DOF;

  for ( ; it != end; ++it)
  {
    const AtomStructNode& nd        = it.node();
    const Atom*           atom      = nd.atom;
    const unsigned int    atom_id   = it.id();
    const Material*       material  = atom->get_material();
    const Crystal*        crystal   = material->get_crystal();
    //2.store the atom volume for each of them into a map (DOFindex, volume)
    //2.1 solve the volume for this atom
    double cell_volume_per_atom = crystal->calculate_primitive_cell_volume()/crystal->get_number_of_prim_cell_atom();
    //NOTE: if there are N orbitals per atom, they should be contracted to a atom of 4 states, i.e. no futher normalization is required
    double periodic_space_volume=domain->return_periodic_space_volume();
    //2.2 get the DOFs of this atom and its orbitals
    const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
    map<short, unsigned int>::const_iterator atom_dof_cit=atom_dofs->begin();
    for(; atom_dof_cit!=atom_dofs->end(); ++atom_dof_cit)
    {
      //make sure that this DOF has not been filled already
      std::map<unsigned int,double>::iterator temp_it=volume_per_DOF.find(atom_dof_cit->second);
      NEMO_ASSERT(temp_it==volume_per_DOF.end(),prefix+"current DOF had been defined before.\n");
      if(std::abs(periodic_space_volume)>NemoMath::d_zero_tolerance)
        volume_per_DOF[atom_dof_cit->second]=periodic_space_volume/cell_volume_per_atom;
      else
        volume_per_DOF[atom_dof_cit->second]=1.0/cell_volume_per_atom;
    }
  }
  //3.create a diagonal matrix for the result
  result = new PetscMatrixParallelComplex(volume_per_DOF.size(),volume_per_DOF.size(),get_simulation_domain()->get_communicator());
  //3.1 set the sparsity pattern for the diagonal matrix
  result->set_num_owned_rows(volume_per_DOF.size());
  for(unsigned int i = 0; i < volume_per_DOF.size(); i++)
    result->set_num_nonzeros(i,1,0);
  result->allocate_memory();
  //4.fill the result matrix with values of 2.
  std::map<unsigned int,double>::const_iterator temp_cit=volume_per_DOF.begin();
  for(; temp_cit!=volume_per_DOF.end(); ++temp_cit)
    result->set(temp_cit->first,temp_cit->first,temp_cit->second);
  //5. finalize the result
  result->assemble();
}

//RK for bandstructure optimization
//----------------------------------------------------
void Schroedinger::do_target_postprocessing(void)
{
  //----------------------------------------------------
  string fileoutname = options.get_option("input_filename_for_targets", string(""));

  if(fileoutname == "")
  {
    throw runtime_error("Schroedinger:: file name with targets is not provided for the do_target_postprocessing option\n");
  }
  //----------------------------------------------------
  this->get_lattice_constants(lattice_a);
  //----------------------------------------------------
  //postprocessing for each target type = returns the calculated value, the error, and the reference value
  void (Schroedinger::*targetTypePpMethod)(double res[3]);
  //----------------------------------------------------
  targets_for_bulk_BS = options.get_option("targets_for_bulk_BS",true);

  int places=15;
  int num_targets;
  std::ifstream kint_file(fileoutname.c_str(),std::ios_base::in);
  kint_file.precision(places);

  kint_file>>num_targets;
  if (!targets_for_bulk_BS) {
    kint_file>>targets_Energy_threshold;
  }

  //----------------------------------------------------
  double goodness_f=0.0;
  double totaltargetnorm=0.0;
  map<int, double> targetweight;
  map<int, double> targeterror;
  map<int, double> targetcalculated;
  map<int, double> targetreference;
  map<int, int> targettype;
  int tgt_num, tgt_type;
  double tgt_weight;
  //----------------------------------------------------
  num_dE=-1;
  num_Mass=-1;
  num_Deriv=-1;
  num_Kmin=-1;
  num_Kmax=-1;

  ik1_dE.clear();
  ik2_dE.clear();
  ie1_dE.clear();
  ie2_dE.clear();
  en1_dE.clear();
  en2_dE.clear();
  entol_dE.clear();
  deref_dE.clear();

  ik1_Mass.clear();
  ik2_Mass.clear();
  ik3_Mass.clear();
  ie_Mass.clear();
  mtol_Mass.clear();
  mref_Mass.clear();

  ik1_Deriv.clear();
  ik2_Deriv.clear();
  ie_Deriv.clear();
  gtol_Deriv.clear();
  gref_Deriv.clear();

  nks_Kmin.clear();
  kvalues_Kmin.clear();
  ie_Kmin.clear();
  ktol_Kmin.clear();
  krefx_Kmin.clear();
  krefy_Kmin.clear();
  krefz_Kmin.clear();

  nks_Kmax.clear();
  kvalues_Kmax.clear();
  ie_Kmax.clear();
  ktol_Kmax.clear();
  krefx_Kmax.clear();
  krefy_Kmax.clear();
  krefz_Kmax.clear();



  //----------------------------------------------------
  for (int ikt=0; ikt<num_targets; ikt++)
  {
    //----------------------------------------------------
    kint_file>>tgt_num>>tgt_type;
    targettype[tgt_num]=tgt_type;
    kint_file>>tgt_weight;
    targetweight[tgt_num]=tgt_weight;
    //----------------------------------------------------
    if(tgt_num != ikt)
    {
      throw runtime_error("Schroedinger:: error on reading target file in do_target_postprocessing option\n");
    }
    //----------------------------------------------------
    switch (tgt_type)
    {
    case 0: //energy diff
      //-----------------------------------
    {
      int ik1, ik2, ie1,ie2;
      double en1, en2, entol,deref;
      num_dE++;
      kint_file >> ik1 >> ik2 >> ie1 >> ie2 >> en1 >> en2 >> entol >> deref;
      ik1_dE[num_dE]=ik1;
      ik2_dE[num_dE]=ik2;
      ie1_dE[num_dE]=ie1;
      ie2_dE[num_dE]=ie2;
      en1_dE[num_dE]=en1;
      en2_dE[num_dE]=en2;
      entol_dE[num_dE]=entol;
      deref_dE[num_dE]=deref;
      //-----------------------------------
    }
    break;

    case 1:  //Mass
      //-----------------------------------
    {
      int ik1, ik2, ik3, ie;
      double mtol, mref;
      kint_file>> ik1 >> ik2 >> ik3 >> ie >> mtol >> mref;
      num_Mass++;
      ik1_Mass[num_Mass]=ik1;
      ik2_Mass[num_Mass]=ik2;
      ik3_Mass[num_Mass]=ik3;
      ie_Mass[num_Mass]=ie;
      mtol_Mass[num_Mass]=mtol;
      mref_Mass[num_Mass]=mref;
      //-----------------------------------
    }
    break;
    case 2: //Deriv
      //-----------------------------------
    {
      int ik1, ik2, ie;
      double gtol, gref;
      kint_file>> ik1 >> ik2 >> ie >> gtol >> gref;
      num_Deriv++;

      ik1_Deriv[num_Deriv]=ik1;
      ik2_Deriv[num_Deriv]=ik2;
      ie_Deriv[num_Deriv]=ie;
      gtol_Deriv[num_Deriv]=gtol;
      gref_Deriv[num_Deriv]=gref;
      //-----------------------------------
    }
    break;
    case 3: //kmin: k-value at which this band is min
      //-----------------------------------
    {
      int ik1,  ie;
      double ktol, krefx, krefy, krefz;
      int nks;
      vector<int> kvalues;
      num_Kmin++;
      kint_file>>nks;

      kvalues.assign(nks,0);
      for (int ik=0; ik<nks; ik++)
      {
        kint_file >> ik1;
        kvalues[ik]=ik1;
      }

      kint_file >> ie >> ktol >> krefx >> krefy >> krefz;
      nks_Kmin[num_Kmin]=nks;
      kvalues_Kmin[num_Kmin]=kvalues;
      ie_Kmin[num_Kmin]=ie;
      ktol_Kmin[num_Kmin]=ktol;
      krefx_Kmin[num_Kmin]=krefx;
      krefy_Kmin[num_Kmin]=krefy;
      krefz_Kmin[num_Kmin]=krefz;
      //-----------------------------------
    }
    break;
    case 4: //kmax: k-value at which this band is max
      //-----------------------------------
    {
      int ik1,  ie;
      double ktol, krefx, krefy, krefz;
      int nks;
      vector<int> kvalues;
      num_Kmax++;
      kint_file>>nks;

      kvalues.assign(nks,0);
      for (int ik=0; ik<nks; ik++)
      {
        kint_file >> ik1;
        kvalues[ik]=ik1;
      }

      kint_file >> ie >> ktol >> krefx >> krefy >> krefz;
      nks_Kmax[num_Kmax]=nks;
      kvalues_Kmax[num_Kmax]=kvalues;
      ie_Kmax[num_Kmax]=ie;
      ktol_Kmax[num_Kmax]=ktol;
      krefx_Kmax[num_Kmax]=krefx;
      krefy_Kmax[num_Kmax]=krefy;
      krefz_Kmax[num_Kmax]=krefz;
      //-----------------------------------
    }

    break;
    default:
      throw runtime_error("Schroedinger:: this target type is not supported for the do_target_postprocessing option\n");
      break;
    }
    //----------------------------------------------------
  }
  //----------------------------------------------------
  kint_file.close();
  //----------------------------------------------------
  num_dE=-1;
  num_Mass=-1;
  num_Deriv=-1;
  num_Kmin=-1;
  num_Kmax=-1;
  //----------------------------------------------------
  for (int ik=0; ik<num_targets; ik++)
  {
    //----------------------------------------------------
    tgt_num=ik;
    tgt_type=targettype[tgt_num];

    switch (tgt_type)
    {
    case 0: //energy diff
      num_dE++;
      targetTypePpMethod = &Schroedinger::do_target_postprocessing_dE;
      break;
    case 1:  //Mass
      num_Mass++;
      targetTypePpMethod = &Schroedinger::do_target_postprocessing_Mass;
      break;
    case 2: //Deriv
      num_Deriv++;
      targetTypePpMethod = &Schroedinger::do_target_postprocessing_Deriv;
      break;
    case 3: //kmin: k-value at which this band is min
      num_Kmin++;
      targetTypePpMethod = &Schroedinger::do_target_postprocessing_Kmin;
      break;
    case 4: //kmax: k-value at which this band is max
      num_Kmax++;
      targetTypePpMethod = &Schroedinger::do_target_postprocessing_Kmax;
      break;
    default:
      throw runtime_error("Schroedinger:: this target type is not supported for the do_target_postprocessing option\n");
      break;
    }

    double tempres[3];
    (this->*targetTypePpMethod)(tempres);
    targetcalculated[tgt_num]=tempres[0];
    targetreference[tgt_num]=tempres[1];
    targeterror[tgt_num]=tempres[2];

    totaltargetnorm+=targetweight[tgt_num];
    goodness_f+=targeterror[tgt_num]*targeterror[tgt_num]*targetweight[tgt_num];
    //----------------------------------------------------
  }

  goodness_f=sqrt(goodness_f/totaltargetnorm);
  //----------------------------------------------------
  string fileoutname1 = options.get_option("output_filename_for_targets", string(""));

  if(fileoutname1 == "")
  {
    throw runtime_error("Schroedinger:: file name with targets results is not provided for the do_target_postprocessing option\n");
  }
  //----------------------------------------------------
  bool do_detailed_target_output = options.get_option("do_detailed_target_output", false);
  ofstream kfile(fileoutname1.c_str(), std::ios_base::out | std::ios_base::trunc);
  kfile.precision(places);
  kfile<<goodness_f<<std::endl;
  if(do_detailed_target_output)
  {
    //----------------------------------------------------
    for (int ik=0; ik<num_targets; ik++)
    {
      tgt_num=ik;
      kfile<<ik
          <<" "<<targettype[tgt_num]
                    <<" "<<targetcalculated[tgt_num]
                                <<" "<<targetreference[tgt_num]
                                           <<" "<<targeterror[tgt_num]
                                                    <<" "<<targetweight[tgt_num]
                                                              <<" "<<targeterror[tgt_num]*targetweight[tgt_num]
                                                                                   <<std::endl;
    }
    //----------------------------------------------------
  }
  kfile.close();
  //----------------------------------------------------
  return;
}

int  Schroedinger::targets_eigen_shift_for_k_point(const int energy_num)
{

  if (targets_for_bulk_BS) {
    return 0;
  }

  int nv=0;
  for (unsigned int ii=0; ii<EnergyV[energy_num].size(); ii++) {
    double entemp=EnergyV[energy_num][ii];
    if(entemp < targets_Energy_threshold){
      nv++;
    }
  }

  return nv;
}


void Schroedinger::do_target_postprocessing_dE(double results[3])
{

  int ik1, ik2, ie1,ie2;
  double en1, en2, entol,deref;

  ik1=ik1_dE[num_dE];
  ik2=ik2_dE[num_dE];
  ie1=ie1_dE[num_dE];
  ie2=ie2_dE[num_dE];
  en1=en1_dE[num_dE];
  en2=en2_dE[num_dE];
  entol=entol_dE[num_dE];
  deref=deref_dE[num_dE];

  if(ik1>-1)
  {
    en1=EnergyV[ik1][ie1+targets_eigen_shift_for_k_point(ik1)];
  }

  if(ik2>-1)
  {
    en2=EnergyV[ik2][ie2+targets_eigen_shift_for_k_point(ik2)];
  }

  results[0]=en1-en2;
  results[1]=deref;
  results[2]=(results[0]-deref)/entol;

  return;
}


void Schroedinger::do_target_postprocessing_Mass(double results[3])
{
  int ik1, ik2, ik3, ie;
  double mtol, mref;
  double dk;
  double secondderiv=0.0;
  double secondderivmin=1.e-9;
  double en1, en2, en0;
  double rhp = NemoPhys::hbar;
  double rm0 = NemoPhys::electron_mass;
  double rqc = NemoPhys::elementary_charge;

  ik1=ik1_Mass[num_Mass];
  ik2=ik2_Mass[num_Mass];
  ik3=ik3_Mass[num_Mass];
  ie=ie_Mass[num_Mass];
  mtol=mtol_Mass[num_Mass];
  mref=mref_Mass[num_Mass];


  const NemoMeshPoint& ki1 = kspace->get_point(ik1);
  const NemoMeshPoint& ki2 = kspace->get_point(ik2);
  en0=EnergyV[ik1][ie+targets_eigen_shift_for_k_point(ik1)];
  en1=EnergyV[ik2][ie+targets_eigen_shift_for_k_point(ik2)];
  en2=EnergyV[ik3][ie+targets_eigen_shift_for_k_point(ik3)];

  dk=sqrt((ki2[0]-ki1[0])*(ki2[0]-ki1[0])+(ki2[1]-ki1[1])*(ki2[1]-ki1[1])+(ki2[2]-ki1[2])*(ki2[2]-ki1[2]));

  secondderiv = (en1 + en2 - 2.0 * en0)
                    / (dk * dk);

  if(fabs(secondderiv)<secondderivmin)
  {
    secondderiv=secondderivmin;
  }

  results[0]= -1.e18 * rhp * rhp / (secondderiv * rm0 * rqc);
  results[1]=mref;
  results[2]=(results[0]-mref)/mtol;
  return;
}

void Schroedinger::do_target_postprocessing_Deriv(double results[3])
{
  int ik1, ik2, ie;
  double gtol, gref;
  double dk;
  double en1, en2;

  ik1=ik1_Deriv[num_Deriv];
  ik2=ik2_Deriv[num_Deriv];
  ie=ie_Deriv[num_Deriv];
  gtol=gtol_Deriv[num_Deriv];
  gref=gref_Deriv[num_Deriv];


  const NemoMeshPoint& ki1 = kspace->get_point(ik1);
  const NemoMeshPoint& ki2 = kspace->get_point(ik2);
  en1=EnergyV[ik1][ie+targets_eigen_shift_for_k_point(ik1)];
  en2=EnergyV[ik2][ie+targets_eigen_shift_for_k_point(ik2)];

  dk=sqrt((ki2[0]-ki1[0])*(ki2[0]-ki1[0])+(ki2[1]-ki1[1])*(ki2[1]-ki1[1])+(ki2[2]-ki1[2])*(ki2[2]-ki1[2]));


  results[0]=(en2-en1)/(2.0*dk);
  results[1]=gref;
  results[2]=(results[0]-gref)/gtol;
  return;
}

void Schroedinger::do_target_postprocessing_Kmin(double results[3])
{

  int ik1, ik2, ik3, ie;
  double ktol, krefx, krefy, krefz;
  double dk; //all k here are on one line
  double dk2; //all k here are on one line
  double en1, en2, en3, de1, de2;
  double kroot=1.e9;
  double krootx=1.e9;
  double krooty=1.e9;
  double krootz=1.e9;
  double ftol=1.e-60;
  double deriv=0.0;
  double deriv2=0.0;
  double kunitv[3];
  double krootxsc,krootysc,krootzsc;
  double krootxsc1,krootysc1,krootzsc1;
  double kdifmin=1.e9;
  vector<int> kvalues;
  int nks;

  nks=nks_Kmin[num_Kmin];
  kvalues=kvalues_Kmin[num_Kmin];
  ie=ie_Kmin[num_Kmin];
  ktol=ktol_Kmin[num_Kmin];
  krefx=krefx_Kmin[num_Kmin];
  krefy=krefy_Kmin[num_Kmin];
  krefz=krefz_Kmin[num_Kmin];


  for (int ik=0; ik<nks-2; ik++)
  {

    ik1=kvalues[ik];
    ik2=kvalues[ik+1];
    ik3=kvalues[ik+2];

    const NemoMeshPoint& ki1 = kspace->get_point(ik1);
    const NemoMeshPoint& ki2 = kspace->get_point(ik2);
    const NemoMeshPoint& ki3 = kspace->get_point(ik3);
    en1=EnergyV[ik1][ie+targets_eigen_shift_for_k_point(ik1)];
    en2=EnergyV[ik2][ie+targets_eigen_shift_for_k_point(ik2)];
    en3=EnergyV[ik3][ie+targets_eigen_shift_for_k_point(ik3)];

    de1=en2-en1;
    de2=en3-en2;
    //---------------------------------------------------
    krootxsc1=ki2[0]/(2.0*NemoMath::pi/fabs(lattice_a[0]));
    krootysc1=ki2[1]/(2.0*NemoMath::pi/fabs(lattice_a[1]));
    krootzsc1=ki2[2]/(2.0*NemoMath::pi/fabs(lattice_a[2]));

    double tmpdf=sqrt((krefx-krootxsc1)*(krefx-krootxsc1)+(krefy-krootysc1)*(krefy-krootysc1)+(krefz-krootzsc1)*(krefz-krootzsc1));
    //---------------------------------------------------
    if(tmpdf<kdifmin)
    {
      krootx=ki2[0];
      krooty=ki2[1];
      krootz=ki2[2];
      kdifmin=tmpdf;
    }
    //---------------------------------------------------

    if(de1*de2<=ftol && de1<de2)
    {
      dk=sqrt((ki3[0]-ki1[0])*(ki3[0]-ki1[0])+(ki3[1]-ki1[1])*(ki3[1]-ki1[1])+(ki3[2]-ki1[2])*(ki3[2]-ki1[2]));
      dk2=sqrt((ki2[0]-ki1[0])*(ki2[0]-ki1[0])+(ki2[1]-ki1[1])*(ki2[1]-ki1[1])+(ki2[2]-ki1[2])*(ki2[2]-ki1[2]));
      kunitv[0]=(ki3[0]-ki1[0])/dk;
      kunitv[1]=(ki3[1]-ki1[1])/dk;
      kunitv[2]=(ki3[2]-ki1[2])/dk;
      kroot=dk2;
      krootx=ki2[0];
      krooty=ki2[1];
      krootz=ki2[2];

      deriv=(en3-en1)/(dk);
      deriv2 = (en1 + en3 - 2.0 * en2)
                   / (dk * dk/4.0);

      if(fabs(deriv2)>ftol)
      {
        kroot=dk2-deriv/deriv2;
        krootx=ki1[0]+kroot*kunitv[0];
        krooty=ki1[1]+kroot*kunitv[1];
        krootz=ki1[2]+kroot*kunitv[2];
      }
      break;
    }
  }

  krootxsc=krootx/(2.0*NemoMath::pi/fabs(lattice_a[0]));
  krootysc=krooty/(2.0*NemoMath::pi/fabs(lattice_a[1]));
  krootzsc=krootz/(2.0*NemoMath::pi/fabs(lattice_a[2]));

  results[0]=sqrt(krootxsc*krootxsc+krootysc*krootysc+krootzsc*krootzsc);
  results[1]=sqrt(krefx*krefx+krefy*krefy+krefz*krefz);
  results[2]=sqrt((krefx-krootxsc)*(krefx-krootxsc)+(krefy-krootysc)*(krefy-krootysc)+(krefz-krootzsc)*(krefz-krootzsc))/ktol;
  return;
}

void Schroedinger::do_target_postprocessing_Kmax(double results[3])
{

  int ik1, ik2, ik3, ie;
  double ktol, krefx, krefy, krefz;
  double dk; //all k here are on one line
  double dk2; //all k here are on one line
  double en1, en2, en3, de1, de2;
  double kroot=1.e9;
  double krootx=1.e9;
  double krooty=1.e9;
  double krootz=1.e9;
  double ftol=1.e-60;
  double deriv=0.0;
  double deriv2=0.0;
  double kunitv[3];
  double krootxsc,krootysc,krootzsc;
  double krootxsc1,krootysc1,krootzsc1;
  double kdifmin=1.e9;

  vector<int> kvalues;
  int nks;

  nks=nks_Kmax[num_Kmax];
  kvalues=kvalues_Kmax[num_Kmax];
  ie=ie_Kmax[num_Kmax];
  ktol=ktol_Kmax[num_Kmax];
  krefx=krefx_Kmax[num_Kmax];
  krefy=krefy_Kmax[num_Kmax];
  krefz=krefz_Kmax[num_Kmax];


  for (int ik=0; ik<nks-2; ik++)
  {

    ik1=kvalues[ik];
    ik2=kvalues[ik+1];
    ik3=kvalues[ik+2];

    const NemoMeshPoint& ki1 = kspace->get_point(ik1);
    const NemoMeshPoint& ki2 = kspace->get_point(ik2);
    const NemoMeshPoint& ki3 = kspace->get_point(ik3);
    en1=EnergyV[ik1][ie+targets_eigen_shift_for_k_point(ik1)];
    en2=EnergyV[ik2][ie+targets_eigen_shift_for_k_point(ik2)];
    en3=EnergyV[ik3][ie+targets_eigen_shift_for_k_point(ik3)];


    de1=en2-en1;
    de2=en3-en2;

    krootxsc1=ki2[0]/(2.0*NemoMath::pi/fabs(lattice_a[0]));
    krootysc1=ki2[1]/(2.0*NemoMath::pi/fabs(lattice_a[1]));
    krootzsc1=ki2[2]/(2.0*NemoMath::pi/fabs(lattice_a[2]));

    double tmpdf=sqrt((krefx-krootxsc1)*(krefx-krootxsc1)+(krefy-krootysc1)*(krefy-krootysc1)+(krefz-krootzsc1)*(krefz-krootzsc1));
    //---------------------------------------------------
    if(tmpdf<kdifmin)
    {
      krootx=ki2[0];
      krooty=ki2[1];
      krootz=ki2[2];
      kdifmin=tmpdf;
    }
    //---------------------------------------------------


    if(de1*de2<=ftol && de1>de2)
    {
      dk=sqrt((ki3[0]-ki1[0])*(ki3[0]-ki1[0])+(ki3[1]-ki1[1])*(ki3[1]-ki1[1])+(ki3[2]-ki1[2])*(ki3[2]-ki1[2]));
      dk2=sqrt((ki2[0]-ki1[0])*(ki2[0]-ki1[0])+(ki2[1]-ki1[1])*(ki2[1]-ki1[1])+(ki2[2]-ki1[2])*(ki2[2]-ki1[2]));
      kunitv[0]=(ki3[0]-ki1[0])/dk;
      kunitv[1]=(ki3[1]-ki1[1])/dk;
      kunitv[2]=(ki3[2]-ki1[2])/dk;
      kroot=dk2;
      krootx=ki2[0];
      krooty=ki2[1];
      krootz=ki2[2];

      deriv=(en3-en1)/(dk);
      deriv2 = (en1 + en3 - 2.0 * en2)
                   / (dk * dk/4.0);

      if(fabs(deriv2)>ftol)
      {
        kroot=dk2-deriv/deriv2;
        krootx=ki1[0]+kroot*kunitv[0];
        krooty=ki1[1]+kroot*kunitv[1];
        krootz=ki1[2]+kroot*kunitv[2];
      }
      break;
    }
  }

  krootxsc=krootx/(2.0*NemoMath::pi/fabs(lattice_a[0]));
  krootysc=krooty/(2.0*NemoMath::pi/fabs(lattice_a[1]));
  krootzsc=krootz/(2.0*NemoMath::pi/fabs(lattice_a[2]));

  results[0]=sqrt(krootxsc*krootxsc+krootysc*krootysc+krootzsc*krootzsc);
  results[1]=sqrt(krefx*krefx+krefy*krefy+krefz*krefz);
  results[2]=sqrt((krefx-krootxsc)*(krefx-krootxsc)+(krefy-krootysc)*(krefy-krootysc)+(krefz-krootzsc)*(krefz-krootzsc))/ktol;
  return;
}

void Schroedinger::get_lattice_constants(double alattice[3])
{
  int my_rank = get_simulation_domain()->get_geometry_rank();

  double ax,ay,az;
  double alattice1[3];

  const Domain *  dom = get_simulation_domain();
  if (dom !=NULL)
  {
    const vector< vector<double> > & transl_vectors = dom->get_translation_vectors();
    const double v1l = NemoMath::vector_norm(transl_vectors[0]);
    const double v2l = NemoMath::vector_norm(transl_vectors[1]);
    const double v3l = NemoMath::vector_norm(transl_vectors[2]);
    const double cos12 = NemoMath::vector_dot(transl_vectors[0],transl_vectors[1])/(v1l*v2l);
    const double cos13 = NemoMath::vector_dot(transl_vectors[0],transl_vectors[2])/(v1l*v3l);
    const double cos23 = NemoMath::vector_dot(transl_vectors[1],transl_vectors[2])/(v2l*v3l);
    if ((NemoMath::abs(cos12)<1e-7)&&(NemoMath::abs(cos13)<1e-7)&&(NemoMath::abs(cos23)<1e-7))
    {
      //Simple orthogonal lattice
      ax = transl_vectors[0][0];
      ay = transl_vectors[1][1];
      az = transl_vectors[2][2];                      
    } 
    else if ((NemoMath::abs(cos12-0.5)<1e-7)&&(NemoMath::abs(cos13-0.5)<1e-7)&&(NemoMath::abs(cos23-0.5)<1e-7))
    {
      //FCC (FCC, Diamond, ZincBlende) lattice
      ax = ay = az = v1l*NemoMath::sqrt(2.0);
    }
    else if ((NemoMath::abs(cos12+(1.0/3.0))<1e-7)&&(NemoMath::abs(cos13+(1.0/3.0))<1e-7)&&(NemoMath::abs(cos23+(1.0/3.0))<1e-7))
    {
      //BCC lattice
      ax = ay = az = v1l*2.0/NemoMath::sqrt(3.0);
    } 
    else if ((NemoMath::abs(cos12-0.5)<1e-5)&&(NemoMath::abs(cos13)<1e-5)&&(NemoMath::abs(cos23)<1e-5))
    {
      //Wurzite-like lattice with v1^v2=60deg and v3 normal to v1 and v2
      ax = v1l;
      ay = v2l;
      az = v3l;
    }
    else
    {
      OUT_ERROR << "v1 = ("<<transl_vectors[0][0]<<","<<transl_vectors[0][1]<<","<<transl_vectors[0][2]<<")\n";
      OUT_ERROR << "v2 = ("<<transl_vectors[1][0]<<","<<transl_vectors[1][1]<<","<<transl_vectors[1][2]<<")\n";
      OUT_ERROR << "v3 = ("<<transl_vectors[2][0]<<","<<transl_vectors[2][1]<<","<<transl_vectors[2][2]<<")\n";
      OUT_ERROR << "cos12 = "<<cos12<<" cos13 = "<<cos13<<" cos23 = "<<cos23<<"\n";
      NEMO_EXCEPTION("void Schroedinger::get_lattice_constants Unrecognized lattice");
    }
  } else
  {
    NEMO_EXCEPTION("void Schroedinger::get_lattice_constants Wrong domain");
  }  

  alattice[0]=ax;
  alattice[1]=ay;
  alattice[2]=az;

  if(my_rank==0) {
    //   cout<<"UNSTRAINED LATTICE ax ay az:"<<alattice[0]<<" "<<alattice[1]<<" "<<alattice[2]<<endl;
  }

  if (constant_strain)
  {

    for (unsigned int i=0; i<3; i++)
    {
      alattice1[i]=0.0;
      for (unsigned int j=0; j<3; j++)
      {
        alattice1[i]+=strain_matrix[i][j]*alattice[j];
      }
    }

    for (unsigned int i=0; i<3; i++)
    {
      alattice[i]=alattice1[i];
    }

    if(my_rank==0)
    {
      //cout<<"STRAINED LATTICE ax ay az:"<<alattice[0]<<" "<<alattice[1]<<" "<<alattice[2]<<endl;
    }

  } //constant strain

}



// end

void Schroedinger::get_data(const std::string&, const unsigned int input_index, double& result)
{
  std::string tic_toc_prefix="Schroedinger(\""+get_name()+"\")::get_data ";
  NemoUtils::tic(tic_toc_prefix);
  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
  const AtomicStructure& atoms = domain->get_atoms();
  const std::map< unsigned int, AtomStructNode >& temp_lattice=atoms.lattice();
  std::map<unsigned int,AtomStructNode>::const_iterator cit=temp_lattice.find(input_index);
  NEMO_ASSERT(cit!=temp_lattice.end(),tic_toc_prefix+"have not found the input atom id\n");
  const AtomStructNode& nd        = cit->second;
  const Atom*           atom      = nd.atom;
  const Material*       material  = atom->get_material();
  const Crystal*        crystal   = material->get_crystal();
  double cell_volume_per_atom = crystal->calculate_primitive_cell_volume()/crystal->get_number_of_prim_cell_atom();
  double periodic_cell_volume=domain->return_periodic_space_volume();
  if(periodic_cell_volume>0.0)
    result=cell_volume_per_atom; //*periodic_cell_volume;
  else
    result=cell_volume_per_atom;
  NemoUtils::toc(tic_toc_prefix);
}

//Bozidar added this dummy implementation.
void Schroedinger::set_job_done_momentum_map(const std::string* variable_name, const std::vector<NemoMeshPoint>* momentum_point, const bool input_status)
{
  std::string tic_toc_prefix = "Schroedinger(\"" + tic_toc_name + "\")::set_job_done_momentum_map ";
  NemoUtils::tic(tic_toc_prefix);
  std::string error_prefix = "Schroedinger(\"" + this->get_name() + "\")::set_job_done_momentum_map: ";

  MsgLevel msg_level = msg.get_level();
  msg.set_level(MsgLevel(5));

  if(variable_name != NULL)
    msg << variable_name;

  if(momentum_point != NULL)
    msg << input_status;

  //Reset the messaging level
  msg.set_level(msg_level);

  NemoUtils::toc(tic_toc_prefix);
}

void Schroedinger::get_distance_map(std::map<std::pair<unsigned int,unsigned int>, double>& output_map, const Domain*)
{
  //1. determine whehter the distance_map is filled already
  if(distance_map.size()==0)
  {
    //2. loop over all atoms
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    DOFmapInterface&                dof_map = get_dof_map();
    const AtomicStructure& atoms   = domain->get_atoms();
    const map< unsigned int,AtomStructNode >& lattice = atoms.lattice();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    vector<const Atom_neighbour*> active_neighbors;  //active neighbours of an atom of this domain
    for(;it!=end;++it)
    {
      const AtomStructNode& nd        = it.node();
      //const Atom*           atom      = nd.atom;
      const unsigned int    atom_id   = it.id();
      const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
      domain->get_active_interaction_neighbours(atom_id, &active_neighbors);
      //2.1 loop over neighbor atoms
      for(unsigned int i=0;i<active_neighbors.size();i++)
      {
        //2.2 determine the distance of this atom and its neighbor
        const unsigned int neighbor_id=active_neighbors[i]->atom_id;
        const AtomStructNode& neighbor_nd = lattice.find(neighbor_id)->second;
        std::vector<double> distance_vector(3,nd.position[0]-neighbor_nd.position[0]);
        distance_vector[1]=nd.position[1]-neighbor_nd.position[1];
        distance_vector[2]=nd.position[2]-neighbor_nd.position[2];
        double temp_distance=NemoMath::vector_norm_3d(&(distance_vector[0]));
        //2.4 determine the neighbor DOFs
        const map<short, unsigned int>* neighbor_atom_dofs = dof_map.get_atom_dof_map(neighbor_id);
        //2.4 store result in output_map for all atom orbital-dofs
        //2.4.1 loop over the atom dofs
        map<short, unsigned int>::const_iterator atom_dof_cit=atom_dofs->begin();
        for(;atom_dof_cit!=atom_dofs->end();++atom_dof_cit)
        {
          //2.4.2 loop over the neighbor atom dofs
          map<short, unsigned int>::const_iterator neighbor_atom_dof_cit=neighbor_atom_dofs->begin();
          for(;neighbor_atom_dof_cit!=neighbor_atom_dofs->end();++neighbor_atom_dof_cit)
          {
            std::pair<unsigned int,unsigned int> temp_coordinate_pair(atom_dof_cit->second,neighbor_atom_dof_cit->second);
            distance_map[temp_coordinate_pair]=temp_distance;
          }
        }
      }
    }
  }
  output_map=distance_map;
}

void Schroedinger::get_distance_vector_map(std::map<std::pair<unsigned int,unsigned int>, pair<unsigned int, vector<double> > >& output_map, const Domain*, vector<unsigned int>* regions)
{
  //1. determine whehter the distance_map is filled already
  if(distance_vector_map.size()==0)
  {
    //2. loop over all atoms
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    DOFmapInterface&                dof_map = get_dof_map();
    const AtomicStructure& atoms   = domain->get_atoms();
    const map< unsigned int,AtomStructNode >& lattice = atoms.lattice();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    vector<const Atom_neighbour*> active_neighbors;  //active neighbours of an atom of this domain

    bool use_regions = false;
    if(regions != NULL)
      use_regions = true;

    //loop through all relevant atoms 
    //std::map<unsigned int, AtomStructNode>::iterator relevant_it = relevant_atoms.begin();
    for(;it!=end;++it)
    {
      const AtomStructNode& nd        = it.node();
      //const Atom*           atom      = nd.atom;
      const unsigned int    atom_id   = it.id();
      const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
      domain->get_active_interaction_neighbours(atom_id, &active_neighbors);

      bool use_atom = true;
      unsigned int region = -1;
      if(use_regions)
      {
        unsigned int nd_reg = nd.region;
        //std::vector<unsigned int>::iterator region_it = regions->find(nd_reg);
        std::vector<unsigned int>::iterator region_it = find(regions->begin(), regions->end(), nd_reg);
        if(region_it==regions->end())
          use_atom = false;
        else
          region = (*region_it);
      }

      if(use_atom)
      {
        //2.1 loop over neighbor atoms
        for(unsigned int i=0;i<active_neighbors.size();i++)
        {
          //2.2 determine the distance of this atom and its neighbor
          const unsigned int neighbor_id=active_neighbors[i]->atom_id;
          const AtomStructNode& neighbor_nd = lattice.find(neighbor_id)->second;
          std::vector<double> distance_vector(3,std::abs(nd.position[0]-neighbor_nd.position[0]));
          distance_vector[1]=std::abs(nd.position[1]-neighbor_nd.position[1]);
          distance_vector[2]=std::abs(nd.position[2]-neighbor_nd.position[2]);

          //2.4 determine the neighbor DOFs
          const map<short, unsigned int>* neighbor_atom_dofs = dof_map.get_atom_dof_map(neighbor_id);
          //2.4 store result in output_map for all atom orbital-dofs
          //2.4.1 loop over the atom dofs
          map<short, unsigned int>::const_iterator atom_dof_cit=atom_dofs->begin();
          for(;atom_dof_cit!=atom_dofs->end();++atom_dof_cit)
          {
            std::pair<unsigned int, unsigned int> temp_same_coordinate_pair(atom_dof_cit->second,atom_dof_cit->second);
            std::vector<double> same_distance(3,0.0);
            std::pair<unsigned int, std::vector<double> > same_distance_pair(region, same_distance);
            distance_vector_map[temp_same_coordinate_pair] = same_distance_pair; 

            //2.4.2 loop over the neighbor atom dofs
            map<short, unsigned int>::const_iterator neighbor_atom_dof_cit=neighbor_atom_dofs->begin();
            for(;neighbor_atom_dof_cit!=neighbor_atom_dofs->end();++neighbor_atom_dof_cit)
            {
              std::pair<unsigned int,unsigned int>  temp_coordinate_pair(atom_dof_cit->second,neighbor_atom_dof_cit->second);
              std::pair<unsigned int, std::vector<double> > distance_vector_pair(region, distance_vector);
              distance_vector_map[temp_coordinate_pair]=distance_vector_pair;
            }
          }
        }
      }
    }
  }
  output_map = distance_vector_map;
}

void Schroedinger::set_input_options_map()
{
  Simulation::set_input_options_map();

  set_input_option_map("atom_order_from",InputOptions::NonReq_Def("",
      "Determines ordering of atoms in Hamiltonian when repartitioner solver is attached to DOFmap. Most likely, set this to the name of the repartition solver"));
  set_input_option_map("automatic_threshold",InputOptions::NonReq_Def("false",
      "If true, then the code automatically determines the energy threshold that separates electrons and holes (i.e. conduction and valence bands) "
      "by looking at the bulk band edges and the electrostatic potential. When iterative band structure solvers (like Krylov-Schur) are used "
      "this option causes the shift value to be equal to the threshold mentioned above. The actual value of the threshold depends on other options. "
      "For more details search for _automatic_threshold variable in the code above. This option is also used for density calculations in Schroedinger "
      "solver and not just band structure calculations. Possible values: true/false."));
  set_input_option_map("beta_degradation",InputOptions::NonReq_Def("0.","Numerical parameter if eigensolver uses lanczos"));
  set_input_option_map("brillouin_zone_meshsize",InputOptions::NonReq_Def(" 0.1",
      "If calculate_brillouin_zone is true, this determines the mesh size. The spacing in every direction, in pi/a"));
  set_input_option_map("calculate_automatic_shift_subbandbased",InputOptions::NonReq_Def(" false",
      "implemented by R. Kotlyar. When true, the convergence scheme based on determining automatic shifts from subband energies is specified." ));
  set_input_option_map("calculate_brillouin_zone",InputOptions::NonReq_Def(" 0.1",
      "When set to true, the entire Brillouin zone will be meshed and computed."));
  set_input_option_map("calculate_elastic_overlap",InputOptions::NonReq_Def(" false", "calculates scattering-related form factors. deprecated" ));
  set_input_option_map("calculate_integrated_electron_density",InputOptions::NonReq_Def(" false",
      "implemented by R. Kotlyar. These options were added to fix the shift of the eigenvalue solver based on monitoring the subband energies from previ- ous iterations. Usually these shifts are determined by conduc- tion or valence band minima/maxima. Roza found that this convergence scheme works the best for high electric fields in the presence of large local band bendings.")); //
  set_input_option_map("calculate_integrated_hole_density",InputOptions::NonReq_Def(" false",
      "implemented by R. Kotlyar. If option is specified and spatially varying threshold = false, the integrated the total carrier density in cm-dim is computed by integration over the confinement dimensions (e.g. for wires this yields 1/cm). As similar option exists for electrons."));
  set_input_option_map("Center_Position",InputOptions::NonReq_Def("(0,0,0)", "related to region: (min+max)/2 for magnetic_vector_potential A"));
  set_input_option_map("charge_model",InputOptions::NonReq_Def("electron_hole",
      "In electron_hole you have negatively charged electrons and positively holes. In electron_core you have negatively charged electrons and positively charged ions. Can be electron_core or electron_hole. When set to electron hole (most cases), then an energy threshold separates between electrons and holes. When set to electron core model all states are assumed to be electrons"));
  set_input_option_map("check_every_steps_number",InputOptions::NonReq_Def("0",
      "Numerical parameter if eigensolver uses lanczos. Try a value between 20 and 500"));
  set_input_option_map("compute_edges_masses",InputOptions::NonReq_Def(" 0.001",
      "If set to true, some band edges and masses will be computed and written to screen. This option was tested with bulk only and fails to compute correct transverse masses for X- and L- valleys in the presence of spin-orbit interaction."));
  set_input_option_map("compute_edges_masses_delta",InputOptions::NonReq_Def(" 0.001",
      "delta K to calculate numerical derivative of the band dispersion.  in pi/a. stepsize for the finite difference scheme employed in compute edges masses,")); // in pi/a
  set_input_option_map("convergence_limit",InputOptions::NonReq_Def("1e-8",
      "Convergence limit for iterative solution methods (accuracy of eigenvalues). " ));
  set_input_option_map("convergence_method",InputOptions::NonReq_Def("",
      "Numerical parameter if eigensolver uses lanczos. Can be full_convergence or partial_convergence"));
  set_input_option_map("cutoff_distance_to_bandedge",InputOptions::NonReq_Def(" 0.2*(Ec_min-Ev_max)",
      "Energy cutoff which is added to bandedge for holes and subtracted for electrons. Setting this to some value Delta (in eV) leads to electron thresholds at Ec - ePhi(x) - Alpha and hole thresholds at Ev - e Phi(x) + Delta."));
  set_input_option_map("dE_for_elastic_overlap",InputOptions::NonReq_Def("0.0,"
      "Warning: option calculate_elastic_overlap is deprecated. "
      "Energy threshold for calculating overlap for elastic scattering. Overlap is calculated if difference between energy_for_elastic_overlap option and energy eigenvalue is smaller than dE. "
      "Only takes effect if take_all_states_and_no_overlap option is false."));
  set_input_option_map("Dirac_gaussian_width",InputOptions::NonReq_Def("1e-3",
      "Width of Dirac delta function used for calculating scattering rate."));
  set_input_option_map("doping_density",InputOptions::NonReq_Def("0.0",
      "Doping density of material. Only used if calculate_Fermi_level is included in job list. Doping type (i,n,p) set by option doping_type."));
  set_input_option_map("doping_type",InputOptions::NonReq_Def("i","Type of doping"));
  set_input_option_map("DOS_broadening_model",InputOptions::NonReq_Def("2",
      "(for DOS calculation, default 2) 0: Lorentzian broadening, 1: Exponential broadening, 2: cosh broadening."));
  set_input_option_map("DOS_broadening",InputOptions::NonReq_Def(" dE", "(for DOS calculation) Broadening strength in eV."));
  set_input_option_map("DOS_energy_grid",InputOptions::NonReq_Def(" energy_input",
      "(for DOS calculation) Determines homogeneous energy grid in the form (Emin,Emax,dE)."));
  set_input_option_map("DOS_max_energy",InputOptions::NonReq_Def("20.0", "(for DOS calculation) alternative specification of Emax."));
  set_input_option_map("DOS_min_energy",InputOptions::NonReq_Def("-20.0", "(for DOS calculation) alternative specification of Emin."));
  set_input_option_map("DOS_points",InputOptions::NonReq_Def("100", "(for DOS calculation) alternative specification of the number of energy points."));
  set_input_option_map("DOS_spin_factor",InputOptions::NonReq_Def(" 1.0" , "(for DOS calculation, default 1) optional multiplicative factor for DOS."));
  set_input_option_map("eigen_values_solver",InputOptions::NonReq_Def("lapack",
      "This defines the type of the numerical eigen value solver.\nPossible types are:\n\"lapack\"\n\"krylovschur\"\n\"lanczos\"\n\"arpack\"\n\"block_lanczos\"\n\"lanczos_then_krylovschur") );
  set_input_option_map("eigen_values_solver",InputOptions::NonReq_Def("",
      "Which eigenvalue solver to use. Setting lapack always com- putes all eigenvalues and is feasible only for very small sys- tems. Recommended choices are krylovschur and arpack. Other choices are arnoldi, jd, gd."));
  //set_input_option_map("chem_pot",InputOptions::NonReq_Def("0.0",NemoDictionary().i18n("chem_pot")));
  set_input_option_map("chem_pot",InputOptions::NonReq_Def("0.0",NemoDictionary().i18n("chem_pot")));
  set_input_option_map("electron_chem_pot",InputOptions::NonReq_Def("0.0",
      "Chemical potential used to calculate density of states times Fermi derivative. Has units of eV."));//[eV]
  set_input_option_map("electron_temperature",InputOptions::NonReq_Def("300","Temperature of the system"));//[K]
  set_input_option_map("energy_for_elastic_overlap",InputOptions::NonReq_Def("0.0", "Energy used for calculating matrix elements for wave functions"));
  set_input_option_map("energy_maximum",InputOptions::NonReq_Def("0.0" , "Eigenslover Lanczos will search the energy within energy_maximum & energy_minimum"));
  set_input_option_map("energy_minimum",InputOptions::NonReq_Def("0.0" , "Eigenslover Lanczos will search the energy within energy_maximum & energy_minimum"));
  set_input_option_map("energy_threshold_for_shift",InputOptions::NonReq_Def("enthresh_foresolvershift",
      "To determine the position of the next shift to be passed to eigensolver based on subband energies in previous iteration. R. Kotylar."));
  set_input_option_map("epsilon_matrix_crystal",InputOptions::NonReq_Def("strain_matrix",
      "Constant strain tensor matrix, given in crystal system coordinates. One needs to include strain in the job list in order to use this. Superimposed strain matrix in the crystallographic system."));
  set_input_option_map("epsilon_matrix",InputOptions::NonReq_Def("strain_matrix",
      "Constant strain tensor matrix, given in laboratory system co- ordinates. One needs to include strain in the job list in order to use this. Superimposed strain matrix in the laboratory system where crystal axis might have been rotated."));
  set_input_option_map("extra_points",InputOptions::NonReq_Def("NULL","Extra K points set by user"));
  set_input_option_map("fixed_shift",InputOptions::NonReq_Def("0.0",
      "Sets the shift to be used by eigenvalue solvers (except when electron_core model: check option shift)."));
  set_input_option_map("init_check_steps_number",InputOptions::NonReq_Def("20",
      "lanczos option, for checking convergence after how many iterations"));
  set_input_option_map("input_shift_subbandbased_for_restart",InputOptions::NonReq_Def("false",
      "When true, the shift is read in from file and is used as initial shift to eigensolver. R. Kotylar"));
  set_input_option_map("input_shiftfilename_for_restart",InputOptions::NonReq_Def(" string("")",
      "Filename for input_shift_subbandbased_for_restart. R. Kotylar"));
  set_input_option_map("integration_order",InputOptions::NonReq_Def(" 1","Sets the order of integration for Gaussian quadrature method"));
  set_input_option_map("job_list",InputOptions::NonReq_Def("job_list",
      "A list that deterimes what is done. Choose from assemble_H, passivate_H, include_strain_H, include_shear_strain_H, calculate_band_structure, electron_density, derivative_electron_density_over_potential, hole_density, derivative_hole_density_over_potential, spin, DOS. assemble_H is activated by any other option automatically."));
  set_input_option_map("k_degeneracy",InputOptions::NonReq_Def(" 1.0",
      "The computed density is multiplied by this number to account for k-space degeneracy. E.g. when kxmax and kymax are set in a simulation with 2D k-space, k degeneracy should be 4."));
  set_input_option_map("k_points",InputOptions::NonReq_Def("k_pointsV",
      "Only relevant for calculate band structure. This parame- ter is a list of points in k-space along which the band structure is calculated."));
  set_input_option_map("k_space_basis",InputOptions::NonReq_Def("reciprocal",
      "(Default: cartesian) Sets the basis in which k points is specified. When set to reciprocal the coordinates given in k points are assumed to be w.r.t. the reciprocal lattice vectors b1 =a2xa3 / a1.(a2xa3 ) etc."));
  set_input_option_map("k0",InputOptions::NonReq_Def("[0 0 0]", "k0 vector to solve eigenvalues at this particular k-point"));
  set_input_option_map("kpoint_to_monitor_subbands_for_shift",InputOptions::NonReq_Def(" kmagref_foresolvershift",
      "The k-point where to look for shift. default is 0. R. Kotylar"));
  set_input_option_map("kpoint_tolerance_to_monitor_subbands_for_shift",InputOptions::NonReq_Def(" kmagreftol_foresolvershift",
      "The k-tolerance when looking for a k-point on k-grid. Default is 0.001. R. Kotylar"));
  set_input_option_map("kxmax",InputOptions::NonReq_Def("1.0",
      "Upper boundary in 2 pi/a of the simulated kx space. kx ranges from 0 to this number."));
  set_input_option_map("kxmin",InputOptions::NonReq_Def("0.0",
      "Lower boundary in 2pi/a range of the simulated kx space. kx begins from zero by default."));
  set_input_option_map("kymax",InputOptions::NonReq_Def("1.0",
      "Upper boundary in 2 pi/a of the simulated kx space. ky ranges from 0 to this number."));
  set_input_option_map("kymin",InputOptions::NonReq_Def("0.0,"
      "Lower boundary in 2pi/a range of the simulated ky space. ky begins from zero by default."));
  set_input_option_map("kzmax",InputOptions::NonReq_Def("1.0",
      "Upper boundary in 2 pi/a of the simulated kz space. kz ranges from 0 to this number."));
  set_input_option_map("kzmin",InputOptions::NonReq_Def("0.0,"
      "Lower boundary in 2pi/a range of the simulated kz space. kz begins from zero by default."));
  set_input_option_map("lanczos_iseed",InputOptions::NonReq_Def("0.","Numerical parameter if eigensolver uses lanczos"));
  set_input_option_map("linear_solver",InputOptions::NonReq_Def("",
      "Linear solver employed in the shift-and-invert operation. This should be preferable a direct linear solver since the LU fac- torization can be reused during the Krylov iterations."));
  set_input_option_map("Magnetic_Field",InputOptions::NonReq_Def("", "Magnetic field value in the form of (Bx, By, Bz)"));
  set_input_option_map("Magnetic_Gauge",InputOptions::NonReq_Def("Symmetric", "options:  Symmetric (default) , Asymmetric_x  , Asymmetric_y , Asymmetric_z."));
  set_input_option_map("max_number_iterations",InputOptions::NonReq_Def("0", "Maximum number of Krylov iterations (irrelevant for lapack)."));
  set_input_option_map("maximum_delta_k",InputOptions::NonReq_Def("1e80",
      "Upper bound of norm of delta_k to calculate scattering rate in units of 1/nm.")); //[1/nm]
  set_input_option_map("maximum_scattering_angle",InputOptions::NonReq_Def("180.0", "Option connected to computation of scattering matrix elements"));
  set_input_option_map("minimum_delta_k",InputOptions::NonReq_Def("0.0",
      "Lower bound of norm of delta_k to calculate scattering rate in units of 1/nm.")); //[1/nm]
  set_input_option_map("minimum_scattering_angle",InputOptions::NonReq_Def("0.0", "Option connected to computation of scattering matrix elements"));
  set_input_option_map("monitor_convergence",InputOptions::NonReq_Def("false",
      "When set to true, terminal output related to the Krylov iteration is generated."));
  set_input_option_map("mpd",InputOptions::NonReq_Def("0","Specialized numerical option. Refer to the SLEPc user manual for the meaning."));
  set_input_option_map("mumps_ordering",InputOptions::NonReq_Def("","Relevant only for eigen values solver=mumps. Choose between pord and metis."));
  set_input_option_map("modes_for_current",InputOptions::NonReq_Def("",
      "Relevant for current calculation. Define the number of modes to be extracted from E-k while calc. ballistic TOB current."));//<srm>
  set_input_option_map("ncv",InputOptions::NonReq_Def("0", "Specialized numerical option (size of the Krylov subspace)."));
  set_input_option_map("no_of_spatial_division",InputOptions::NonReq_Def("256",
      "No of cores used for parallelizing the device(for tb_basis = kp only)"));
  set_input_option_map("number_of_eigenvalues_to_use",InputOptions::NonReq_Def("0","Number of requested eigenvalues"));
  set_input_option_map("number_of_eigenvalues",InputOptions::NonReq_Def("0","Number of eigenvalues to compute (irrelevant for lapack)."));
  set_input_option_map("number_of_k_points",InputOptions::NonReq_Def("10",
      "This list gives the uniform discretization of each segment set by the k points parameter (note that specifying N points means N - 1 segments). For density calculations, setting this parameter to 0 leads to computation of k = 0 only and application of an analytical formula that assumes parabolic subbands."));
  set_input_option_map("number_of_nodes",InputOptions::NonReq_Def("()", "number of nodes for each k-segment"));
  set_input_option_map("output_eigenstate_energy",InputOptions::NonReq_Def("0.0", "Energy around which eigenpair has to be output."));
  set_input_option_map("output_k_point",InputOptions::NonReq_Def("k0","Value of k-point dumped as output"));
  set_input_option_map("output_precision",InputOptions::NonReq_Def("12", "Accuracy of saved eigenvalues within output file."));
  set_input_option_map("output_shift_subbandbased_for_restart",InputOptions::NonReq_Def("false",
      "When true, at the end of simulation the file can be output with the last eigensolver shift information. Using this shift and the last potential, the next simulation can be restarted from the last state of simulation. R. Kotylar."));
  set_input_option_map("output",InputOptions::NonReq_Def("output_list_v",
      "Choose from Hamiltonian, energies, k-points, DOS, electron density, electron density VTK, hole density, hole density VTK, ion density, eigenfunctions, eigenfunctions k0, eigenfunctions VTK, eigenfunctions VTK k0, eigenfunctions Silo, eigenfunctions Silo k0, spin.")); //list of outputs, options are energies, ???eigenfunctions_Silo
  set_input_option_map("p_parameter",InputOptions::NonReq_Def("0",
      "It's the block size for block Lanczos. If the degeneracy of the eigenvalues is known, then set it to that value, otherwise start with 4."));
  set_input_option_map("parallelize_here",InputOptions::NonReq_Def("true" ,
      "When set to false, some other simulation (like Propagator) can determine the parallelization.")); //must be true
  set_input_option_map("Peierls_Phase",InputOptions::NonReq_Def("true", "true/false for applying peierls phase for nonzero magnetic field.."));
  set_input_option_map("potential_solver",InputOptions::NonReq_Def("",
      "The (optional) name of the simulation object where the electrostatic potential is drawn from.")); //the name of the potential solver
  set_input_option_map("preconditioner",InputOptions::NonReq_Def("lu",
      "This option controls how the eigenvalue problem is transformed before solution\nPossible options are:\n\"lu\"\n\"mumps\"...work ongoing\n. (default: lu) Preconditioner employed in the shift-and-invert operation."));
  set_input_option_map("project_DOS_",InputOptions::NonReq_Def("false", " This option allows to calculate the project density of states "));
  set_input_option_map("rate_of_shift_changes_between_iterations",InputOptions::NonReq_Def("enrate_foresolvershift",
      "To determine the position of the next shift to be passed to eigensolver based on subband energies in previous iteration. R. Kotylar."));
  set_input_option_map("resolution",InputOptions::NonReq_Def("0.0","The resolution to be used for the eigensolver"));
  set_input_option_map("reuse_hamiltonian",InputOptions::NonReq_Def("true",
      "if this option is false, hamiltonian would be recalculated every iteration, for different k always it is recalculated"));
  set_input_option_map("shift",InputOptions::NonReq_Def("0.001",
      "The eigensolver will search for eigenvalues around the shift value. This seems to only be used for "
      "electron_core model. fixed_shift is used otherwise."));
  set_input_option_map("solver_transformation_type",InputOptions::NonReq_Def("sinvert",
      "(default: sinvert) Use sinvert for shift-and-invert, shift for shift."));
  set_input_option_map("sort_like_electrons",
      InputOptions::NonReq_Def(" true if job list has electron density or band structure or electron core model. false if job list "
          "has hole density.","This option controls the way the eigenenergies are sorted. When true (electrons) then ascending and when false (holes) then "
          "descending. Possible values: true/false."));
  set_input_option_map("spatially_varying_threshold",InputOptions::NonReq_Def(" false",
      "If true, then the threshold energy discriminating between electrons and holes will vary spatially according to the elec- trostatic potential. Needed in Schroedinger-Poisson simulations."));
  set_input_option_map("State_solver",InputOptions::NonReq_Def("","Gets the external state solver"));
  set_input_option_map("strict_threshold",
      InputOptions::NonReq_Def("true if job list has electron density or hole density. false if job list has band structure or if "
          "electron_core charge model is used.","Solve for either conduction (electron density) or valence (hole density) bands depending on "
          "option sort_like_electrons. Possible values: true/false."));
  set_input_option_map("temperature",InputOptions::NonReq_Def("300", "Temperature (in kelvin) of the device to be simulated"));
  set_input_option_map("tb_basis",InputOptions::NonReq_Def("sp3",NemoDictionary().i18n("tb_basis")));
  set_input_option_map("tic_toc_name",InputOptions::NonReq_Def("get_name()", "Name for advanced timing output")); //
  set_input_option_map("tolerance",InputOptions::NonReq_Def("0.0", "The judgement of the convergence. Once the target reaches the tolerance then end the loop."));
  set_input_option_map("solve_on_single_replica",InputOptions::NonReq_Def("false", "Gather density to the single rank only"));
  set_input_option_map("Neumann_BC",InputOptions::NonReq_Def("false",
      "Set Neumann BC for hamiltonian, which can generate a good basis for EM LRA, implemented by Lang Zeng")); //
  set_input_option_map("set_valley",InputOptions::NonReq_Def("DEFAULT", "set valley name to get Material properties")); //
  set_input_option_map("AP_vca_model",InputOptions::NonReq_Def("false", "strained SiGe VCA model as implemented by A. Paul, IEEE EDL v31 no4 2010")); //
  set_input_option_map("Zeeman_Splitting",InputOptions::NonReq_Def("true", "true/false for applying zeeman splitting for nonzero magnetic field."));
  set_input_option_map("direct_order_of_partitions_$neighbor_domain_name$",InputOptions::NonReq_Def(" true", "Orders neighbor domain partitions directly if true,"
      " in reverse if false. Considered only with neighbor domain repartitioner is provided."));
  set_input_option_map("tic_toc_name",InputOptions::NonReq_Def("", "Tic toc name to be used for saving timing results for this simulation."));
  set_input_option_map("use_neighbour_repartitioner_$neighbor_domain_name$",InputOptions::NonReq_Def("", "Repartitioner solver name for neighbor domain."));
  set_input_option_map("gaussian_quadrature_order_automatic",InputOptions::NonReq_Def("false", "Affects k-space integration for periodic directions. "
      "If option integration_order is not defined and if gaussian_quadrature_order_automatic = true, then for number_of_k_points <= 22 use Gaussian quadrature with single "
      "partition, otherwise use simple rectangular with number of elements equal to number_of_k_points. In both cases the actual number of k points equals number_of_k_points."));
  set_input_option_map("take_all_states_and_no_overlap",InputOptions::NonReq_Def("false","This needs the option \"calculate_elastic_overlap\" to be true."
      " Allows to switch the Schroedinger from solving the scattering overlap for a subset of states to "
      "collect all states and make them ready for other solvers to get via get_data. Primary purpose of this is to support the fitting of TB to DFT including the TB wavefunctions\n"));
  set_input_option_map("sparse_storage_format",InputOptions::NonReq_Def("csr",
      "Sparse storage format in which PETSc will allocate sparse matrices; options available are \"csr\" (compressed sparse row) and \"bsr\" (block sparse row)"));
  set_input_option_map("BSR_block_size", InputOptions::NonReq_Def("10",
      "Size of matrix blocks when using block sparse row (BSR) format with sparse_storage_format option; ignored if format is compressed sparse row (CSR)"));
  set_input_option_map("spin_resolved",InputOptions::NonReq_Def("false",
      "if this option is enabled all parameters will be spin dependant Up/Down"));
  set_input_option_map("use_monomials",InputOptions::NonReq_Def("true",
      "calculate monomials table based on the orbitals")); //TODO DM: this documentation need to be improved.
  set_input_option_map("k_rotational_symmetry",InputOptions::NonReq_Def("false", "Assumes rotational symmetry in a 2D k-space\n"));
  set_input_option_map("basis_generator",InputOptions::NonReq_Def("",
      "The (optional) name of the simulation object where the basis set for transforming the Hamiltonian is drawn from.")); //the name of the mode space solver
  set_input_option_map("OMEN_style_transverse_k_points",InputOptions::NonReq_Def("false",
      "Construct the same transverse k (wave vector) grid as in OMEN. This was necessary in order to do benchmarking against OMEN. "
      "For now implemented for UTB only (1D transverse k)."));
  set_input_option_map("custom_transverse_k_points",InputOptions::NonReq_Def("false",
      "Transverse k (wave vector) grid which has denser points around the center of "
      "the Brillouin zone. This option is used along with dense_grid_ratio and grid_density options. For now implemented for UTB only (1D transverse k)."));
  set_input_option_map("dense_grid_ratio",InputOptions::NonReq_Def("0.2",
      "The ratio of the total number of k (wave vector) points that will belong to the denser grid. "
      "Look at description of custom_transverse_k_points for more details."));
  set_input_option_map("grid_density",InputOptions::NonReq_Def("1.0","How many times denser will the dense part of the transverse k (wave vector) grid be. "
      "Look at description of custom_transverse_k_points for more details."));
  set_input_option_map("electron_hole_heuristics",InputOptions::NonReq_Def("sharp",
      "Given the atomically resolved energy threshold between electrons and holes, this "
      "option tells how to actually determine whether a particle is an electron or hole. Two options available: sharp (sharp transition at the threshold) "
      "and omen_smooth (smooth transition around the threshold, like in OMEN code)"));
  set_input_option_map("reinit_parallelize",InputOptions::NonReq_Def("false",
      "When Schroedinger solver is reinitialized, reinitialize the MPI parallelization as well."));
  set_input_option_map("allow_reinit_k_points",InputOptions::NonReq_Def("false",
      "When Schroedinger solver is reinitialized, reinitialize the k (wave vector) space "
      "object in order to be able to change the values of k points. Implemented for use with band solvers associated with adaptive grid."));
  set_input_option_map("noise_potential",InputOptions::NonReq_Def("0.0","noise potential used in qtbm to break degenerate modes."));
  set_input_option_map("seed",InputOptions::NonReq_Def("-1","seed for noise potential."));
  set_input_option_map("passivate_with_oxide",InputOptions::NonReq_Def("false","passivate with oxide or not. only works for Si-sp3d5sstar_yaohua so far."));
  set_input_option_map("surface_atom_potential",InputOptions::NonReq_Def("0.0","potential added to diagonal of surface atom Hamiltonian."));
  set_input_option_map("passivation_energy",InputOptions::NonReq_Def("0.0","energy input of passivation atom greens function, set to constant parameter so far."));
  set_input_option_map("radius_for_passivate_neighbors",InputOptions::NonReq_Def("0.0","radius controls the effective passivation atoms, e.g. 2NN model of black phosphorus"
      "if it is beyond the radius, the passivation atoms are ignored."));
  /*------------- For K.P only (BEGIN) -------------*/
  set_input_option_map("kp_band_number",InputOptions::NonReq_Def("8",
      "8 band for both conduction and valence bands of direct band gap material / 6 for valence band only / 4 for rhombohedral crystals of TI"));
  set_input_option_map("eff_mass_dump",InputOptions::NonReq_Def("false",
      "true: for effective mass calculation / false: for disabling it "));
  set_input_option_map("Burt_Foreman_ordering",InputOptions::NonReq_Def("false",
      "true: use Burt-Foreman operator ordering for heterostructures / false: use symmetrized operator ordering for heterostructures"));
  set_input_option_map("Foreman_strategy",InputOptions::NonReq_Def("false",
      "true: use Foreman strategy for parameter adjustment / false: no parameter adjustment"));
  set_input_option_map("orientation_x",InputOptions::NonReq_Def("(1,0,0)",
      "crystal orientation of x direction, if device coordinate is not aligned with crystal coordinate"));
  set_input_option_map("orientation_y",InputOptions::NonReq_Def("(0,1,0)",
      "crystal orientation of y direction, if device coordinate is not aligned with crystal coordinate"));
  set_input_option_map("is_kp_strain_on",InputOptions::NonReq_Def("false",
      "true: for enabling strain calculation with k.p/ false: for disabling strain calculation with k.p"));
  set_input_option_map("is_piezoelectric_effect_on",InputOptions::NonReq_Def("false",
      "true/false; for enabling piezoelectric calculation with k.p"));
  set_input_option_map("is_spatially_parallelized",InputOptions::NonReq_Def("false",
      "true/false; Whether the device is spatially parallelized;"));
  set_input_option_map("is_vff_strain_on",InputOptions::NonReq_Def("false",
      "true/false; for enabling VFF strain calculation with k.p;"));
  set_input_option_map("input_stress_or_strain",InputOptions::NonReq_Def("strain",
      "stress: for giving stress as input/ strain: for giving strain as input directly"));
  set_input_option_map("strain_calc_option",InputOptions::NonReq_Def("uniaxial",
      "hydrostatic: for calculating the effect of hydrostatic strain/ biaxial: for calculating the effect of biaxial strain/ uniaxial: for calculating the effect of uniaxial strain (use this option only when giving strain as input)"));
  set_input_option_map("strained_material_names",InputOptions::NonReq_Def("",
      "use this option to apply strain to a set of materials in a heterostructure"));
  set_input_option_map("epsilon_vector_for_each_material",InputOptions::NonReq_Def("",
      "when input_stress_or_strain= strain and strained_material_names is set, the strain vectors for the strained materials. Specify (e_xx,e_yy,e_zz,e_yz,e_xz,e_xy) in the device coordinate system for each material"));
  set_input_option_map("stress_vector_for_each_material",InputOptions::NonReq_Def("",
      "when input_stress_or_strain= stress and strained_material_names is set, the stress vectors for the strained materials. Specify (stress_xx,stress_yy,stress_zz,stress_yz,stress_xz,stress_xy) in the device coordinate system for each material"));
  set_input_option_map("e_xx",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= strain and strained_material_names is not set, the strain will be applied to all materials, use this option to set strain component e_xx in the device coordinate system "));
  set_input_option_map("e_yy",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= strain and strained_material_names is not set, the strain will be applied to all materials, use this option to set strain component e_yy in the device coordinate system "));
  set_input_option_map("e_zz",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= strain and strained_material_names is not set, the strain will be applied to all materials, use this option to set strain component e_zz in the device coordinate system "));
  set_input_option_map("e_xy",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= strain and strained_material_names is not set, the strain will be applied to all materials, use this option to set strain component e_xy in the device coordinate system "));
  set_input_option_map("e_yz",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= strain and strained_material_names is not set, the strain will be applied to all materials, use this option to set strain component e_yz in the device coordinate system "));
  set_input_option_map("e_xz",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= strain and strained_material_names is not set, the strain will be applied to all materials, use this option to set strain component e_xz in the device coordinate system "));
  set_input_option_map("stress_xx",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= stress and strained_material_names is not set, the stress will be applied to all materials, use this option to set stress component stress_xx in the device coordinate system "));
  set_input_option_map("stress_yy",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= stress and strained_material_names is not set, the stress will be applied to all materials, use this option to set stress component stress_yy in the device coordinate system "));
  set_input_option_map("stress_zz",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= stress and strained_material_names is not set, the stress will be applied to all materials, use this option to set stress component stress_zz in the device coordinate system "));
  set_input_option_map("stress_xy",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= stress and strained_material_names is not set, the stress will be applied to all materials, use this option to set stress component stress_xy in the device coordinate system "));
  set_input_option_map("stress_yz",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= stress and strained_material_names is not set, the stress will be applied to all materials, use this option to set stress component stress_yz in the device coordinate system "));
  set_input_option_map("stress_xz",InputOptions::NonReq_Def("0.00",
      "when input_stress_or_strain= stress and strained_material_names is not set, the stress will be applied to all materials, use this option to set stress component stress_xz in the device coordinate system "));
  set_input_option_map("kp_strain_dir1",InputOptions::NonReq_Def("1",
      "If stress is perpendicular to [x y z] plane then input x here"));
  set_input_option_map("kp_strain_dir2",InputOptions::NonReq_Def("0",
      "If stress is perpendicular to [x y z] plane then input y here"));
  set_input_option_map("kp_strain_dir3",InputOptions::NonReq_Def("0",
      "If stress is perpendicular to [x y z] plane then input z here"));
  set_input_option_map("kp_stress_mag",InputOptions::NonReq_Def("1",
      "When giving stress as input, put stress in GPa here"));

  /*------------- For K.P only (END) -------------*/
  //Aryan add
    set_input_option_map("ref_domain_atom_4matprop",InputOptions::NonReq_Def("0",
          "integer values to choose which atom of the Atomistic subdomain will be used to determine material property, by default 0, i.e. 1st atom of the subdomain"));



}

void Schroedinger::set_description()
{
  description =
      "This simulation solves the tight-binding schrodinger equation. It calculates eigenenergies and eigenstates as well as integrates the results to obtain electron and hole densities (if requested).";
}

void Schroedinger::get_data(const std::string& name, std::map<std::string, double >& result)
{
  if (name=="output_map")
  {
    result = out_param;
  }
}

void Schroedinger::get_data(const std::string& variable, const std::map<unsigned int, double>*& data) const
{
  if (variable == "free_charge")
    data = &( _total_charge );
  else if (variable == "derivative_total_charge_density_over_potential")
    data = &( _total_charge_derivative );
  else
    throw logic_error("[Schroedinger::get_data] wrong variable " +
        variable + "\n");

}

void Schroedinger::prepare_total_charge()
{
  _total_charge.clear();
  _total_charge_derivative.clear();

  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
  const AtomicStructure& atoms   = domain->get_atoms();
  ConstActiveAtomIterator it  = atoms.active_atoms_begin();
  ConstActiveAtomIterator end = atoms.active_atoms_end();

  for ( ; it != end; ++it)
  {
    const unsigned int atom_id = it.id();
    double density = 0;
    double density_derive = 0;

    if (electron_density)
    {
      density -= get_electron_density(atom_id);
      density_derive -= get_derivative_density_over_potential(atom_id,"electron");
    }
    if (hole_density)
    {
      density += get_hole_density(atom_id);
      density_derive += get_derivative_density_over_potential(atom_id,"hole");
    }

    _total_charge.insert(pair<unsigned int,double> (atom_id, density));
    _total_charge_derivative.insert(pair<unsigned int,double> (atom_id, density_derive));
  }

}
void Schroedinger::get_overlap_matrix(const std::vector<NemoMeshPoint>& input_momentum_tuple, const Domain* row_domain, const Domain* column_domain,
    PetscMatrixParallelComplex*& output, DOFmapInterface*& neighbor_DOFmapInterface)
{
  if(row_domain==NULL&&column_domain==NULL)
  {
	  get_data(std::string("overlap_matrix"), output, input_momentum_tuple, false, get_const_simulation_domain() );
  }

  else if(row_domain==column_domain)
    get_data(std::string("overlap_matrix"), output, input_momentum_tuple,get_const_simulation_domain());
  else if(row_domain!=NULL || column_domain!=NULL)
  {
    NEMO_ASSERT(row_domain==get_const_simulation_domain()||column_domain==get_const_simulation_domain(),"Schroedinger(\""+get_name()+"\")::get_overlap_matrix called with 2 foreign domains\n");
    DOFmapInterface* coupling_DOFmap = neighbor_DOFmapInterface;
    if(row_domain==get_const_simulation_domain())
      get_data(std::string("overlap_matrix_coupling"), input_momentum_tuple, column_domain, output, coupling_DOFmap,get_const_simulation_domain());
    else
      get_data(std::string("overlap_matrix_coupling"), input_momentum_tuple, row_domain, output, coupling_DOFmap,get_const_simulation_domain()); //TODO: check whether transpost of output is needed
    delete coupling_DOFmap;
  }
  else
    throw std::runtime_error("Schroedinger(\""+get_name()+"\")::get_overlap_matrix called configuration not implemented yet");
}

std::string Schroedinger::particle_to_string( particule_type type )
{
  if (type == ELECTRON)
    return "electron";
  else if (type == HOLE)
    return "hole";
  return "undefined";
}

void Schroedinger::output_current_modes( particule_type type, double threshold_energy)
{
  string prefix = "";
  if (type == HOLE)
    prefix= "_VB";
  msg << "[Schrodinger]  calculating " << particle_to_string(type) << " current..\n";
  // Find Efs and Efd -> Ids=0 if Efs=Efd
  double chem_pot=options.get_option("chem_pot",0.0);//[eV]
  double chem_pot_drain=options.get_option("chem_pot_drain",0.0);//[eV]
  double T = options.get_option("electron_temperature",NemoPhys::temperature);
  //double DOS_spin_factor = options.get_option("DOS_spin_factor", 1.0);
  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
  const AtomicStructure& atoms   = domain->get_atoms();
  ConstActiveAtomIterator it  = atoms.active_atoms_begin();
  ConstActiveAtomIterator end = atoms.active_atoms_end();
  const AtomStructNode& nd        = it.node();
  const Atom*           atom      = nd.atom;
  const Material*       material  = atom->get_material();
  HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(material));
  int spin_factor = tb_ham->get_spin_degeneracy();
  int  k_degeneracy = options.get_option("k_degeneracy", 1.0);

  // Find (free) directions along which transport happen
  vector<bool> periodic = this->get_simulation_domain()->get_domain_periodicity();
  unsigned int free_dir=periodic[0]+periodic[1]+periodic[2];
  msg<< "[Schrodinger] Found "<<free_dir<<" free directions :";
  if(periodic[0])
    msg<<" X  ";
  if(periodic[1])
    msg<<" Y  ";
  if(periodic[2])
    msg<<" Z  ";
  msg<<"\n";
  // Find the two dir that are free in K in terms of Dir1 and Dir2
  msg<< "[Schrodinger] " << particle_to_string(type) << " threshold energy ="<<threshold_energy<<"\n";

  // create new vector<vector> with valid states and size k x n_bands for CB and VB
  vector< vector <double> > Ek_particle;
  double tempval;
  for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
  {
    // const NemoMeshPoint& ki = kspace->get_point(ii);
    // msg<<"k-points "<<ki[0]<<" "<<ki[1]<<" "<<ki[2]<<"\n";
    Ek_particle.push_back(vector <double> ());
    {
      for (unsigned int jj=0; jj<EnergyV[ii].size(); jj++)
      {
        tempval=EnergyV[ii][jj];
        if(type == ELECTRON && EnergyV[ii][jj] > threshold_energy)
        {
          Ek_particle[ii].push_back(tempval);
        }
        if(type == HOLE && EnergyV[ii][jj] < threshold_energy)
        {
          Ek_particle[ii].push_back(tempval);
        }
      }
    }
    sort(Ek_particle[ii].begin(),Ek_particle[ii].end()); // sort after saving
  }

  // Need to find number of bands and E range for M(E)
  unsigned int bands=9999;
  double Emin=999;
  double Emax=-999;
  vector <double>::iterator itrtr;
  for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
  {
    if(Ek_particle[ii].size()<bands)
      bands=Ek_particle[ii].size();
    itrtr = min_element(Ek_particle[ii].begin(),Ek_particle[ii].end());
    if(Emin > ((double)(*(itrtr))))
      Emin = ((double)(*(itrtr)));
    itrtr = max_element(Ek_particle[ii].begin(),Ek_particle[ii].end());
    if(Emax < ((double)(*(itrtr))))
      Emax = ((double)(*(itrtr)));
  }
  if(bands>0)
  {
    msg<<"[Schrodinger] Extracting "<< bands <<" bands "<<"Emin="<<Emin<<" and Emax="<<Emax<<"\n";
    // 2D transport. Calculate M(E) and current. There will be 2 free directions ->  2 current
    if (free_dir==2)
    {
      string temp1,temp2,temp3,temp4;
      double Klen = sqrt(kspace->get_num_points());// #k1 =#k2 =K
      double dk1=0.0;
      double dk2=0.0;
      const NemoMeshPoint& Klast = kspace->get_point(kspace->get_num_points()-1);
      //msg<<"k-last "<<Klast[0]<<" "<<Klast[1]<<" "<<Klast[2]<<"\n";
      if(!periodic[0]) // Y(=1) and Z(=2) are periodic [Y is index 1 of Ec /Z is index 2 of EC]
      {
        dk1=Klast[1]/(Klen-1);
        msg<<"[Schrodinger] dky(1/nm)="<<dk1<<"\n";
        dk2=Klast[2]/(Klen-1);
        msg<<"[Schrodinger] dkz(1/nm)="<<dk2<<"\n";
      }
      if(!periodic[1]) // X(=1) and Z(=2) are periodic
      {
        dk1=Klast[0]/(Klen-1);
        msg<<"[Schrodinger] dkx(1/nm)="<<dk1<<"\n";
        dk2=Klast[2]/(Klen-1);
        msg<<"[Schrodinger] dkz(1/nm)="<<dk2<<"\n";
      }
      if(!periodic[2]) // X(=1) and Y(=2) are periodic
      {
        dk1=Klast[0]/(Klen-1);
        msg<<"[Schrodinger] dkx(1/nm)="<<dk1<<"\n";
        dk2=Klast[1]/(Klen-1);
        msg<<"[Schrodinger] dky(1/nm)="<<dk2<<"\n";
      }

      // Arrange Ek (k1 , k2 , E)
      vector <double> tmp2(bands,-99);
      vector< vector <double> > tmp1(Klen,tmp2);
      vector< vector< vector <double> > > energy_band(Klen,tmp1);
      for (unsigned int ii=0; ii<Klen; ii++)
      {
        for (unsigned int jj=0; jj<Klen; jj++)
        {
          for(unsigned int kk=0; kk<bands; kk++)
          {
            energy_band[ii][jj][kk]=Ek_particle[Klen*ii+jj][kk];
          }
        }
      }
      //initialize Egrid and ME
      int Enum=(Emax-Emin+0.05)/0.002;
      vector <double> Egrid(Enum,0);
      vector <double> ME1(Enum,0);
      vector <double> ME2(Enum,0);
      vector <double> inflexpt; //need to store inflexion points in the bands
      for (int ii=0; ii<Enum; ii++)
      {
        if (type == ELECTRON)
          Egrid[ii]=Emin-0.05 + ii*0.002;
        if (type == HOLE)
          Egrid[ii]=Emin-0.05 - ii*0.002;
      }
      for (unsigned int mm=0; mm<Klen; mm++)
      {
        // loop over all the transverse k
        //now for each band extract modes along dir 1
        for (unsigned int nn=0; nn<bands; nn++ )
        {
          //travel along k
          inflexpt.resize(0);
          inflexpt.push_back(energy_band[0][mm][nn]);
          for (unsigned int kk=1; kk<Klen-1; kk++)
          {
            //inflextion point if slope of band changes
            if(NemoMath::sign(energy_band[kk][mm][nn]-energy_band[kk-1][mm][nn]) != NemoMath::sign(energy_band[kk+1][mm][nn]-energy_band[kk][mm][nn]))
            {
              inflexpt.push_back(energy_band[kk][mm][nn]);
            }
          }
          inflexpt.push_back(energy_band[Klen-1][mm][nn]);
          for (unsigned int pp=0; pp<inflexpt.size()-1; pp++)
          {
            // for each pair of inflexion point update ME by 1
            for (int ee=0; ee<Enum; ee++)
            {
              if (inflexpt[pp+1]>inflexpt[pp])
              {
                if ( Egrid[ee] > inflexpt[pp] && Egrid[ee] < inflexpt[pp+1])
                {
                  ME1[ee]=ME1[ee]+1;
                }
              }

              if (inflexpt[pp+1]<inflexpt[pp])
              {
                if ( Egrid[ee] > inflexpt[pp+1] && Egrid[ee] < inflexpt[pp])
                {
                  ME1[ee]=ME1[ee]+1;
                }
              }
            }
          }
        }
      }
      // Dump modes for DIR 1
      if(!periodic[0]) // Y(=1) and Z(=2) are periodic [Y is index 1 of Ec /Z is index 2 of energy_band]
        temp1 = get_name()+ get_output_suffix() + prefix + "_DOM_Y.dat";
      if(!periodic[1]) // X(=1) and Z(=2) are periodic
        temp1 = get_name()+ get_output_suffix() + prefix + "_DOM_X.dat";
      if(!periodic[2]) // X(=1) and Y(=2) are periodic
        temp1 = get_name()+ get_output_suffix() + prefix + "_DOM_X.dat";
      msg << "[Schrodinger] saving Mode vs Energy to " << temp1 << "...\n";
      ofstream file1(temp1.c_str(), std::ios_base::out | std::ios_base::trunc);
      file1.precision(output_precision);
      file1 << "% Energy(eV) M(E) (#/m)  \n";
      for (int ii=0; ii<Enum; ii++)
      {
        // Modes are needed for 1/2 of BZ hence k_deg/2
        file1 << Egrid[ii] << " " << spin_factor*(1/(2*3.1415))*k_degeneracy*0.5*1e9*dk2* ME1[ii] <<"\t"; // (k_deg/2)*(dk in m)*M(E)
        file1 << "\n";
      }
      file1.close();

      /* DIRECTION 2 */
      for (unsigned int mm=0; mm<Klen; mm++)
      {
        // loop over all the transverse k
        //now for each band extract modes along dir 2
        for (unsigned int nn=0; nn<bands; nn++ )
        {
          //travel along k
          inflexpt.resize(0);
          inflexpt.push_back(energy_band[mm][0][nn]);
          for (unsigned int kk=1; kk<Klen-1; kk++)
          {
            //inflextion point if slope of band changes
            if(NemoMath::sign(energy_band[mm][kk][nn]-energy_band[mm][kk-1][nn]) != NemoMath::sign(energy_band[mm][kk+1][nn]-energy_band[mm][kk][nn]))
            {
              inflexpt.push_back(energy_band[mm][kk][nn]);
            }
          }
          inflexpt.push_back(energy_band[mm][Klen-1][nn]);
          for (unsigned int pp=0; pp<inflexpt.size()-1; pp++)
          {
            // for each pair of inflexion point update ME by 1
            for (int ee=0; ee<Enum; ee++)
            {
              if (inflexpt[pp+1]>inflexpt[pp])
              {
                if ( Egrid[ee] > inflexpt[pp] && Egrid[ee] < inflexpt[pp+1])
                {
                  ME2[ee]=ME2[ee]+1;
                }
              }

              if (inflexpt[pp+1]<inflexpt[pp])
              {
                if ( Egrid[ee] > inflexpt[pp+1] && Egrid[ee] < inflexpt[pp])
                {
                  ME2[ee]=ME2[ee]+1;
                }
              }
            }
          }
        }
      }

      // Dump modes for DIR 2
      if(!periodic[0]) // Y(=1) and Z(=2) are periodic [Y is index 1 of Ec /Z is index 2 of energy_band]
        temp2 = get_name()+ get_output_suffix() + prefix + "_DOM_Z.dat";
      if(!periodic[1])
        temp2 = get_name()+ get_output_suffix() + prefix + "_DOM_Z.dat";
      if(!periodic[2])
        temp2 = get_name()+ get_output_suffix() + prefix + "_DOM_Y.dat";
      msg << " [Schrodinger] saving Mode vs Energy to " << temp2 << "...\n";
      ofstream file2(temp2.c_str(), std::ios_base::out | std::ios_base::trunc);
      file2.precision(output_precision);
      file2 << "% Energy(eV) M(E) (#/m)  \n";
      for (int ii=0; ii<Enum; ii++)
      {
        file2 << Egrid[ii] << " " << spin_factor*(1/(2*3.1415))*k_degeneracy*0.5*1e9*dk1* ME2[ii] <<"\t";
        file2 << "\n";
      }
      file2.close();

      // Now compute the 2 currents

      // now calculate current
      double fin_current1=0;
      double fin_current2=0;
      for(int ii=0; ii<Enum-1; ii++)
      {
        double dE=abs(Egrid[ii+1]-Egrid[ii]);
        double M1 =0.5*abs(ME1[ii+1]+ME1[ii]);
        double M2 =0.5*abs(ME2[ii+1]+ME2[ii]);
        double E=0.5*(Egrid[ii+1]+Egrid[ii]);
        double f1=NemoMath::fermi_distribution(chem_pot,0.0256*(T/298),E);
        double f2=NemoMath::fermi_distribution(chem_pot_drain,0.0256*(T/298),E);
        if (type == HOLE)
        {
          f1=1-f1;
          f2=1-f2;
        }
        fin_current1 +=spin_factor * (1/(2*3.1415)) * k_degeneracy * 0.5 * 1e9* dk2 * M1 * dE * abs(f1-f2);
        fin_current2 +=spin_factor * (1/(2*3.1415)) * k_degeneracy * 0.5 * 1e9* dk1 * M2 * dE * abs(f1-f2);
      }
      fin_current1 = 3.87e-5 * fin_current1;
      fin_current2 = 3.87e-5 * fin_current2;

      //now calculate carrier density
      double Qinv=0;
      for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
      {
        for (unsigned int jj=0; jj<Ek_particle[ii].size(); jj++)
        {
          double f1=NemoMath::fermi_distribution(chem_pot,0.0256*(T/298),Ek_particle[ii][jj]);
          double f2=NemoMath::fermi_distribution(chem_pot_drain,0.0256*(T/298),Ek_particle[ii][jj]);
          if (type == HOLE){
            f1=1-f1;
            f2=1-f2;
          }
          Qinv+=spin_factor * (1/(4*3.1415*3.1415)) * dk1*1e9 * dk2*1e9 * k_degeneracy*0.5 * abs(f1+f2);
        }
      }

      if(!periodic[0]) // Y(=1) and Z(=2) are periodic [Y is index 1 of Ec /Z is index 2 of energy_band]
                                temp3 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_Y.dat";
      if(!periodic[1])
        temp3 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_X.dat";
      if(!periodic[2])
        temp3 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_X.dat";
      msg << "[Schrodinger] saving current calculation to " << temp3 << "...\n";
      ofstream file3(temp3.c_str(), std::ios_base::out | std::ios_base::trunc);
      file3.precision(output_precision);
      file3 << "% 2D carrier density (#/m2)  | 1D Ballistic TOB "<< particle_to_string(type) << " current (A/m) \n";
      file3 << Qinv <<"\t"<< fin_current1;
      file3.close();

      if(!periodic[0])
        temp4 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_Z.dat";
      if(!periodic[1])
        temp4 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_Z.dat";
      if(!periodic[2])
        temp4 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_Y.dat";
      msg << "[Schrodinger] saving current calculation to " << temp4 << "...\n";
      ofstream file4(temp4.c_str(), std::ios_base::out | std::ios_base::trunc);
      file4.precision(output_precision);
      file4 << "% 2D carrier density (#/m2)  | 1D Ballistic TOB " << particle_to_string(type) << " current (A/m) \n";
      file4 << Qinv <<"\t"<<fin_current2;
      file4.close();

    }//end of 2D loop

    // 1D transport. Calculated M(E) and current
    if (free_dir==1)
    {
      string temp,temp1;
      double dk=0.0;
      double Klen = kspace->get_num_points();// #k1 =K
      const NemoMeshPoint& Klast = kspace->get_point(kspace->get_num_points()-1);
      if(periodic[0])
      {
        dk=Klast[0]/(Klen-1);
        msg<<"[Schrodinger] dkx(1/nm)="<<dk<<"\n";
      }
      if(periodic[1])
      {
        dk=Klast[1]/(Klen-1);
        msg<<"[Schrodinger] dky(1/nm)="<<dk<<"\n";
      }
      if(periodic[2])
      {
        dk=Klast[2]/(Klen-1);
        msg<<"[Schrodinger] dkx(1/nm)="<<dk<<"\n";
      }

      // Arrange Ek (k1 , k2 , E)
      //initialize Egrid and ME
      int Enum=(Emax-Emin+0.05)/0.002;
      vector <double> Egrid(Enum,0);
      vector <double> ME(Enum,0);
      vector <double> inflexpt; //need to store inflexion points in the bands

      for (int ii=0; ii<Enum; ii++)
      {
        if (type == ELECTRON)
        {
          Egrid[ii]=Emin-0.05 + ii*0.002;
        }
        if (type == HOLE)
        {
          Egrid[ii]=Emax+0.05 - ii*0.002;
        }
      }
      //now for each band extract modes
      for (unsigned int nn=0; nn<bands; nn++ )
      {
        //travel along k
        inflexpt.resize(0);
        inflexpt.push_back(Ek_particle[0][nn]);
        for (unsigned int kk=1; kk<Klen-1; kk++)
        {
          //inflextion point if slope of band changes
          if(NemoMath::sign(Ek_particle[kk][nn]-Ek_particle[kk-1][nn]) != NemoMath::sign(Ek_particle[kk+1][nn]-Ek_particle[kk][nn]))
          {
            inflexpt.push_back(Ek_particle[kk][nn]);
          }
        }
        inflexpt.push_back(Ek_particle[Klen-1][nn]);
        for (unsigned int pp=0; pp<inflexpt.size()-1; pp++)
        {
          // for each pair of inflexion point update ME by 1
          for (int ee=0; ee<Enum; ee++)
          {
            if (inflexpt[pp+1]>inflexpt[pp])
            {
              if ( Egrid[ee] > inflexpt[pp] && Egrid[ee] < inflexpt[pp+1])
              {
                ME[ee]=ME[ee]+1;
              }
            }
            if (inflexpt[pp+1]<inflexpt[pp])
            {
              if ( Egrid[ee] > inflexpt[pp+1] && Egrid[ee] < inflexpt[pp])
              {
                ME[ee]=ME[ee]+1;
              }
            }
          }
        }
      }

      if(periodic[0]) // X
        temp = get_name()+ get_output_suffix() + prefix + "_DOM_X.dat";
      if(periodic[1]) // Y
        temp = get_name()+ get_output_suffix() + prefix + "_DOM_Y.dat";
      if(periodic[2]) // Z
        temp = get_name()+ get_output_suffix() + prefix + "_DOM_Z.dat";
      msg << " [Schrodinger] saving Mode vs Energy to " << temp << "...\n";
      ofstream file(temp.c_str(), std::ios_base::out | std::ios_base::trunc);
      file.precision(output_precision);
      file << "% Energy(eV) M(E) (unitless) [should relate to E-k data] \n";
      for (int ii=0; ii<Enum; ii++)
      {
        file << Egrid[ii] << " " << spin_factor* ME[ii] <<"\t";
        file << "\n";
      }
      file.close();

      // now calculate current
      double fin_current=0;
      for(int ii=0; ii<Enum-1; ii++)
      {
        double dE=abs(Egrid[ii+1]-Egrid[ii]);
        double M =0.5*abs(ME[ii+1]+ME[ii]);
        double E=0.5*(Egrid[ii+1]+Egrid[ii]);
        double f1=NemoMath::fermi_distribution(chem_pot,0.0256*(T/298),E);
        double f2=NemoMath::fermi_distribution(chem_pot_drain,0.0256*(T/298),E);
        if (type == HOLE)
        {
          f1=1-f1;
          f2=1-f2;
        }
        fin_current += M * dE * abs(f1-f2);
      }
      fin_current = spin_factor * 3.8829e-5 * fin_current;
      //now calculate carrier density
      double Qinv=0;
      for (unsigned int ii=0; ii<kspace->get_num_points(); ii++)
      {
        for (unsigned int jj=0; jj<Ek_particle[ii].size(); jj++)
        {
          double f1=NemoMath::fermi_distribution(chem_pot,0.0256*(T/298),Ek_particle[ii][jj]);
          double f2=NemoMath::fermi_distribution(chem_pot_drain,0.0256*(T/298),Ek_particle[ii][jj]);
          if (type == HOLE){
            f1=1-f1;
            f2=1-f2;
          }
          Qinv+=spin_factor * (1/(2*3.1415)) *  dk*1e9 * k_degeneracy*0.5 * abs(f1+f2);
        }
      }

      if(periodic[0]) // X
          temp1 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_X.dat";
      if(periodic[1]) // Y
          temp1 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_Y.dat";
      if(periodic[2]) // Z
        temp1 = get_name()+ get_output_suffix() + "_" + particle_to_string(type) + "_current_Z.dat";
      msg << "[Schrodinger] saving current calculation to " << temp1 << "...\n";
      ofstream file1(temp1.c_str(), std::ios_base::out | std::ios_base::trunc);
      file1.precision(output_precision);
      file1 << "%1D Carrier density (#/m) |  1D Ballistic TOB " << particle_to_string(type) << " current (A) \n";
      file1 << Qinv<<"\t"<<fin_current;
      file1.close();
    }
    else
      msg<<"[Schrodinger] NOT computing current : no valid band found \n";
  }
}


void Schroedinger::output_relation_xyz_H_index()
{
  const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
  DOFmapInterface& dof_map = get_dof_map();

  const AtomicStructure& atoms = domain->get_atoms();
  ConstActiveAtomIterator it = atoms.active_atoms_begin();
  ConstActiveAtomIterator end = atoms.active_atoms_end();

  std::string filename = get_name() + "_relation_xyz_H_index.dat";

  ofstream output_file(filename.c_str());
  output_file << std::left << std::setw(10) << "% x[nm]"
    << std::left << std::setw(10) << "y[nm]"
    << std::left << std::setw(10) << "z[nm]"
    << std::left << std::setw(10) << "min_index"
    << std::left << std::setw(10) << "max_index" << "\n";

  for (; it != end; ++it)
  {
    const unsigned int    atom_id = it.id();
    const AtomStructNode& nd = it.node();

    const std::map<short, unsigned int>* atom_dof_map = dof_map.get_atom_dof_map(atom_id);
    std::map<short, unsigned int>::const_iterator temp_it = atom_dof_map->begin();
    unsigned int min_index = UINT_MAX;
    unsigned int max_index = 1;
    for (; temp_it != atom_dof_map->end(); temp_it++)
    {
      unsigned int temp_index = temp_it->second;
      if (temp_index > max_index)
        max_index = temp_index;
      if (temp_index < min_index)
        min_index = temp_index;
    }

    output_file << std::left << std::setw(10) << nd.position[0]
      << std::left << std::setw(10) << nd.position[1]
      << std::left << std::setw(10) << nd.position[2]
      << std::left << std::setw(10) << min_index
      << std::left << std::setw(10) << max_index << "\n";
  }

  output_file.close();

}
