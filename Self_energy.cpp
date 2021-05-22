//$Id: Self_energy.cpp $
/*  Purpose:    Self_energy class = base class for every self-energy - calculation
                irrespective of system's dimensionality or its representation
    Note:       child class of the propagation class
*/

#include "Simulation.h"
#include "Propagator.h"
#include "Propagation.h"
#include "Greensolver.h"
#include "Self_energy.h"
#include "Matrix.h"
#include "PetscMatrixParallel.h"
#include "PetscMatrixParallelComplex.h"
#include "PetscVectorNemo.h"
#include "LinearSolverPetscComplex.h"
#include "LinearSolverPetsc.h"
#include <map>
#include <complex>
#include <stdexcept>
#include "Nemo.h"
#include "DOFmap.h"
#include "NemoUtils.h"
#include "Domain.h"
#include "Atom.h"
#include "AtomicStructure.h"
#include "AtomisticDomain.h"
#include "AtomStructNode.h"
#include "ActiveAtomIterator.h"
#include "HamiltonConstructor.h"
#include "Material.h"
#include "EigensolverSlepc.h"
#include "NemoMath.h"
#include "NemoPhys.h"
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include "NRLTBHamiltonConstructor.h"
#include "TBHamiltonConstructor.h"
#include "EMHamiltonConstructor.h"
#include "KPHamiltonConstructor.h"
#include "ExtendedHuckel_HamiltonConstructor.h"
#include "ExtHuckel_with_ETB_passivation.h"
#include "QuantumNumberUtils.h"
#include "EOMMatrixInterface.h"
#include "PropagationUtilities.h"
#include "Schroedinger.h"

#ifndef NO_MKL
#include "mkl.h"
#endif

#include <iostream>
#include <fstream>
using NemoUtils::MsgLevel;




void Self_energy::do_init(void)
{
  msg.set_level(MsgLevel(4));
  //initialize the base class
  base_init();
  known_Propagators.clear();
  std::map<std::string, const Propagator*>::iterator prop_it=Propagators.begin();
  for(;prop_it!=Propagators.end();++prop_it)
    if(prop_it->second!=NULL)
      known_Propagators.insert(prop_it->second);
  if(writeable_Propagator!=NULL)
    known_Propagators.insert(writeable_Propagator);

  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_init ";
  NemoUtils::tic(tic_toc_prefix);
  //-----------------------------
  //first, check that all propagators to solve are indeed self-energies
  //-----------------------------
  //std::map<std::string,Propagator*>::const_iterator it = writeable_Propagators.begin();
  //for (; it!=writeable_Propagators.end(); ++it)
  if(writeable_Propagator!=NULL)
  {
    NEMO_ASSERT(Propagator_type_map.find(Propagator_types.find(name_of_writeable_Propagator)->second)->second.find(std::string("self"))!=std::string::npos,
                std::string("Self_energy(\""+this->get_name()+"\")::do_init() NEGF-object \""+name_of_writeable_Propagator+"\" is not a self-energy\n"));

    if(name_of_writeable_Propagator.find(std::string("constant_eta"))!=std::string::npos)
      do_init_constant_eta();
    else if(name_of_writeable_Propagator.find(std::string("contact"))!=std::string::npos)
    {
      if (name_of_writeable_Propagator.find(std::string("retarded"))!=std::string::npos)
        do_init_retarded_contact();
      else if (name_of_writeable_Propagator.find(std::string("lesser"))!=std::string::npos)
        do_init_lesser_contact();
      else
        throw std::invalid_argument("Self_energy(\""+get_name()+"\")::do_init unknown contact self-energy type: \""+name_of_writeable_Propagator+"\"\n");
    }
  }
  msg<<"Self_energy(\""+this->get_name()+"\")::do_init() done."<<std::endl;
  NemoUtils::toc(tic_toc_prefix);
}
void Self_energy::do_reinit()
{
  std::string prefix = "Self_energy(\""+tic_toc_name+"\")::do_reinit ";
  chemical_potential_map.clear();
  bp_chemical_potential_resolved=false;
  bp_chemical_potential_initialized=false;
  Propagation::do_reinit();
}


void Self_energy::do_init_constant_eta(void)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_init_constant_eta ";
  NemoUtils::tic(tic_toc_prefix);
  //NEMO_ASSERT(writeable_Propagators.end()==++writeable_Propagators.begin(),
  //            "Self_energy(\""+get_name()+"\")::do_init_constant_eta found more than one writeable Propagator\n");
  writeable_Propagator->set_storage_type("diagonal");
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_init_lesser_contact(void)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_init_lesser_contact ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix=tic_toc_prefix;
  //if standard equilibrium model...
  if(options.get_option("equilibrium_model",true))
  {
    //check the presence of all required input (temperature, chemical potential)
    //NEMO_ASSERT(options.check_option("temperature"),prefix+"input missing: temperature\n");
    //NEMO_ASSERT(options.check_option("chemical_potential"),prefix+"input missing: chemical_potential\n");

    std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
    NEMO_ASSERT(writeable_Propagator!=NULL,prefix+"wrong number of writeable Propagators (only exactly one allowed)\n");
    bool correct_inputdeck=false;
    unsigned int number_of_retarded=0, number_of_lesser=0,number_of_advanced=0;
    for(; c_it!=Propagators.end(); c_it++)
    {
      if(get_Propagator_type(c_it->first)==NemoPhys::Boson_retarded_self||get_Propagator_type(c_it->first)==NemoPhys::Fermion_retarded_self)
        ++number_of_retarded;
      else if (get_Propagator_type(c_it->first)==NemoPhys::Boson_lesser_self||get_Propagator_type(c_it->first)==NemoPhys::Fermion_lesser_self)
        ++number_of_lesser;
      else if (get_Propagator_type(c_it->first)==NemoPhys::Boson_advanced_self||get_Propagator_type(c_it->first)==NemoPhys::Fermion_advanced_self)
        ++number_of_advanced;
    }
    if(Propagators.size()==2)
    {
      if(number_of_lesser==1&&number_of_retarded==1)
        correct_inputdeck=true;
    }
    else if(Propagators.size()==3)
    {
      if(number_of_advanced==1&&number_of_lesser==1&&number_of_retarded==1)
        correct_inputdeck=true;
    }
    else
      throw std::invalid_argument(prefix+"wrong number of Propagators; required: 1 retarded, 1 lesser; optional: 1 advanced in addition\n");

    NEMO_ASSERT(correct_inputdeck,prefix+"wrong types of Propagators given; required: 1 retarded, 1 lesser; optional: 1 advanced in addition\n");
  }
  else
  {
    //this "nonequilibrium model" assumes that a surface lesser Green's function is available (as for RGF-version2 for example)
    //check that there is a lesser Green's function available
    std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
    NEMO_ASSERT(writeable_Propagator!=NULL,prefix+"wrong number of writeable Propagators (only exactly one allowed)\n");
    NEMO_ASSERT(Propagators.size()==2,prefix+"wrong number of total Propagators (only two allowed)\n");
    for(; c_it!=Propagators.end(); c_it++)
    {
      if(c_it->first!=name_of_writeable_Propagator)
        NEMO_ASSERT(get_Propagator_type(c_it->first)==NemoPhys::Boson_lesser_Green||get_Propagator_type(c_it->first)==NemoPhys::Fermion_lesser_Green,
                    prefix+"readable propagator has to be a lesser Green\'s function\n");
    }
    //throw std::invalid_argument("Self_energy(\""+get_name()+"\")::do_init_lesser_contact: only equilibrium leads implemented so far\n");
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_init_retarded_contact(void)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_init_retarded_contact ";
  NemoUtils::tic(tic_toc_prefix);
  msg<<"Self_energy(\""+this->get_name()+"\")::do_init_retarded_contact()"<<std::endl;


  // -----------------------------------
  // check that there is exactly one Fermion_retarded_Green Propagator
  // -----------------------------------
  std::map<std::string, const Propagator*>::const_iterator c_prop_it=Propagators.begin();
  bool found=false;
  for(; c_prop_it!=Propagators.end(); ++c_prop_it)
  {
    std::map<std::string, NemoPhys::Propagator_type>::const_iterator c_prop_type_it=Propagator_types.find(c_prop_it->first);
    NEMO_ASSERT(c_prop_type_it!=Propagator_types.end(),
                std::string("Self_energy("+this->get_name()+")::do_init_retarded_contact: have not found \""+c_prop_it->first+"\" in the Propagator map\n"));
    if(c_prop_type_it->second==NemoPhys::Fermion_retarded_Green)
    {
      NEMO_ASSERT(!found,std::string("Self_energy("+this->get_name()+")::do_init_retarded_contact: more than one Fermion_retarded_Green defined\n"));
      found=true;
      ////check that the Fermion_retarded_Green constructor is defined on the same domain as the HamiltonConstructor
      //Simulation * Green_constructor = pointer_to_Propagator_Constructors->find(c_prop_type_it->first)->second;
      //NEMO_ASSERT(Hamilton_Constructor->get_const_simulation_domain()==Green_constructor->get_const_simulation_domain(),"Self_energy("+this->get_name()+")::do_init_contact: mismatch of simulation domains\n");
    }
  }
  //NEMO_ASSERT(std::string("Self_energy("+this->get_name()+")::do_init_contact: no Fermion_retarded_Green has been declared in the input deck!\n"));


  bool stationary=options.get_option("stationary_Propagation",true);
  NEMO_ASSERT(stationary,std::string("Self_energy(\""+get_name()
                                     +"\")::do_init_retarded_contact: stationary is set to false; however, only stationary Propgation is implemented so far\n"));

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
        throw std::invalid_argument("Self_energy(\""+get_name()+"\")::do_init_retarded_contact: found more than one energy: "+energy_name+" & "+c_it->first
                                    +"\n");
    }
  }
  NEMO_ASSERT(energy_name!=std::string(""),"Self_energy(\""+get_name()+"\")::do_init_retarded_contact: found no energy\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::get_Selfenergy(const std::vector<NemoMeshPoint>& momentum,PetscMatrixParallelComplex*& result,
                                const DOFmapInterface* row_dofmap, const DOFmapInterface* column_dofmap, const NemoPhys::Propagator_type&)
{
  get_data(name_of_writeable_Propagator,&momentum,result,row_dofmap,column_dofmap);
}

void Self_energy::get_data(const std::string& variable, const std::set<std::pair<int,int> >*& data)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::get_data ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_data ";
  if(variable=="sparsity_pattern")
  {
    if(set_of_row_col_indices.size()>0)
      data=&set_of_row_col_indices;
    else
      data=NULL;
  }
  else
    throw std::runtime_error(prefix+"called with unknown data-string\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::set_self_energies_to_zero(void)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::set_self_energies_to_zero ";
  NemoUtils::tic(tic_toc_prefix);
  //std::map<std::string,Propagator*>::iterator it=writeable_Propagators.begin();
  //for(; it!=writeable_Propagators.end(); ++it)
  if(writeable_Propagator!=NULL)
  {
    if(name_of_writeable_Propagator.find("combined")==std::string::npos) //do not set to zero if the self-energy is a combined propagator...
    {
      writeable_Propagator->set_storage_type("empty");
      Propagator::PropagatorMap::iterator it2=writeable_Propagator->propagator_map.begin();
      const DOFmapInterface* defining_DOFmap=&(Hamilton_Constructor->get_dof_map());
      for(; it2!=writeable_Propagator->propagator_map.end(); ++it2)
      {
        allocate_propagator_matrices(defining_DOFmap,&(it2->first));
        it2->second->set_to_zero();
        it2->second->assemble();
        //it->second->ready_momentum_Propagator_map[it2->first]=true;
      }
      std::map<std::string, bool>::iterator it3=ready_Propagator_map.find(name_of_writeable_Propagator);
      NEMO_ASSERT(it3!=ready_Propagator_map.end(),"Self_energy(\""+get_name()+"\")::set_self_energies_to_zero have not found \""+name_of_writeable_Propagator
                  +"\" in ready_map\n");
      it3->second=true;
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::initialize_Propagation(void)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::initialize_Propagation ";
  NemoUtils::tic(tic_toc_prefix);
  if(!Propagation_is_initialized)
  {
    msg<<"\n\ninitializing: "<<get_name()<<"\n\n";
    if(!one_energy_only)
    {
      PropagationUtilities::set_parallel_environment(this, Mesh_tree_names, 
        Mesh_tree_downtop,Mesh_tree_topdown,
        Momentum_meshes,momentum_mesh_types,
        Mesh_Constructors,Parallelizer,this->get_simulation_communicator());
      //set_parallel_environment();
      fill_all_momenta();
    }
    initialize_Propagators();
    //set_self_energies_to_zero();
  }
  Propagation_is_initialized=true;
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::convert_set_to_vector(std::set<communication_pair>*& set_ptr,
                                        std::vector<communication_pair>*& vector_ptr)
{
  std::set<communication_pair>::iterator I_set_it = set_ptr->begin();

  vector_ptr->resize(set_ptr->size());
  for (unsigned int i = 0; i < set_ptr->size(); i++)
  {
    (*vector_ptr)[i] = (*I_set_it);
    I_set_it++;
  }
}

void Self_energy::build_comm_table(std::vector<std::set<communication_pair> >* comm_table_order,
                                   std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > >* rank_local_momentum_map)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::build_comm_table ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::build_comm_table ";

  // determine the communication table due to the scattering type
  std::set<communication_pair> scattering_communication_table;
  std::set<communication_pair>* temp_pointer = &scattering_communication_table;
  std::set<communication_pair> scattering_communication_table2;
  std::set<communication_pair>* temp_pointer_absorption = &scattering_communication_table2;

  std::string scattering_type = options.get_option("scattering_type",std::string(""));

  if (scattering_type == "deformation_potential" || scattering_type== "roughness")
  {
    double energy_interval=0.0;
    if(scattering_type=="deformation_potential")
      energy_interval=options.get_option("scattering_energy_interval", 0.0);

    get_communication_continual_scattering(energy_interval,
                                           options.get_option("scattering_momentum_interval", 1e20), temp_pointer);
    //check_and_correct_coupling_consistency(temp_pointer);
  }
  else if (scattering_type == "optical_deformation_potential" || scattering_type=="polar_optical_Froehlich")
  {
    NEMO_ASSERT(options.check_option("optical_phonon_energy"),prefix+"please define \"optical_phonon_energy\"\n");
    double phonon_energy=options.get_option("optical_phonon_energy", 0.0);
    if(!options.get_option("readin_coupling_table",bool(false)))
      get_communication_discrete_scattering(phonon_energy,options.get_option("scattering_momentum_interval", 1e20),
                                            temp_pointer, temp_pointer_absorption);
    else
      get_communication_readin_discrete_scattering(options.get_option("scattering_momentum_interval",1e20),temp_pointer, temp_pointer_absorption);
  }
  else
    throw std::invalid_argument(prefix + "scattering type is not implemented\n");


  //Build the communication table according to the inputdeck options
  std::vector<communication_pair> scattering_communication_table_vector_sort;
  std::vector<communication_pair>* temp_pointer_vector_sort =
    &scattering_communication_table_vector_sort;

  std::string level = options.get_option("build_communication_table",std::string("fully_sequential"));

  if (level == std::string("fully_sequential"))
  {
    //order the communication table
    get_communication_order(scattering_communication_table, comm_table_order,rank_local_momentum_map);
    if(scattering_communication_table2.size()>0)
    {
      std::vector<std::set<communication_pair> > comm_table_order2;
      std::vector<std::set<communication_pair> >* pointer_to_comm_table_order2=&comm_table_order2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > > rank_local_momentum_map2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > >* pointer_to_rank_local_momentum_map2=&rank_local_momentum_map2;
      get_communication_order(scattering_communication_table2, pointer_to_comm_table_order2,pointer_to_rank_local_momentum_map2);
      //append the second communication table with the first and the second local...map with the first
      std::vector<std::set<communication_pair> > temp_comm_table_order(comm_table_order->size()+comm_table_order2.size(),
          comm_table_order2[0]);
      for(unsigned int i=0; i<comm_table_order->size(); i++)
        temp_comm_table_order[i]=(*comm_table_order)[i];
      for(unsigned int i=comm_table_order->size(); i<temp_comm_table_order.size(); i++)
        temp_comm_table_order[i]=comm_table_order2[i-comm_table_order->size()];
      comm_table_order->clear();
      *comm_table_order=temp_comm_table_order;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > > temp_rank_local_momentum_map(rank_local_momentum_map->size()+
          rank_local_momentum_map2.size(),rank_local_momentum_map2[0]);
      for(unsigned int i=0; i<rank_local_momentum_map->size(); i++)
        temp_rank_local_momentum_map[i]=(*rank_local_momentum_map)[i];
      for(unsigned int i=rank_local_momentum_map->size(); i<temp_rank_local_momentum_map.size(); i++)
        temp_rank_local_momentum_map[i]=rank_local_momentum_map2[i-rank_local_momentum_map->size()];
      rank_local_momentum_map->clear();
      *rank_local_momentum_map=temp_rank_local_momentum_map;
    }
  }
  else if (level == std::string("improved_fully_sequential"))
  {
    convert_set_to_vector(temp_pointer, temp_pointer_vector_sort);
    //order the communication table
    get_communication_order_parallel(temp_pointer_vector_sort, comm_table_order,
                                     rank_local_momentum_map, false);
    if(scattering_communication_table2.size()>0)
    {
      std::vector<std::set<communication_pair> > comm_table_order2;
      std::vector<std::set<communication_pair> >* pointer_to_comm_table_order2=&comm_table_order2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > > rank_local_momentum_map2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > >* pointer_to_rank_local_momentum_map2=&rank_local_momentum_map2;
      std::vector<communication_pair> scattering_communication_table_vector_sort2;
      std::vector<communication_pair>* temp_pointer_vector_sort2 =
        &scattering_communication_table_vector_sort2;
      convert_set_to_vector(temp_pointer_absorption, temp_pointer_vector_sort2);

      //order the communication table
      get_communication_order_parallel(temp_pointer_vector_sort2, pointer_to_comm_table_order2,
                                       pointer_to_rank_local_momentum_map2, false);

      comm_table_order->insert(comm_table_order->end(),pointer_to_comm_table_order2->begin(),
                               pointer_to_comm_table_order2->end());
      rank_local_momentum_map->insert(rank_local_momentum_map->end(),
                                      pointer_to_rank_local_momentum_map2->begin(),
                                      pointer_to_rank_local_momentum_map2->end());
    }


  }
  else if (level == std::string("fully_parallel"))
  {

    convert_set_to_vector(temp_pointer, temp_pointer_vector_sort);
    //order the communication table
    get_communication_order_parallel(temp_pointer_vector_sort, comm_table_order,
                                     rank_local_momentum_map, true);
    if(scattering_communication_table2.size()>0)
    {
      std::vector<std::set<communication_pair> > comm_table_order2;
      std::vector<std::set<communication_pair> >* pointer_to_comm_table_order2=&comm_table_order2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > > rank_local_momentum_map2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > >* pointer_to_rank_local_momentum_map2=&rank_local_momentum_map2;
      std::vector<communication_pair> scattering_communication_table_vector_sort2;
      std::vector<communication_pair>* temp_pointer_vector_sort2 =
        &scattering_communication_table_vector_sort2;
      convert_set_to_vector(temp_pointer_absorption, temp_pointer_vector_sort2);

      //order the communication table
      get_communication_order_parallel(temp_pointer_vector_sort2, pointer_to_comm_table_order2,
                                       pointer_to_rank_local_momentum_map2, false);

      comm_table_order->insert(comm_table_order->end(),pointer_to_comm_table_order2->begin(),
                               pointer_to_comm_table_order2->end());
      rank_local_momentum_map->insert(rank_local_momentum_map->end(),
                                      pointer_to_rank_local_momentum_map2->begin(),
                                      pointer_to_rank_local_momentum_map2->end());
    }
  }
  else if (level == std::string("fully_parallel_minimum_first"))
  {
    bool maximum_first = false;
    communication_rank_sort(temp_pointer, temp_pointer_vector_sort,
                            maximum_first);
    //order the communication table
    get_communication_order_parallel(temp_pointer_vector_sort, comm_table_order,
                                     rank_local_momentum_map, true);
    if(scattering_communication_table2.size()>0)
    {
      std::vector<std::set<communication_pair> > comm_table_order2;
      std::vector<std::set<communication_pair> >* pointer_to_comm_table_order2=&comm_table_order2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > > rank_local_momentum_map2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > >* pointer_to_rank_local_momentum_map2=&rank_local_momentum_map2;
      std::vector<communication_pair> scattering_communication_table_vector_sort2;
      std::vector<communication_pair>* temp_pointer_vector_sort2 =
        &scattering_communication_table_vector_sort2;
      convert_set_to_vector(temp_pointer_absorption, temp_pointer_vector_sort2);

      //order the communication table
      get_communication_order_parallel(temp_pointer_vector_sort2, pointer_to_comm_table_order2,
                                       pointer_to_rank_local_momentum_map2, false);

      comm_table_order->insert(comm_table_order->end(),pointer_to_comm_table_order2->begin(),
                               pointer_to_comm_table_order2->end());
      rank_local_momentum_map->insert(rank_local_momentum_map->end(),
                                      pointer_to_rank_local_momentum_map2->begin(),
                                      pointer_to_rank_local_momentum_map2->end());
    }
  }
  else if (level == std::string("fully_parallel_maximum_first"))
  {
    bool maximum_first = true;

    communication_rank_sort(temp_pointer, temp_pointer_vector_sort,
                            maximum_first);
    //order the communication table
    get_communication_order_parallel(temp_pointer_vector_sort, comm_table_order,
                                     rank_local_momentum_map, true);
    if(scattering_communication_table2.size()>0)
    {
      std::vector<std::set<communication_pair> > comm_table_order2;
      std::vector<std::set<communication_pair> >* pointer_to_comm_table_order2=&comm_table_order2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > > rank_local_momentum_map2;
      std::vector<std::map<int, std::set<std::vector<NemoMeshPoint> > > >* pointer_to_rank_local_momentum_map2=&rank_local_momentum_map2;
      std::vector<communication_pair> scattering_communication_table_vector_sort2;
      std::vector<communication_pair>* temp_pointer_vector_sort2 =
        &scattering_communication_table_vector_sort2;
      convert_set_to_vector(temp_pointer_absorption, temp_pointer_vector_sort2);

      //order the communication table
      get_communication_order_parallel(temp_pointer_vector_sort2, pointer_to_comm_table_order2,
                                       pointer_to_rank_local_momentum_map2, false);

      comm_table_order->insert(comm_table_order->end(),pointer_to_comm_table_order2->begin(),
                               pointer_to_comm_table_order2->end());
      rank_local_momentum_map->insert(rank_local_momentum_map->end(),
                                      pointer_to_rank_local_momentum_map2->begin(),
                                      pointer_to_rank_local_momentum_map2->end());
    }

  }
  else
  {
    throw runtime_error( prefix + "option build_communication_table  = " + level
                         + " is not implemented   \n");

  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_full_MPI_solve(Propagator*& result)
{

  //if the reduced version of the communication table that contain one entry only needed
  //call the second version do_full_MPI_solve_reduced() and return
  if(options.get_option("single_entry_comm_table",false))
  {
    do_full_MPI_solve_reduced(result);
    return;
  }
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_full_MPI_solve ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::do_full_MPI_solve ";

  std::vector<bool> this_is_for_absorption;
  std::vector<bool> coupling_to_same_energy;

  //check that the Propagation is initialized...
  if(!Propagation_is_initialized)
    initialize_Propagation();

  //1. fill the global_job_list
  update_global_job_list();

  //2. determine the scattering type and some scattering type specific variables
  std::string scattering_type=std::string("");
  if(options.check_option("scattering_type"))
    scattering_type = options.get_option("scattering_type",std::string(""));
  else
    throw std::invalid_argument(prefix+"please define \"scattering_type\"\n");

  //for inelastic scattering:
  NemoPhys::Propagator_type result_type=get_Propagator_type(result->get_name());
  const Propagator* same_type_as_self_energy_propagator=NULL;
  Simulation* same_type_as_self_energy_propagator_solver=NULL;
  const Propagator* different_type_as_self_energy_propagator=NULL;
  Simulation* different_type_as_self_energy_propagator_solver=NULL;
  //for elastic scattering:
  Simulation* propagator_source=NULL;
  const Propagator* temp_propagator=NULL;

  if(scattering_type=="deformation_potential" || scattering_type == "roughness")
  {
    //2.1 check that there is exactly one readable Propagator and get it as the input_Propagator
    NEMO_ASSERT(Propagators.size()==2,prefix
                +"received incorrect number of Propagators (has to be one writeable and one readable for deformation_potential or general elastic scattering\n");
    //get the readable Propagator
    std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
    if(c_it->first==result->get_name())
      c_it++;
    propagator_source=find_source_of_data(c_it->first);
    //propagator_source->get_data(c_it->first,temp_propagator); //using get_data we make sure that this Propagator is initialized
    if(result_type==NemoPhys::Fermion_retarded_self)
    {
      //here, there are two Propagators needed: a retarded and a lesser Green's function
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_retarded_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
    }
    else if(result_type==NemoPhys::Fermion_lesser_self)
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      different_type_as_self_energy_propagator=same_type_as_self_energy_propagator;
      different_type_as_self_energy_propagator_solver=same_type_as_self_energy_propagator_solver;
    }
    else
      throw std::invalid_argument(prefix+"not implemented for Propagator type \""+Propagator_type_map.find(result_type)->second+"\"\n");
  }
  else if(scattering_type=="optical_deformation_potential" || scattering_type=="polar_optical_Froehlich")
  {
    //2.1 determine the type of the self-energy and find the required Propagators accordingly
    if(result_type==NemoPhys::Fermion_retarded_self)
    {
      //here, there are two Propagators needed: a retarded and a lesser Green's function
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_retarded_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,different_type_as_self_energy_propagator,
          different_type_as_self_energy_propagator_solver);
    }
    else if(result_type==NemoPhys::Fermion_lesser_self)
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      different_type_as_self_energy_propagator=same_type_as_self_energy_propagator;
      different_type_as_self_energy_propagator_solver=same_type_as_self_energy_propagator_solver;
    }
    else
      throw std::invalid_argument(prefix+"not implemented for Propagator type \""+Propagator_type_map.find(result_type)->second+"\"\n");
  }
  else
    throw std::runtime_error(prefix+"scattering type \""+scattering_type+"\" not implemented yet.\n");



  //3 & 4 determine the communication table due to the scattering type and order it.
  std::vector < std::set<communication_pair> > comm_table_order_obj;
  std::vector < std::map<int, std::set<std::vector<NemoMeshPoint> > > > rank_local_momentum_map_obj;

  std::vector < std::set<communication_pair> >* comm_table_order = &comm_table_order_obj;
  std::vector < std::map<int, std::set<std::vector<NemoMeshPoint> > > >* rank_local_momentum_map = &rank_local_momentum_map_obj;

  std::string comm_holder_name = options.get_option("communication_table_holder", std::string(""));

  if (comm_holder_name == std::string(""))
  {
    build_comm_table(comm_table_order, rank_local_momentum_map);
  }
  else
  {
    std::string comm_solver_name = options.get_option("communication_table_holder", std::string(""));
    Simulation* temp_simulation = find_simulation(comm_solver_name);
    bool temp;
    bool* isTableStored = &temp;
    std::string variable =std::string("");
    NEMO_ASSERT(temp_simulation!=NULL,
                prefix+": Simulation \""+comm_solver_name+"\" has not been found!\n");
    temp_simulation->get_data(variable,comm_table_order, rank_local_momentum_map,
                              isTableStored);
    if ((*isTableStored) == false)
    {
      build_comm_table(comm_table_order, rank_local_momentum_map);
    }
  }
  //-----------------------------------------------------------------------
  //First Loop :: //for 5 (optional): perform local calculations upfront
  //---------------------------------------------------------------------
  //---------------------------------------------------------------------
  if(get_save_local_sigma())
  {
    std::vector<const Propagator*> list_of_propagators_to_integrate;
    std::vector<Simulation*> list_of_propagator_solvers;
    if(temp_propagator!=NULL) //elastic scattering
    {
      list_of_propagators_to_integrate.resize(1,temp_propagator);
      list_of_propagator_solvers.resize(1,propagator_source);
    }
    else //inelastic scattering
    {
      if(result_type==NemoPhys::Fermion_retarded_self && (scattering_type=="optical_deformation_potential" || scattering_type=="polar_optical_Froehlich") )
      {
        list_of_propagators_to_integrate.resize(2,same_type_as_self_energy_propagator);
        list_of_propagators_to_integrate[1]=different_type_as_self_energy_propagator;
        list_of_propagator_solvers.resize(2,same_type_as_self_energy_propagator_solver);
        list_of_propagator_solvers[1]=different_type_as_self_energy_propagator_solver;
      }
      else
      {
        list_of_propagators_to_integrate.resize(1,same_type_as_self_energy_propagator);
        list_of_propagator_solvers.resize(1,same_type_as_self_energy_propagator_solver);
      }
    }
    save_local_sigma(comm_table_order,rank_local_momentum_map,list_of_propagators_to_integrate,
                     list_of_propagator_solvers,scattering_type);
  }

  //5.0 get the spatially-global rank of this MPI-process
  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);

  //5.1 loop over the ordered communication table and do the communication between ranks,
  //-----------------------------------------------------------------------
  //Second Loop :: This loop does the communication only or communication
  //           + post processing based on the option type_of_MPI_reduce
  //---------------------------------------------------------------------
  //---------------------------------------------------------------------
  //this will hold the sigma after communication to be saved - for Green's functions of the same type as the self-energy.
  std::vector<std::vector<std::complex<double> > * > result_pointer_vector1;
  //this will hold the sigma after communication to be saved - for Green's functions of the different type as the self-energy.
  std::vector<std::vector<std::complex<double> > * > result_pointer_vector2;
  // vector set of momentum that will be integrated.
  std::vector <std::set<std::vector<NemoMeshPoint> > > pointer_set_of_local_final_moment_vector;
  std::vector < std::vector<NemoMeshPoint> > self_energy_momentum_vector;
  vector< std::set<std::vector<NemoMeshPoint> > > final_coupling_energy_vector;
  std::string type_of_MPI_reduce=options.get_option("type_of_MPI_reduce",std::string("synchronous_split"));
  for(unsigned int i=0; i<comm_table_order->size(); i++)
  {
    msg<<prefix<<"communication iteration: "<<i<<"\n";
    //5.1 determine the MPI communicator for all MPI-communication pairs of this iteration
    std::set<communication_pair>::const_iterator single_communication_it;
    std::set<communication_pair>::const_iterator temp_communication_it=(*comm_table_order)[i].begin();

    MPI_Comm self_energy_comm=MPI_COMM_NULL;
    if(type_of_MPI_reduce != "synchronous_split")
    {
      self_energy_comm=holder.one_partition_total_communicator;
    }

    bool perform_local_calculation = false;
    bool this_process_is_receiver  = false;
    int communication_pair_counter = 0;
    //group of ranks that will be communicate together to be used by MPI_NEMO_Reduce()
    std::vector<int> group;
    //used to remove the duplicate ranks in group.
    std::set<int> group_temp;
    //5.1.1 loop over all communication_pairs of this communication iteration until the pair this MPI-process belongs to is found
    for(; temp_communication_it!=(*comm_table_order)[i].end()&& !perform_local_calculation; temp_communication_it++,communication_pair_counter++)
    {
      //5.1.1.1 check whether the initial momentum is solved on this MPI-process
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*temp_communication_it).first);
      NEMO_ASSERT(job_cit!=global_job_list.end(),prefix+"inconsistent search within global_job_list\n");
      //save the ranks in the set to be used by MPI_NEMO_Reduce()
      //this is the rank of the receiver.
      group.push_back(job_cit->second);
      if(job_cit->second==my_local_rank)
      {
        perform_local_calculation=true;
        this_process_is_receiver =true;
        single_communication_it=temp_communication_it;
      }

      //5.1.1.2 check whether at least one of the involved final momenta is solved on this MPI-process
      const std::set<std::vector<NemoMeshPoint> >& temp_communication_set =(*temp_communication_it).second;
      std::set<std::vector<NemoMeshPoint> >::const_iterator temp_set_cit=temp_communication_set.begin();
      for(; temp_set_cit!=temp_communication_set.end()/* && !perform_local_calculation*/; temp_set_cit++)
      {
        job_cit=global_job_list.find((*temp_set_cit));
        NEMO_ASSERT(job_cit!=global_job_list.end(),prefix+"inconsistent search within global_job_list\n");
        //save the ranks in the set to be used by MPI_NEMO_Reduce()
        group_temp.insert(job_cit->second);
        if(job_cit->second==my_local_rank)
        {
          perform_local_calculation=true;
          single_communication_it=temp_communication_it;
        }
      }
      // save the ranks that will communicate together after removing the duplicates.
      if((perform_local_calculation==true)||(this_process_is_receiver==true))
      {
        //iterate over the set and save it in the vector the first
        //element in the vector have to be the receiver.

        std::set<int>::iterator group_temp_it=group_temp.begin();
        for(; group_temp_it!=group_temp.end(); group_temp_it++)
        {
          //to avoid having the receiver rank twice in the vector
          if((*group_temp_it)!=group[0])
          {
            group.push_back(*group_temp_it);
          }
        }
      }
      else
      {
        //if the local process will not do any processing for that communication pair
        // clear the group vector and the set group_temp
        group.clear();
        group_temp.clear();
      }
    }
    //5.1.3 generate the MPI_Comm for this communication problem using the MPI_Comm_split command (color is communication_pair_counter, key depends on this_process_is_receiver)
    int root_rank = 0;
    //if the option synchronous_split then the group of ranks will not used
    //instead the MPI_comm_splir() will be used to make new communicator
    if(type_of_MPI_reduce=="synchronous_split")
    {
      int key = root_rank+1;
      if(this_process_is_receiver) key = root_rank;
      if(perform_local_calculation)
      {
        MPI_Comm_split(holder.one_partition_total_communicator, communication_pair_counter, key, &self_energy_comm);
        int new_rank;
        MPI_Comm_rank(self_energy_comm,&new_rank);
        NEMO_ASSERT((new_rank==0&&this_process_is_receiver) || (new_rank!=0&&!this_process_is_receiver),prefix+"inconsistent rank definition\n");
      }
      else
        MPI_Comm_split(holder.one_partition_total_communicator, MPI_UNDEFINED, key, &self_energy_comm);
    }
    else
    {
      // in case of using asynchronous MPI_NEMO_Reduce: the receiver rank
      // which is the root address exists in the first element of vector group
      root_rank = group[0];
    }
    //5.2 do the actual calculation for single_communication_it
    if(perform_local_calculation)
    {
      //momentum for which the self-energy will be calculated
      std::vector<NemoMeshPoint> self_energy_momentum=(*single_communication_it).first;
      //get the rank of the root for the self-energy of this momentum...
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find(self_energy_momentum);
      NEMO_ASSERT(job_cit!=global_job_list.end(),prefix+"inconsistent find result in global_job_list of self-energy momentum\n");
      //int global_root_rank = job_cit->second;


      //set of final scattering momenta, calculated on this MPI-process
      const std::set<std::vector<NemoMeshPoint> >* pointer_set_of_local_final_momenta=NULL;
      std::set<std::vector<NemoMeshPoint> > set_of_local_final_momenta;
      //5.2.1 check for local final momenta
      std::map<int,std::set<std::vector<NemoMeshPoint> > >::const_iterator MPI_cit=(*rank_local_momentum_map)[i].find(my_local_rank);
      if(MPI_cit!=(*rank_local_momentum_map)[i].end())
      {
        //pointer_set_of_local_final_momenta=&(MPI_cit->second);
        set_of_local_final_momenta=MPI_cit->second;
        //loop over all set_of_local_final_momenta and delete those that are not in (*single_communication_it).second
        std::set<std::vector<NemoMeshPoint> >::iterator temp_it=set_of_local_final_momenta.begin();
        while(temp_it!=set_of_local_final_momenta.end())
        {
          std::set<std::vector<NemoMeshPoint> >::const_iterator temp_cit_local=(*single_communication_it).second.find(*temp_it);
          if(temp_cit_local==(*single_communication_it).second.end())
            set_of_local_final_momenta.erase(temp_it++);
          else
            temp_it++;
        }
        pointer_set_of_local_final_momenta=&(set_of_local_final_momenta);
      }
      //hold the final results of sigma
      //PetscMatrixParallelComplex* result_matrix=NULL;
      //These vectors hold the intermediate results after integration to be multiplied by the prefactor
      std::vector<std::complex<double> >* pointer_to_result_vector1 = new std::vector<std::complex<double> >;
      std::vector<std::complex<double> >* pointer_to_result_vector2 = new std::vector<std::complex<double> >;
      if(scattering_type=="deformation_potential")
      {
        //5.3 solve the scattering self-energy
        scattering_deformation_potential_phonon(result, same_type_as_self_energy_propagator, root_rank,self_energy_comm, i, self_energy_momentum, pointer_set_of_local_final_momenta,
                                                pointer_to_result_vector1,pointer_to_result_vector2,group);
        // In case of synchronous post processing will be done directly after each communication
        // iteration. Otherwise the post processing and saving the results will be done after all
        // communication iterations.
        if((type_of_MPI_reduce == "synchronous_split")&&this_process_is_receiver)
        {
          //0. try to find the matrix of this momentum and if found, let result_matrix point to it
          Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum));
          NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");

          save_and_postprocess_deformation_phonon(result, same_type_as_self_energy_propagator, self_energy_momentum, pointer_set_of_local_final_momenta,
                                                  pointer_to_result_vector1,pointer_to_result_vector2,prop_map_it->second);
          delete pointer_to_result_vector1;
          pointer_to_result_vector1=NULL;
          delete pointer_to_result_vector2;
          pointer_to_result_vector2=NULL;
        }
        //5.4 set the job_done_momentum_map to true for this momentum
        set_job_done_momentum_map(&(result->get_name()), &self_energy_momentum, true);
      }
      else if(scattering_type=="optical_deformation_potential")
      {
        //2. solve the scattering self-energy
        //scattering_optical_deformation_potential_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
        //    same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,root_rank,
        //    self_energy_comm, self_energy_momentum, pointer_set_of_local_final_momenta,pointer_to_result_vector1,pointer_to_result_vector2,group);


        double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_energy_momentum, result);
        double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,(*(*single_communication_it).second.begin()), result);
        bool this_is_for_absorption=true;
        bool coupling_to_same_energy = false;

        if(self_energy==representative_coupling_energy)
        {
          coupling_to_same_energy = true;
        }
        else
        {
          if(self_energy>representative_coupling_energy)
          {
            this_is_for_absorption=false;
          }
        }

        //2. solve the scattering self-energy
        scattering_optical_deformation_potential_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
            same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,root_rank,
            self_energy_comm, self_energy_momentum, pointer_set_of_local_final_momenta,pointer_to_result_vector1,pointer_to_result_vector2,group,
            this_is_for_absorption);



        // In case of synchronous post processing will be done directly after eact communication
        // iteration. Otherwise the post processing and saving the results will be done after all
        // communication iterations.
        if((type_of_MPI_reduce == "synchronous_split")&&this_process_is_receiver)
        {
          //0. try to find the matrix of this momentum and if found, let result_matrix point to it
          Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum));
          NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");

          save_and_postprocess_deformation_optical_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
              same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,
              (self_energy_momentum), this_is_for_absorption, coupling_to_same_energy, pointer_set_of_local_final_momenta,&((*single_communication_it).second),
              pointer_to_result_vector1, pointer_to_result_vector2,prop_map_it->second);
          delete pointer_to_result_vector1;
          pointer_to_result_vector1=NULL;
          delete pointer_to_result_vector2;
          pointer_to_result_vector2=NULL;
        }
        //5.4 set the job_done_momentum_map to true for this momentum
        set_job_done_momentum_map(&(result->get_name()), &self_energy_momentum, true);
      }
      else if(scattering_type=="roughness")
      {
        //5.3 solve the scattering self-energy
        scattering_roughness(result, same_type_as_self_energy_propagator, root_rank,self_energy_comm, i, self_energy_momentum, pointer_set_of_local_final_momenta,
                                                pointer_to_result_vector1,group);
        // In case of synchronous post processing will be done directly after each communication
        // iteration. Otherwise the post processing and saving the results will be done after all
        // communication iterations.
        if((type_of_MPI_reduce == "synchronous_split")&&this_process_is_receiver)
        {
          //0. try to find the matrix of this momentum and if found, let result_matrix point to it
          Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum));
          NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");

          save_and_postprocess_roughness(result, temp_propagator, self_energy_momentum, pointer_set_of_local_final_momenta,
                                                  pointer_to_result_vector1,prop_map_it->second);
          delete pointer_to_result_vector1;
          pointer_to_result_vector1=NULL;
          delete pointer_to_result_vector2;
          pointer_to_result_vector2=NULL;
        }
        //5.4 set the job_done_momentum_map to true for this momentum
        set_job_done_momentum_map(&(result->get_name()), &self_energy_momentum, true);
      }
      else if(scattering_type=="polar_optical_Froehlich")
      {
          double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_energy_momentum, result);
          double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,(*(*single_communication_it).second.begin()), result);
          bool this_is_for_absorption=true;
          bool coupling_to_same_energy = false;

          if(self_energy==representative_coupling_energy)
          {
            coupling_to_same_energy = true;
          }
          else
          {
            if(self_energy>representative_coupling_energy)
            {
              this_is_for_absorption=false;
            }
          }

        //2. solve the scattering self-energy
        scattering_polar_optical_Froehlich_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
            same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,root_rank,
            self_energy_comm, self_energy_momentum, pointer_set_of_local_final_momenta,pointer_to_result_vector1,pointer_to_result_vector2,group,this_is_for_absorption);



        // In case of synchronous post processing will be done directly after eact communication
        // iteration. Otherwise the post processing and saving the results will be done after all
        // communication iterations.
        if((type_of_MPI_reduce == "synchronous_split")&&this_process_is_receiver)
        {
          //0. try to find the matrix of this momentum and if found, let result_matrix point to it
          Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum));
          NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");

          save_and_postprocess_polar_optical_Froehlich_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
              same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,
              (self_energy_momentum), this_is_for_absorption, coupling_to_same_energy, pointer_set_of_local_final_momenta,&((*single_communication_it).second),
              pointer_to_result_vector1, pointer_to_result_vector2,prop_map_it->second);
          delete pointer_to_result_vector1;
          pointer_to_result_vector1=NULL;
          delete pointer_to_result_vector2;
          pointer_to_result_vector2=NULL;
        }
        //5.4 set the job_done_momentum_map to true for this momentum
        set_job_done_momentum_map(&(result->get_name()), &self_energy_momentum, true);
      }
      else
        throw std::runtime_error(prefix+"scattering type \""+scattering_type+"\" not implemented yet.\n");
      // In case of synchronous post processing will be done directly after each communication
      // iteration. Otherwise the post processing and saving the results will be done after all
      // communication iterations.

      if(this_process_is_receiver) //my_local_integration_rank==root_rank)
      {
        if(type_of_MPI_reduce != "synchronous_split")
        {
          // In case of asynchronous communication data will be saved temporary until
          // the end of the communication section
          result_pointer_vector1.push_back(pointer_to_result_vector1);
          result_pointer_vector2.push_back(pointer_to_result_vector2);
          double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_energy_momentum, result);
          double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,(*(*single_communication_it).second.begin()), result);
          bool temp_this_is_for_absorption=true;
          bool temp_coupling_to_same_energy = false;
          if (self_energy==representative_coupling_energy)
          {
            temp_coupling_to_same_energy = true;
          }
          else
          {
            if(self_energy>=representative_coupling_energy)
              temp_this_is_for_absorption=false;
          }
          this_is_for_absorption.push_back(temp_this_is_for_absorption);
          coupling_to_same_energy.push_back(temp_coupling_to_same_energy);
          pointer_set_of_local_final_moment_vector.push_back(set_of_local_final_momenta);
          self_energy_momentum_vector.push_back(self_energy_momentum);
          final_coupling_energy_vector.push_back((*single_communication_it).second);
        }
      }
      else
      {
        delete pointer_to_result_vector1;
        pointer_to_result_vector1 = NULL;
        delete pointer_to_result_vector2;
        pointer_to_result_vector2 = NULL;
      }
    }//end of if(perform_local_calculation)
    else
    {
      //do nothing, but wait until this MPI-process has someone to communicate too (at least to itself)
      msg<<prefix<<"waiting for other MPI-processes, rank="<<my_local_rank<<"\n";
    }
    if(type_of_MPI_reduce=="synchronous_split")
    {
      MPI_Barrier(holder.one_partition_total_communicator);
      MPI_Comm_free(&self_energy_comm);
    }
  }//end of the comm_order loop!
  //-----------------------------------------------------------------------
  //Third Loop :: This loop does the post processing
  //          based on the option type_of_MPI_reduce (not needed for "synchronous_split")
  //---------------------------------------------------------------------
  //---------------------------------------------------------------------
  // In case of asynchronous communication the data received at the communication
  // section will be post processed and saved  after finishing all the
  // communication operations
  if(type_of_MPI_reduce != "synchronous_split")
  {
    std::vector<std::complex<double> >* pointer_to_result_vector1;
    std::vector<std::complex<double> >* pointer_to_result_vector2;
    const std::set<std::vector<NemoMeshPoint> >* pointer_set_of_local_final_moment_temp;
    // std::vector<NemoMeshPoint>  *self_energy_momentum_vector;

    for(int i=0; i< (int) result_pointer_vector1.size(); i++)
    {
      //PetscMatrixParallelComplex* result_matrix=NULL;
      //resored saved data
      pointer_to_result_vector1=(result_pointer_vector1[i]);
      pointer_to_result_vector2=(result_pointer_vector2[i]);
      pointer_set_of_local_final_moment_temp=&(pointer_set_of_local_final_moment_vector[i]);
      if(scattering_type=="deformation_potential")
      {
        //0. try to find the matrix of this momentum and if found, let result_matrix point to it
        Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum_vector[i]));
        NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");
        // save the data
        save_and_postprocess_deformation_phonon(result, same_type_as_self_energy_propagator, self_energy_momentum_vector[i], pointer_set_of_local_final_moment_temp,
                                                pointer_to_result_vector1,pointer_to_result_vector2,prop_map_it->second);

        delete pointer_to_result_vector1;
        pointer_to_result_vector1=NULL;
        delete pointer_to_result_vector2;
        pointer_to_result_vector2=NULL;
      }
      else if(scattering_type=="optical_deformation_potential")
      {
        //0. try to find the matrix of this momentum and if found, let result_matrix point to it
        Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum_vector[i]));
        NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");
        save_and_postprocess_deformation_optical_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
            same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,
            (self_energy_momentum_vector[i]), this_is_for_absorption[i], coupling_to_same_energy[i], pointer_set_of_local_final_moment_temp,
            &(final_coupling_energy_vector[i]), pointer_to_result_vector1, pointer_to_result_vector2,prop_map_it->second);

        delete pointer_to_result_vector1;
        pointer_to_result_vector1=NULL;
        delete pointer_to_result_vector2;
        pointer_to_result_vector2=NULL;

      }
      else if(scattering_type=="roughness")
      {
        //0. try to find the matrix of this momentum and if found, let result_matrix point to it
        Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum_vector[i]));
        NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");
        // save the data
        save_and_postprocess_roughness(result, temp_propagator, self_energy_momentum_vector[i], pointer_set_of_local_final_moment_temp,
                                                pointer_to_result_vector1,prop_map_it->second);

        delete pointer_to_result_vector1;
        pointer_to_result_vector1=NULL;
        delete pointer_to_result_vector2;
        pointer_to_result_vector2=NULL;
      }
      else if(scattering_type=="polar_optical_Froehlich")
      {
        //0. try to find the matrix of this momentum and if found, let result_matrix point to it
        Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum_vector[i]));
        NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");
        // save the data
        save_and_postprocess_polar_optical_Froehlich_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
            same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,
            (self_energy_momentum_vector[i]), this_is_for_absorption[i], coupling_to_same_energy[i], pointer_set_of_local_final_moment_temp,
            &(final_coupling_energy_vector[i]), pointer_to_result_vector1, pointer_to_result_vector2,prop_map_it->second);

        delete pointer_to_result_vector1;
        pointer_to_result_vector1=NULL;
        delete pointer_to_result_vector2;
        pointer_to_result_vector2=NULL;
      }
      else
        throw std::runtime_error(prefix+"scattering type \""+scattering_type+"\" not implemented yet.\n");
    }
    MPI_Barrier(holder.one_partition_total_communicator);
  }
  //Save the results to the writable propagator combined propagator
  Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.begin();
  for(; prop_map_it!=result->propagator_map.end(); prop_map_it++)
  {
    NEMO_ASSERT(prop_map_it->second!=NULL,prefix+"result matrix pointer is NULL\n");
    //5.4 store the result in the Propagator
    write_propagator(result->get_name(),prop_map_it->first, prop_map_it->second);

    set_job_done_momentum_map(&(result->get_name()), &(prop_map_it->first), true);

    if(options.get_option("do_output_scattering_rate",false))
      calculate_scattering_rate(result,prop_map_it->first);
    //5.5 do the Propagator output
    if(do_outputL)
      print_Propagator(result,&(prop_map_it->first));
    //5.6 do cleanup if requested
    if(clean_up)
      delete_propagator_matrices(result,&(prop_map_it->first));
  }
  NemoUtils::toc(tic_toc_prefix);
}


void Self_energy::do_full_MPI_solve_reduced(Propagator*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_full_MPI_solve_reduced ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::do_full_MPI_solve_reduced ";

  std::vector<bool> this_is_for_absorption;
  std::vector<bool> coupling_to_same_energy;

  //check that the Propagation is initialized...
  if(!Propagation_is_initialized)
    initialize_Propagation();

  //1. fill the global_job_list
  //this->set_global_job_list();
  update_global_job_list();

  //2. determine the scattering type and some scattering type specific variables
  std::string scattering_type=std::string("");
  if(options.check_option("scattering_type"))
    scattering_type = options.get_option("scattering_type",std::string(""));
  else
    throw std::invalid_argument(prefix+"please define \"scattering_type\"\n");



  //for inelastic scattering:
  NemoPhys::Propagator_type result_type=get_Propagator_type(result->get_name());
  const Propagator* same_type_as_self_energy_propagator=NULL;
  Simulation* same_type_as_self_energy_propagator_solver=NULL;
  const Propagator* different_type_as_self_energy_propagator=NULL;
  Simulation* different_type_as_self_energy_propagator_solver=NULL;
  //for elastic scattering:
  
  const Propagator* temp_propagator=NULL;

  if(scattering_type=="deformation_potential")
  {
    //Simulation* propagator_source=NULL;
    //2.1 check that there is exactly one readable Propagator and get it as the input_Propagator
    NEMO_ASSERT(Propagators.size()==2,prefix
                +"received incorrect number of Propagators (has to be one writeable and one readable for deformation_potential scattering\n");
    //get the readable Propagator
    std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
    if(c_it->first==result->get_name())
      c_it++;
    //propagator_source=find_source_of_data(c_it->first);
    //propagator_source->get_data(c_it->first,temp_propagator); //using get_data we make sure that this Propagator is initialized
    if(result_type==NemoPhys::Fermion_retarded_self)
    {
      //here, there are two Propagators needed: a retarded and a lesser Green's function
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_retarded_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
    }
    else if(result_type==NemoPhys::Fermion_lesser_self)
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      different_type_as_self_energy_propagator=same_type_as_self_energy_propagator;
      different_type_as_self_energy_propagator_solver=same_type_as_self_energy_propagator_solver;
    }
    else
      throw std::invalid_argument(prefix+"not implemented for Propagator type \""+Propagator_type_map.find(result_type)->second+"\"\n");


  }
  else if(scattering_type=="optical_deformation_potential" || scattering_type=="polar_optical_Froehlich")
  {
    //2.1 determine the type of the self-energy and find the required Propagators accordingly
    if(result_type==NemoPhys::Fermion_retarded_self)
    {
      //here, there are two Propagators needed: a retarded and a lesser Green's function
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_retarded_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,different_type_as_self_energy_propagator,
          different_type_as_self_energy_propagator_solver);
    }
    else if(result_type==NemoPhys::Fermion_lesser_self)
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      different_type_as_self_energy_propagator=same_type_as_self_energy_propagator;
      different_type_as_self_energy_propagator_solver=same_type_as_self_energy_propagator_solver;
    }
    else
      throw std::invalid_argument(prefix+"not implemented for Propagator type \""+Propagator_type_map.find(result_type)->second+"\"\n");
  }
  else
    throw std::runtime_error(prefix+"scattering type \""+scattering_type+"\" not implemented yet.\n");


  unsigned int momentum_count =pointer_to_all_momenta->size();
  double energy_interval= options.get_option("scattering_energy_interval", 0.0);
  double momentum_interval= options.get_option("scattering_momentum_interval", 1e20);


  //5.0 get the spatially-global rank of this MPI-process
  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);

  //5.1 loop over the ordered communication table and do the communication between ranks,
  //-----------------------------------------------------------------------
  //Second Loop :: This loop does the communication only or communication
  //           + post processing based on the option type_of_MPI_reduce
  //---------------------------------------------------------------------
  //---------------------------------------------------------------------
  //this will hold the sigma after communication to be saved - for Green's functions of the same type as the self-energy.
  std::vector<std::vector<std::complex<double> > * > result_pointer_vector1;
  //this will hold the sigma after communication to be saved - for Green's functions of the different type as the self-energy.
  std::vector<std::vector<std::complex<double> > * > result_pointer_vector2;
  // vector set of momentum that will be integrated.
  std::vector <std::set<std::vector<NemoMeshPoint> > > pointer_set_of_local_final_moment_vector;
  std::vector < std::vector<NemoMeshPoint> > self_energy_momentum_vector;
  vector< std::set<std::vector<NemoMeshPoint> > > final_coupling_energy_vector;
  std::vector< communication_pair> output_comm_table_obj;
  std::vector< communication_pair>* output_comm_table=&output_comm_table_obj;
  std::vector< std::set< std::vector< NemoMeshPoint> > > local_point_obj;
  std::vector<std::set< std::vector< NemoMeshPoint> > >* local_point=&local_point_obj;
  double phonon_energy=options.get_option("optical_phonon_energy", 0.0);
  //bool save_local_sigma_v2 = options.get_option("save_local_sigma",false);
  bool save_local_sigma_v2 = get_save_local_sigma();
  local_sigma_maps_v2.clear();
  //print the communication table entry by entry
  //first clear the old table file.
  MPI_Comm self_energy_comm=holder.one_partition_total_communicator;
  print_comm_single_entry(  output_comm_table, true);
  for(unsigned int i=0; i<momentum_count; )
  {

    if(scattering_type=="deformation_potential")
    {
      if(!get_communication_continual_scattering_entry( energy_interval, momentum_interval,  output_comm_table, local_point,  i) )
      {
        break;
      }
    }
    else if (scattering_type=="optical_deformation_potential" || scattering_type=="polar_optical_Froehlich")
    {
      i=momentum_count;
      if(!get_communication_discrete_scattering_entry( phonon_energy, momentum_interval,  output_comm_table, local_point) )
      {
        break;
      }
    }
    //print the communication table entry by entry
    print_comm_single_entry(  output_comm_table, false);

    //5.1 determine the MPI communicator for all MPI-communication pairs of this iteration
    std::vector<communication_pair>::const_iterator single_communication_it;
    std::vector<communication_pair>::const_iterator temp_communication_it=output_comm_table->begin();
    std::vector<std::set<std::vector<NemoMeshPoint> > >::iterator local_point_it=local_point->begin();
    int communication_pair_counter = 0;
    //group of ranks that will be communicate to gether to be used by MPI_NEMO_Reduce()
    std::vector<int> group;
    //used to remove the duplicate ranks in group.
    std::set<int> group_temp;
    //5.1.1 loop over all communication_pairs of this communication iteration until the pair this MPI-process belongs to is found
    bool perform_local_calculation = false;
    bool this_process_is_receiver  = false;
    for(; temp_communication_it!=output_comm_table->end(); temp_communication_it++,communication_pair_counter++, local_point_it++)
    {
      perform_local_calculation = false;
      this_process_is_receiver  = false;
      //5.1.1.1 check whether the initial momentum is solved on this MPI-process
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*temp_communication_it).first);
      NEMO_ASSERT(job_cit!=global_job_list.end(),prefix+"inconsistent search within global_job_list\n");
      //save the ranks in the set to be used by MPI_NEMO_Reduce()
      //this is the rank of the receiver.
      group_temp.clear();
      group.clear();
      group.push_back(job_cit->second);
      if(job_cit->second==my_local_rank)
      {
        perform_local_calculation=true;
        this_process_is_receiver =true;
        single_communication_it=temp_communication_it;
      }

      //5.1.1.2 check whether at least one of the involved final momenta is solved on this MPI-process
      const std::set<std::vector<NemoMeshPoint> >& temp_communication_set =(*temp_communication_it).second;
      std::set<std::vector<NemoMeshPoint> >::const_iterator temp_set_cit=temp_communication_set.begin();
      for(; temp_set_cit!=temp_communication_set.end(); temp_set_cit++)
      {
        job_cit=global_job_list.find((*temp_set_cit));
        NEMO_ASSERT(job_cit!=global_job_list.end(),prefix+"inconsistent search within global_job_list\n");
        //save the ranks in the set to be used by MPI_NEMO_Reduce()
        group_temp.insert(job_cit->second);
        if(job_cit->second==my_local_rank)
        {
          perform_local_calculation=true;
          single_communication_it=temp_communication_it;
        }
      }
      // save the ranks that will communicate together after removing the duplicates.
      if((perform_local_calculation==true)||(this_process_is_receiver==true))
      {
        //iterate over the set and save it in the vector the first
        //element in the vector have to be the receiver.

        std::set<int>::iterator group_temp_it=group_temp.begin();
        for(; group_temp_it!=group_temp.end(); group_temp_it++)
        {
          //to avoid having the receiver rank twice in the vector
          if((*group_temp_it)!=group[0])
          {
            group.push_back(*group_temp_it);
          }
        }
      }
      else
      {
        //if the local process will not do any processing for that communication pair
        // clear the group vector and the set group_temp
        group.clear();
        group_temp.clear();
      }

      //5.1.3 generate the MPI_Comm for this communication problem using the MPI_Comm_split command (color is communication_pair_counter, key depends on this_process_is_receiver)
      int root_rank = group[0];

      //5.2 do the actual calculation for single_communication_it
      if(perform_local_calculation)
      {
        //momentum for which the self-energy will be calculated
        std::vector<NemoMeshPoint> self_energy_momentum=(*single_communication_it).first;
        //get the rank of the root for the self-energy of this momentum...
        std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find(self_energy_momentum);
        NEMO_ASSERT(job_cit!=global_job_list.end(),prefix+"inconsistent find result in global_job_list of self-energy momentum\n");


        //set of final scattering momenta, calculated on this MPI-process
        const std::set<std::vector<NemoMeshPoint> >* pointer_set_of_local_final_momenta=NULL;
        std::set<std::vector<NemoMeshPoint> > set_of_local_final_momenta;
        set_of_local_final_momenta=(*local_point_it);
        pointer_set_of_local_final_momenta=&(set_of_local_final_momenta);

        //hold the final results of sigma
        //PetscMatrixParallelComplex* result_matrix=NULL;
        //This vectors hold the intermediate results after integration to be multiplied by the prefactor
        std::vector<std::complex<double> >* pointer_to_result_vector1 = new std::vector<std::complex<double> >;
        std::vector<std::complex<double> >* pointer_to_result_vector2 = new std::vector<std::complex<double> >;
        if(scattering_type=="deformation_potential")
        {
          //5.3 solve the scattering self-energy
          scattering_deformation_potential_phonon(result, temp_propagator, root_rank,self_energy_comm, i, self_energy_momentum, pointer_set_of_local_final_momenta,
                                                  pointer_to_result_vector1,pointer_to_result_vector2,group, save_local_sigma_v2);
          //5.4 set the job_done_momentum_map to true for this momentum
          set_job_done_momentum_map(&(result->get_name()), &self_energy_momentum, true);
        }
        else if(scattering_type=="optical_deformation_potential")
        {

          double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_energy_momentum, result);
          double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,(*(*single_communication_it).second.begin()), result);
          bool this_is_for_absorption=true;
          if(self_energy>representative_coupling_energy)
          {
            this_is_for_absorption=false;
          }

          //2. solve the scattering self-energy
          scattering_optical_deformation_potential_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
              same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,root_rank,
              self_energy_comm, self_energy_momentum, pointer_set_of_local_final_momenta,pointer_to_result_vector1,pointer_to_result_vector2,group,
              this_is_for_absorption);

        //5.4 set the job_done_momentum_map to true for this momentum
          set_job_done_momentum_map(&(result->get_name()), &self_energy_momentum, true);
        }
        else if("polar_optical_Froehlich")
        {
          double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_energy_momentum, result);
          double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,(*(*single_communication_it).second.begin()), result);
          bool this_is_for_absorption=true;
          if(self_energy>representative_coupling_energy)
          {
            this_is_for_absorption=false;
          }

          //2. solve the scattering self-energy
          scattering_polar_optical_Froehlich_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
              same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,root_rank,
              self_energy_comm, self_energy_momentum, pointer_set_of_local_final_momenta,pointer_to_result_vector1,pointer_to_result_vector2,group,
              this_is_for_absorption);

          //5.4 set the job_done_momentum_map to true for this momentum
          set_job_done_momentum_map(&(result->get_name()), &self_energy_momentum, true);
        }
        else
          throw std::runtime_error(prefix+"scattering type \""+scattering_type+"\" not implemented yet.\n");
        // In case of synchronous post processing will be done directly after each communication
        // iteration. Otherwise the post processing and saving the results will be done after all
        // communication iterations.

        if(this_process_is_receiver)
        {

          result_pointer_vector1.push_back(pointer_to_result_vector1);
          result_pointer_vector2.push_back(pointer_to_result_vector2);
          double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_energy_momentum, result);
          double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,(*(*single_communication_it).second.begin()), result);
          bool temp_this_is_for_absorption=true;
          bool temp_coupling_to_same_energy = false;
          if (self_energy==representative_coupling_energy)
          {
            temp_coupling_to_same_energy = true;
          }
          else
          {
            if(self_energy>=representative_coupling_energy)
              temp_this_is_for_absorption=false;
          }
          this_is_for_absorption.push_back(temp_this_is_for_absorption);
          coupling_to_same_energy.push_back(temp_coupling_to_same_energy);
          pointer_set_of_local_final_moment_vector.push_back(set_of_local_final_momenta);
          self_energy_momentum_vector.push_back(self_energy_momentum);
          final_coupling_energy_vector.push_back((*single_communication_it).second);

        }
        else
        {
          delete pointer_to_result_vector1;
          pointer_to_result_vector1 = NULL;
          delete pointer_to_result_vector2;
          pointer_to_result_vector2 = NULL;
        }
      }//end of if(perform_local_calculation)
      else
      {
        //do nothing, but wait until this MPI-process has someone to communicate too (at least to itself)
        msg<<prefix<<"waiting for other MPI-processes, rank="<<my_local_rank<<"\n";
      }
    }
  }//end of the comm_order loop!

  //-----------------------------------------------------------------------
  //Third Loop :: This loop do the post processing
  //          based on the option type_of_MPI_reduce
  //---------------------------------------------------------------------
  //---------------------------------------------------------------------
  // In case of asynchronous communication the data received at the communication
  // section will be post processed and saved  after finishing all the
  // communication operations
  // finialize the nonblocking communication started in the previous loop.
  Propagation::MPI_NEMO_Reduce_nonblocking_finalize( MPI_DOUBLE_COMPLEX ,self_energy_comm);

  std::vector<std::complex<double> >* pointer_to_result_vector1;
  std::vector<std::complex<double> >* pointer_to_result_vector2;
  const std::set<std::vector<NemoMeshPoint> >* pointer_set_of_local_final_moment_temp;

  for(int i=0; i< (int) result_pointer_vector1.size(); i++)
  {
    //PetscMatrixParallelComplex* result_matrix=NULL;
    //resored saved data
    pointer_to_result_vector1=(result_pointer_vector1[i]);
    pointer_to_result_vector2=(result_pointer_vector2[i]);
    pointer_set_of_local_final_moment_temp=&(pointer_set_of_local_final_moment_vector[i]);
    if(scattering_type=="deformation_potential")
    {
      //0. try to find the matrix of this momentum and if found, let result_matrix point to it
      Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum_vector[i]));
      NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");
      // save the data
      save_and_postprocess_deformation_phonon(result, same_type_as_self_energy_propagator, self_energy_momentum_vector[i], pointer_set_of_local_final_moment_temp,
                                              pointer_to_result_vector1,pointer_to_result_vector2,prop_map_it->second);

      delete pointer_to_result_vector1;
      pointer_to_result_vector1=NULL;
      delete pointer_to_result_vector2;
      pointer_to_result_vector2=NULL;

    }
    else if(scattering_type=="optical_deformation_potential")
    {
      //0. try to find the matrix of this momentum and if found, let result_matrix point to it
      Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum_vector[i]));
      NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");
      save_and_postprocess_deformation_optical_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
          same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,
          (self_energy_momentum_vector[i]), this_is_for_absorption[i], coupling_to_same_energy[i], pointer_set_of_local_final_moment_temp,
          &(final_coupling_energy_vector[i]), pointer_to_result_vector1, pointer_to_result_vector2,prop_map_it->second);

      delete pointer_to_result_vector1;
      pointer_to_result_vector1=NULL;
      delete pointer_to_result_vector2;
      pointer_to_result_vector2=NULL;

    }
    else if(scattering_type=="polar_optical_Froehlich")
    {
      //0. try to find the matrix of this momentum and if found, let result_matrix point to it
     Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.find((self_energy_momentum_vector[i]));
     NEMO_ASSERT(prop_map_it!=result->propagator_map.end(),prefix+"cant find the self_energy_momentum in the propagator_map \n");
     save_and_postprocess_polar_optical_Froehlich_phonon(result, same_type_as_self_energy_propagator, different_type_as_self_energy_propagator,
        same_type_as_self_energy_propagator_solver, different_type_as_self_energy_propagator_solver,
        (self_energy_momentum_vector[i]), this_is_for_absorption[i], coupling_to_same_energy[i], pointer_set_of_local_final_moment_temp,
        &(final_coupling_energy_vector[i]), pointer_to_result_vector1, pointer_to_result_vector2,prop_map_it->second);

     delete pointer_to_result_vector1;
     pointer_to_result_vector1=NULL;
     delete pointer_to_result_vector2;
     pointer_to_result_vector2=NULL;
    }
    else
      throw std::runtime_error(prefix+"scattering type \""+scattering_type+"\" not implemented yet.\n");

  }

  //Save the results to the writable propagator combined propagator
  Propagator::PropagatorMap::iterator prop_map_it=result->propagator_map.begin();
  for(; prop_map_it!=result->propagator_map.end(); prop_map_it++)
  {
    NEMO_ASSERT(prop_map_it->second!=NULL,prefix+"result matrix pointer is NULL\n");
    //5.4 store the result in the Propagator
    write_propagator(result->get_name(),prop_map_it->first, prop_map_it->second);

    set_job_done_momentum_map(&(result->get_name()), &(prop_map_it->first), true);

    if(options.get_option("do_output_scattering_rate",false))
      calculate_scattering_rate(result,prop_map_it->first);
    //5.5 do the Propagator output
    if(do_outputL)
      print_Propagator(result,&(prop_map_it->first));
    //5.6 do cleanup if requested
    if(clean_up)
      delete_propagator_matrices(result,&(prop_map_it->first));
  }
  NemoUtils::toc(tic_toc_prefix);
}


// This function load the self energy matrix from binary file.
void Self_energy::do_full_MPI_load(Propagator*& result, int sc_it_counter,std::string load_poisson_it_number_str,std::string file_name )
{

  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_full_MPI_load ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::do_full_MPI_load ";

  //check that the Propagation is initialized...
  if(!Propagation_is_initialized)
    initialize_Propagation();

  //1. fill the global_job_list
  update_global_job_list();

  //for inelastic scattering:
  //2. determine the scattering type and some scattering type specific variables
  std::string scattering_type=std::string("");
  if(options.check_option("scattering_type"))
    scattering_type = options.get_option("scattering_type",std::string(""));
  else
    throw std::invalid_argument(prefix+"please define \"scattering_type\"\n");

  //for inelastic scattering:
  NemoPhys::Propagator_type result_type=get_Propagator_type(result->get_name());
  const Propagator* same_type_as_self_energy_propagator=NULL;
  Simulation* same_type_as_self_energy_propagator_solver=NULL;

  //for elastic scattering:
  
  const Propagator* temp_propagator=NULL;
  if(scattering_type=="deformation_potential" || scattering_type == "roughness")
  {
    //Simulation* propagator_source=NULL;
    //2.1 check that there is exactly one readable Propagator and get it as the input_Propagator
    NEMO_ASSERT(Propagators.size()==2,prefix
                +"received incorrect number of Propagators (has to be one writeable and one readable for deformation_potential scattering\n");
    //get the readable Propagator
    std::map<std::string, const Propagator*>::const_iterator c_it=Propagators.begin();
    if(c_it->first==result->get_name())
      c_it++;
    //propagator_source=find_source_of_data(c_it->first);
    //propagator_source->get_data(c_it->first,temp_propagator); //using get_data we make sure that this Propagator is initialized
    if(result_type==NemoPhys::Fermion_retarded_self)
    {
      //here, there are two Propagators needed: a retarded and a lesser Green's function
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_retarded_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
    }
    else if(result_type==NemoPhys::Fermion_lesser_self)
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      //different_type_as_self_energy_propagator=same_type_as_self_energy_propagator;
      //different_type_as_self_energy_propagator_solver=same_type_as_self_energy_propagator_solver;
    }
    else
      throw std::invalid_argument(prefix+"not implemented for Propagator type \""+Propagator_type_map.find(result_type)->second+"\"\n");
    temp_propagator = same_type_as_self_energy_propagator;
  }
  else if(scattering_type=="optical_deformation_potential" || scattering_type=="polar_optical_Froehlich")
  {
    //2.1 determine the type of the self-energy and find the required Propagators accordingly
    if(result_type==NemoPhys::Fermion_retarded_self)
    {
      //here, there are two Propagators needed: a retarded and a lesser Green's function
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_retarded_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      temp_propagator= same_type_as_self_energy_propagator;
    }
    else if(result_type==NemoPhys::Fermion_lesser_self)
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,same_type_as_self_energy_propagator, same_type_as_self_energy_propagator_solver);
      temp_propagator= same_type_as_self_energy_propagator;
    }
    else
      throw std::invalid_argument(prefix+"not implemented for Propagator type \""+Propagator_type_map.find(result_type)->second+"\"\n");
  }
  else
    throw std::runtime_error(prefix+"scattering type \""+scattering_type+"\" not implemented yet.\n");

  //5.0 get the spatially-global rank of this MPI-process
  Propagator::PropagatorMap::iterator momentum_it= result->propagator_map.begin();

  //The local rank to be used in the file name
  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);
  std::string my_local_rank_str;
  ostringstream convert_rank;
  convert_rank<<my_local_rank;
  my_local_rank_str = convert_rank.str();

  //counter to be used in the name of the loaded file
  int mom_counter = 0;
  PetscMatrixParallelComplex* temp_matrix=NULL;

  for(; momentum_it!=result->propagator_map.end(); momentum_it++,mom_counter++)
  {
    //1. find the solver of the retarded Green's function
    Simulation* solver = find_source_of_data(temp_propagator);

    //1.1 get the local DOFs from the retarded Green's function (pick the first available matrix)
    PetscMatrixParallelComplex* Green=NULL;
    solver->get_data(temp_propagator->get_name(),&(momentum_it->first),Green);
    if(Green->if_container())
      Green->assemble();
    int start_local_row;
    int end_local_row;
    Green->get_ownership_range(start_local_row,end_local_row);

    //2. define which section of the input Green's function to include for the scattering self-energy - if set_of_row_col_indices is not defined yet
    if(set_of_row_col_indices.size()==0)
      fill_set_of_row_col_indices(start_local_row,end_local_row);

    int number_of_off_diagonals = options.get_option("store_offdiagonals",0);
    unsigned int number_of_own_super_rows = 0;
    Hamilton_Constructor->get_data("local_size",number_of_own_super_rows);
    int temp_locals=std::min(2*number_of_off_diagonals+1,int(number_of_own_super_rows));//number_of_off_diagonals+1;

    //whatever is left
    int temp_nonlocals=std::min(number_of_off_diagonals,int(number_of_own_super_rows-temp_locals));




    //convert the momentum counter to string to be used as a part of  the file name
    string mom_str;
    ostringstream convert_mom;
    convert_mom<<mom_counter;
    mom_str = convert_mom.str();

    //convert the sc porn iteration counter to string to be used as a part of the file name
    string sc_it_counter_str;
    ostringstream convert_sc_it_counter;
    convert_sc_it_counter<<sc_it_counter;
    sc_it_counter_str = convert_sc_it_counter.str();


    //3 create matrix to hold the data
    temp_matrix = new PetscMatrixParallelComplex(Green->get_num_rows(),Green->get_num_cols(),
        get_simulation_domain()->get_communicator());

    temp_matrix->set_num_owned_rows(number_of_own_super_rows);

    std::vector<int> local_nonzeros = std::vector<int> (number_of_own_super_rows,temp_locals);
    std::vector<int> nonlocal_nonzeros = std::vector<int> (number_of_own_super_rows,temp_nonlocals);

    for (int i = 0; i < start_local_row; i++)
      temp_matrix->set_num_nonzeros(i,0,0);
    for (int i = start_local_row; i < end_local_row; i++)
      temp_matrix->set_num_nonzeros(i,local_nonzeros[i],nonlocal_nonzeros[i]);
    for (unsigned int i = end_local_row; i < Green->get_num_rows(); i++)
      temp_matrix->set_num_nonzeros(i,0,0);

    temp_matrix->allocate_memory();
    temp_matrix->set_to_zero();
    //4 read the matrix from binary file

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

    //5 The flag for job done
    set_job_done_momentum_map(&(result->get_name()),&(momentum_it->first), true);

    //6 save the matrix to the output propagator
    write_propagator(result->get_name(),(momentum_it->first), temp_matrix);

    //7 do output for scattering rate
    if(options.get_option("do_output_scattering_rate",false))
      calculate_scattering_rate(result,(momentum_it->first));

    //8 do the Propagator output
    if(do_outputL)
      print_Propagator(result,&(momentum_it->first));

    //9 do cleanup if requested
    if(clean_up)
      delete_propagator_matrices(result,&(momentum_it->first));
  }
  MPI_Barrier(holder.one_partition_total_communicator);
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_retarded(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                    PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_retarded ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy("+this->get_name()+")::do_solve_retarded: ";



  std::string Propagator_name=output_Propagator->get_name();
  std::map<std::string, NemoPhys::Propagator_type>::const_iterator type_c_it=Propagator_types.find(Propagator_name);
  NEMO_ASSERT(type_c_it!=Propagator_types.end(),prefix+"have not found propagator \""+Propagator_name+"\" in the type map\n");
  if(type_c_it->second==NemoPhys::Fermion_retarded_self || type_c_it->second==NemoPhys::Boson_retarded_self)
  {
    //call the appropiate do_solve_retarded method:
    if(Propagator_name.find(std::string("constant_eta"))!=std::string::npos)
      do_solve_constant_eta(output_Propagator,momentum_point,result);
    else if(Propagator_name.find(std::string("Buettiker_probe"))!=std::string::npos)
      do_solve_Buettiker_retarded(output_Propagator,momentum_point,result);
    else if(Propagator_name.find(std::string("contact"))!=std::string::npos)
    {
      do_solve_retarded_contact(output_Propagator,momentum_point,result);
    }
    else if(Propagator_name.find(std::string("scattering"))!=std::string::npos)
      do_solve_scattering_retarded(output_Propagator,momentum_point,result);
    else
      throw std::invalid_argument(prefix+"called with a Propagator name\""+Propagator_name+"\" that is not known how to interpret\n");
  }
  else
  {
    std::string prop_type=Propagator_type_map.find(type_c_it->second)->second;
    throw std::runtime_error(prefix+"called with unknown Propagator type:\""+prop_type+"\"\n");
  }
  set_job_done_momentum_map(&(output_Propagator->get_name()), &momentum_point, true);
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_scattering_retarded(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_scattering_retarded ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy("+this->get_name()+")::do_solve_scattering_retarded: ";

  //1. get the type of the output_Propagator (either Boson or Fermion retarded self-energy)
  std::map<std::string, NemoPhys::Propagator_type>::const_iterator Propagator_type_c_it=Propagator_types.find(output_Propagator->get_name());
  NEMO_ASSERT(Propagator_type_c_it!=Propagator_types.end(),prefix+"have not found the propagator type of \""+output_Propagator->get_name()+"\"\n");
  NemoPhys::Propagator_type output_propagator_type=Propagator_type_c_it->second;

  //2. get the solver of the retarded Green's function of this momentum that as the same particle type as the output_Propagator (ret_Green)
  //we assume that there is exactly one retarded Green's function in the list of the readable Propagators that has the same particle type as the output_Propagator
  //this is a prerequisite for all scattering self energies in the self-consistent Born approximation (NOTE: add a flag - Buettiker probe later)
  const Propagator* ret_Green=NULL;
  Simulation* ret_Green_solver=NULL;
  if(output_propagator_type==NemoPhys::Fermion_retarded_self)
  {
    find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_retarded_Green,ret_Green,ret_Green_solver);
    NEMO_ASSERT(ret_Green!=NULL,prefix+"pointer to propagator of type \"Fermion_retarded_Green\" is NULL\n");
  }
  else
  {
    find_solver_and_readable_propagator_of_type(NemoPhys::Boson_retarded_Green,ret_Green,ret_Green_solver);
    NEMO_ASSERT(ret_Green!=NULL,prefix+"pointer to propagator of type \"Boson_retarded_Green\" is NULL\n");
  }

  //3. and get the solver of the retarded Green's function of this momentum that as different particle type as the output_Propagator (ret_scattering_Green)
  //and we assume that there is exactly one retarded Green's function in the list of the readable Propagators that has the different particle type as the output_Propagator
  //if this Propagator and its solver is given, we assume Sigma = D*G*potential
  //otherwise we will use Sigma = F*G, with F being a scattering type specific function
  std::string scattering_type=std::string("");
  if(options.check_option("scattering_type"))
  {
    scattering_type = options.get_option("scattering_type",std::string(""));
  }
  else
    throw std::invalid_argument(prefix+"please define \"scattering_type\"\n");
  if(scattering_type=="lambda_G_scattering")
  {
     //scattering self-energy is the Green's function multiplied with lambda
     //read in the proportionality constant lambda
     //check whether material specific lambda is specified
     if(options.get_option("material_specific_lambda",false))
     {
        scattering_lambda_proportional_to_G(output_Propagator, ret_Green, ret_Green_solver, momentum_point,result);
     }
     else
     {
        double lambda=options.get_option("lambda",0.0);
        scattering_lambda_proportional_to_G(output_Propagator, ret_Green, ret_Green_solver, lambda, momentum_point,result);
     }
  }

  else
    throw std::invalid_argument(prefix+"scattering of type \""+scattering_type+"\" not implemented, yet\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::scattering_roughness(Propagator*& output_Propagator, const Propagator*& input_Propagator,const int root_address,
        const MPI_Comm& MPI_communicator, const int communication_number,
        const std::vector<NemoMeshPoint>& self_momentum_point, const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
        std::vector<std::complex<double> >*& pointer_to_result_vector,
        std::vector <int>& group,  bool save_local_sigma_v2)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::scattering_roughness ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy("+this->get_name()+")::scattering_roughness: ";
  
  const std::vector<vector<double> >reciprocal_basis = get_simulation_domain()->get_reciprocal_vectors();
  int rec_basis_size = reciprocal_basis.size();
  NEMO_ASSERT(rec_basis_size==0,prefix + " roughness NYI implemented for periodic structures");
  
  bool only_store_lower = options.get_option("store_lower_triangular_matrix",false);
  
  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //0. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {
    if(relevant_local_momenta->size()==0)
    {
      Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
      NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
    }
    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }
  else
  {
    Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
    NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
  }
  
  
  //1. get regions that are to be considered rough
  std::vector<unsigned int> regions;
  options.get_option("roughness_regions",regions);
  
  //cerr << " regions to consider " << regions[0] << " \n";
  
  //2. get the distance map for setting up the sparsity pattern
  EOMMatrixInterface* eom_source=dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
  NEMO_ASSERT(eom_source!=NULL,prefix+Hamilton_Constructor->get_name()+"is not an EOMMatrixInterface \n");

  std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, vector<double> > > distance_vector_map;   
  eom_source->get_distance_vector_map(distance_vector_map,NULL,&regions);
  NEMO_ASSERT(!distance_vector_map.empty(), prefix + " distance map is empty. Did you set the interaction radius for the given material? \n");
  
  if(debug_output)
  {
    ofstream outfile;
    std::string filename = get_name() + "_distance_vector_map.dat";
    outfile.open(filename.c_str());
    outfile << "row \t column \t distance\n ";


    std::map<std::pair<unsigned int, unsigned int>,std::pair<unsigned int, vector<double> > >::iterator it = distance_vector_map.begin();
    for(; it != distance_vector_map.end(); ++it)
    {
      //std::pair<unsigned int, unsigned int> temp_pair = it->first;
      outfile << it->first.first << "\t" << it->first.second << "\t" ;
      for(unsigned int i =0; i <it->second.second.size(); i++)
        outfile << it->second.second[i] << "\t";
      
      outfile << "\n";
    }
    
    outfile.close();
  }
  

  //3. find the solver of the retarded Green's function
  NemoPhys::Propagator_type temp_type=NemoPhys::Fermion_retarded_Green;
  if(!particle_type_is_Fermion)
    temp_type=NemoPhys::Boson_retarded_Green;
  Simulation* solver = NULL;
  if(exact_GR_solver!=NULL)
    solver=exact_GR_solver;
  else if(exact_GL_solver!=NULL)
  {
    solver=exact_GL_solver;
    temp_type=NemoPhys::Fermion_lesser_Green;
    if(!particle_type_is_Fermion)
      temp_type=NemoPhys::Boson_lesser_Green;
  }
  else
    solver=find_source_of_data(input_Propagator);
    
  
  PetscMatrixParallelComplex* Green=NULL;

  GreensfunctionInterface* green_source=dynamic_cast<GreensfunctionInterface*>(solver);
  NEMO_ASSERT(green_source!=NULL,prefix+solver->get_name()+"is not a GreensfunctionInterface\n");
  //solver->get_data(input_Propagator->get_name(),&momentum_point,Green);
  green_source->get_Greensfunction(momentum_point,Green,NULL,NULL, temp_type);
    
  //if(Green->if_container() /*&& !is_diagonal_only*/)
  //  Green->assemble();
   
  //Green->save_to_matlab_file(get_name()+".m");  
  //int start_local_row;
  //int end_local_row;
  //Green->get_ownership_range(start_local_row,end_local_row);
  //int size = end_local_row - start_local_row;
  //4. set up sparsity pattern from this map. This distance map does not include same atom (diagonal contribution) 
  if(set_of_row_col_indices.empty())
  {
    std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, vector<double> > >::iterator it = distance_vector_map.begin();

    if(!only_store_lower)
    {
      for(; it != distance_vector_map.end(); ++it)
      set_of_row_col_indices.insert(it->first);
    }
    else
    {
      //if(temp_type==NemoPhys::Fermion_retarded_Green)
      {
        for(; it != distance_vector_map.end(); ++it)
          if(it->first.first >= it->first.second)
            set_of_row_col_indices.insert(it->first);
      }
      //else //temporary for lesser case
      //  for(; it != distance_vector_map.end(); ++it)
       //   if(it->first.first == it->first.second)
       //     set_of_row_col_indices.insert(it->first);
    }
  }



  //std::set<std::string> dummy_exclude_list; //no mesh to exclude from integral
  std::set<std::string> exclude_integration;
  //find the energy mesh name
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
  std::string energy_name=std::string("");
  for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
    if(momentum_name_it->second==NemoPhys::Energy)
      energy_name=momentum_name_it->first;
  exclude_integration.insert(energy_name);
      
  std::map<std::set<std::vector<NemoMeshPoint> > ,std::vector<std::complex<double> > >* pointer_to_local_data=NULL;
  if(local_sigma_maps.size()!=0)
  {
    NEMO_ASSERT(local_sigma_maps.size()==1,prefix+"found inconsistent size of local_sigma_maps\n");
    pointer_to_local_data=&((local_sigma_maps.begin()->second)[communication_number]);
  }
  else if(save_local_sigma_v2)
  {
    pointer_to_local_data=&local_sigma_maps_v2;
  }
  
  //5. calculate the normal vector to plane through atoms for the wire case
  std::vector<double> normal(3,0.0);
  std::map<unsigned int, vector<vector<double> > > relevant_atom_positions_per_region; // stores atoms in each roughness region. This is separated because the normal vectors are calculated per region.
  std::map<unsigned int, vector<double> > normal_per_region; 
  if(rec_basis_size == 0)
  {
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
    //DOFmap&                dof_map = get_dof_map();
    const AtomicStructure& atoms   = domain->get_atoms();
    //const map< unsigned int,AtomStructNode >& lattice = atoms.lattice();
    ConstActiveAtomIterator it  = atoms.active_atoms_begin();
    ConstActiveAtomIterator end = atoms.active_atoms_end();
    
    //loop through atoms if the atom is part of a roughness region insert its position (x,y,z) into the map.
    for(; it != end; ++it)
    { 
      const AtomStructNode& nd = it.node();
      bool use_atom = true;
      unsigned int region = -1; 
      unsigned int nd_reg = nd.region;
      //std::vector<unsigned int>::iterator region_it = regions->find(nd_reg);
      std::vector<unsigned int>::iterator region_it = find(regions.begin(), regions.end(), nd_reg);
      if(region_it==regions.end())
        use_atom = false;
      else 
        region = (*region_it);
      
      if(use_atom)
      {
       //store position into the correct vector of points for region
       
       std::vector<double> temp_vector(3,nd.position[0]); 
       temp_vector[1] = nd.position[1]; 
       temp_vector[2] = nd.position[2]; 
       std::map<unsigned int, vector<vector<double> > >::iterator map_it = relevant_atom_positions_per_region.find(region);

       if(map_it == relevant_atom_positions_per_region.end())
       {
         
         std::vector<vector<double> > temp_vector_vector;
         temp_vector_vector.push_back(temp_vector);
         relevant_atom_positions_per_region[region] = temp_vector_vector;
       }
       else
       {
         map_it->second.push_back(temp_vector);
       }
      }
    
    }
    //call helper function to get normal 
    unsigned int num_regions = regions.size(); 
    for(unsigned int i = 0; i < num_regions; ++i)
    {
      std::map<unsigned int, vector<vector<double> > >::iterator map_it = relevant_atom_positions_per_region.find(regions[i]);
      NEMO_ASSERT(map_it!=relevant_atom_positions_per_region.end(), prefix + "could not find atom positions for region");
      normal_per_region[regions[i]] = NemoMath::calculate_plane_from_points(map_it->second);
    }
     
  }
  
  double correlation_length = options.get_option("correlation_length", 0.0);
  std::string correlation_type = options.get_option("correlation_type", std::string("exponential")); //right now : exponential or Gaussian
  //string to bool for fast access
  bool use_exponential_correlation = true;
  if(correlation_type != "exponential")
    use_exponential_correlation = false;// for now this means Gaussian 
  std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*> local_weight_function_map;
  
  //6. //loop through set of distance_vector_map and put into the local weight function map
  //this weighting is geometry dependent 
  
  std::set<double> normal_distance_set;
  double dz = 0.0; 
  vector<double> z_mesh;
  if(rec_basis_size == 0)
  {
    std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, vector<double> > >::iterator map_it = distance_vector_map.begin();
    //loop through once just to find the unique normal distances 
    for(; map_it != distance_vector_map.end(); ++map_it)
    {
      vector<double> distance_vector_between_atoms = map_it->second.second;
      vector<double> normal_vector = normal_per_region[map_it->second.first];
      double normal_distance = NemoMath::vector_dot(distance_vector_between_atoms,normal_vector);
      normal_distance_set.insert(normal_distance);
    }
    unsigned int num_z = normal_distance_set.size();
    NEMO_ASSERT(num_z > 1, prefix + " normal_distance_set should have at least have 2 entries");
    z_mesh.resize(num_z); //this is the mesh along the normal direction for the distance vectors
    set<double>::iterator set_it = normal_distance_set.begin();
    double z_min = (*set_it);
    set_it = normal_distance_set.end();
    --set_it;
    double z_max = (*set_it);
    for(unsigned int i = 0; i < num_z; ++i)
    {
      z_mesh[i] = z_min + i*(z_max-z_min)/(num_z-1);
    }
    dz = std::abs(z_mesh[1]-z_mesh[0]); //normalization factor since I use a homogenous grid
  }
  
  double roughness_region_distance = options.get_option("roughness_region_distance",options.get_option("step_height",0.0));

  //loop through again to do the prefactor 
  std::map<std::pair<unsigned int, unsigned int>, std::map<std::vector<NemoMeshPoint>, double > > temp_maps;
  std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, vector<double> > >::iterator map_it = distance_vector_map.begin();
  for(; map_it != distance_vector_map.end(); ++map_it)
  {

    //calculate exponential factor depending on reciprocal basis size 
    double weight_prefactor = 1.0;
    if(rec_basis_size == 0)
    {
      
      //need delta function dependent on normal to interface
      vector<double> distance_vector_between_atoms = map_it->second.second;
      vector<double> normal_vector = normal_per_region[map_it->second.first];
      double normal_distance = NemoMath::vector_dot(distance_vector_between_atoms,normal_vector);
      //if(//normal_distance < dz)
     if(normal_distance < roughness_region_distance)
      {

      //normal_distance_set.insert(normal_distance);
      vector<double> normal_distance_vector(3,normal_distance*normal_vector[0]);
      normal_distance_vector[1] = normal_distance*normal_vector[1]; 
      normal_distance_vector[2] = normal_distance*normal_vector[2]; 
      //need exponental dependent on parallel to interface 
      //take distance between atom

      vector<double> parallel_component(3,std::abs(distance_vector_between_atoms[0]-normal_distance_vector[0]));
      parallel_component[1] = std::abs(distance_vector_between_atoms[1] - normal_distance_vector[1]);
      parallel_component[2] = std::abs(distance_vector_between_atoms[2] - normal_distance_vector[2]);
      double parallel_distance = NemoMath::sqrt(parallel_component[0]*parallel_component[0]+parallel_component[1]*parallel_component[1]+parallel_component[2]*parallel_component[2]);

      if(!use_exponential_correlation)
        weight_prefactor *= NemoMath::exp(-parallel_distance*parallel_distance/(correlation_length*correlation_length));
      else //exponential
      {
        double distance = NemoMath::sqrt(distance_vector_between_atoms[0]*distance_vector_between_atoms[0] + distance_vector_between_atoms[1]*distance_vector_between_atoms[1]
                                         + distance_vector_between_atoms[2]*distance_vector_between_atoms[2]);
        weight_prefactor *= NemoMath::exp(-sqrt(2)*distance/correlation_length);
      }
      //weight_prefactor *= NemoMath::exp(-parallel_distance/correlation_length)/(dz);


      //weight_prefactor *= NemoMath::exp(-sqrt(2)*parallel_distance*parallel_distance/(correlation_length*correlation_length))/(dz);
      //double distance = NemoMath::sqrt(distance_vector_between_atoms[0]*distance_vector_between_atoms[0] + distance_vector_between_atoms[1]*distance_vector_between_atoms[1]
      //                                 + distance_vector_between_atoms[2]*distance_vector_between_atoms[2]);
      
      //weight_prefactor *= NemoMath::exp(-sqrt(2)*distance/correlation_length)/dz;
      //OMEN style 
      //double distance = NemoMath::sqrt(distance_vector_between_atoms[0]*distance_vector_between_atoms[0] + distance_vector_between_atoms[1]*distance_vector_between_atoms[1]
      //                                + distance_vector_between_atoms[2]*distance_vector_between_atoms[2]);
      //weight_prefactor *= NemoMath::exp(-sqrt(2)*distance/correlation_length)/dz;
      }
      else 
        weight_prefactor = 0.0;
      
      //no dependence on momentum for weighting prefactor .. only on spatial coordinates. 
      if(!only_store_lower)
      {
      std::map<std::vector<NemoMeshPoint>, double> temp_map;
      std::set<std::vector<NemoMeshPoint> >::iterator local_momenta_it = relevant_local_momenta->begin();//relevant_local_momenta
      for(; local_momenta_it != relevant_local_momenta->end(); ++local_momenta_it)
        temp_map[*local_momenta_it] = weight_prefactor;        
      temp_maps[map_it->first] = temp_map;
      local_weight_function_map[map_it->first] = &(temp_maps[map_it->first]);
      }
      else
      {
        //if(temp_type==NemoPhys::Fermion_retarded_Green)
        {

          if(map_it->first.first >= map_it->first.second)
          {
            std::map<std::vector<NemoMeshPoint>, double> temp_map;
            std::set<std::vector<NemoMeshPoint> >::iterator local_momenta_it = relevant_local_momenta->begin();//relevant_local_momenta
            for(; local_momenta_it != relevant_local_momenta->end(); ++local_momenta_it)
              temp_map[*local_momenta_it] = weight_prefactor;
            temp_maps[map_it->first] = temp_map;
            local_weight_function_map[map_it->first] = &(temp_maps[map_it->first]);
          }
        }
        /*else //temporary for lesser case
        {
          if(map_it->first.first == map_it->first.second)
          {
            std::map<std::vector<NemoMeshPoint>, double> temp_map;
            std::set<std::vector<NemoMeshPoint> >::iterator local_momenta_it = relevant_local_momenta->begin();//relevant_local_momenta
            for(; local_momenta_it != relevant_local_momenta->end(); ++local_momenta_it)
              temp_map[*local_momenta_it] = weight_prefactor;
            temp_maps[map_it->first] = temp_map;
            local_weight_function_map[map_it->first] = &(temp_maps[map_it->first]);
          }

        }*/
      }
    }
       
  }
  
  const std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*>* pointer_to_weight_function=&local_weight_function_map;
  //6. call the integrate submatrix method
  NemoPhys::Propagator_type propagator_type = get_Propagator_type(input_Propagator->get_name());
  PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices, relevant_local_momenta, solver, propagator_type, MPI_communicator,root_address, group,
                      pointer_to_result_vector, exclude_integration, false/*is_diagonal_only*/,pointer_to_weight_function,pointer_to_local_data); //for n-dim periodicity: [eV/eV/nm^n]
  
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::save_and_postprocess_roughness(Propagator*& /*output_Propagator*/,
        const Propagator*& /*input_Propagator*/,
        const std::vector<NemoMeshPoint>& self_momentum_point,
        const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
        std::vector<std::complex<double> >*& pointer_to_result_vector,
        PetscMatrixParallelComplex*& result)
{
  std::string prefix="Self_energy("+this->get_name()+")::save_and_postprocess_roughness: ";

  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //1. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {

    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }

  //1. find the solver of the retarded Green's function
  NemoPhys::Propagator_type temp_type=NemoPhys::Fermion_retarded_Green;
  Simulation* solver =NULL;
  if(exact_GR_solver!=NULL)
  {
    solver=exact_GR_solver;
    if(!particle_type_is_Fermion)
      temp_type=NemoPhys::Boson_retarded_Green;
  }
  else if(exact_GL_solver!=NULL)
  {
    solver=exact_GL_solver;
    if(particle_type_is_Fermion)
      temp_type=NemoPhys::Fermion_lesser_Green;
    else
      temp_type=NemoPhys::Boson_lesser_Green;
  }
  else
    throw std::runtime_error(prefix+"should not run into this\n");
  //solver=find_source_of_data(input_Propagator);
  //1.1 get the local DOFs from the retarded Green's function (pick the first available matrix)
  PetscMatrixParallelComplex* Green=NULL;
  GreensfunctionInterface* temp_interface=dynamic_cast<GreensfunctionInterface*>(solver);
  NEMO_ASSERT(temp_interface!=NULL,prefix+solver->get_name()+"is not a GreensfunctionInterface\n");
  temp_interface->get_Greensfunction(momentum_point,Green,NULL,NULL,temp_type);
  //solver->get_data(input_Propagator->get_name(),&momentum_point,Green);

  int start_local_row;
  int end_local_row;
  Green->get_ownership_range(start_local_row,end_local_row);

  //2. postprocessing if this MPI-rank is the root_rank

  NEMO_ASSERT(pointer_to_result_vector->size()==set_of_row_col_indices.size(),prefix+"mismatch in the result_vector size\n");
      
  //3. translate the result_vector into a matrix and store in result
  delete result;
  //3.1 set up the result matrix
  result = new PetscMatrixParallelComplex(Green->get_num_rows(),Green->get_num_cols(),
                                          get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);

  //calculate prefactor
  double step_height = options.get_option("step_height",0.0); 
  double perturbing_potential = options.get_option("perturbing_potential",0.0);
  double prefactor = step_height*perturbing_potential*perturbing_potential;
  //4.1 loop over the set_of_row_col_indices
  std::set<std::pair<int,int> >::const_iterator set_cit=set_of_row_col_indices.begin();
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
    
  }
  
  result->set_num_owned_rows(Green->get_num_owned_rows());
  for (int i = 0; i < start_local_row; i++)
    result->set_num_nonzeros(i,0,0);
  for (int i = start_local_row; i < end_local_row; i++)
  {
    nonzero_it = nonzero_map.find(i);
    //NEMO_ASSERT(nonzero_it != nonzero_map.end(),prefix + " could not find the number of nonzeros ");
    double num_nonzeros = 0; 
    if(nonzero_it != nonzero_map.end())
      num_nonzeros = nonzero_it->second;
    result->set_num_nonzeros(i,num_nonzeros,0);
    //result->set_num_nonzeros(i,off_diagonals+1,off_diagonals);
  }
  for (unsigned int i = end_local_row; i < Green->get_num_rows(); i++)
    result->set_num_nonzeros(i,0,0);
  result->allocate_memory();
  result->set_to_zero();
  //4.2 loop over the set_of_row_col_indices (running index counter is the key of the result_vector)
  unsigned int counter=0;
  set_cit=set_of_row_col_indices.begin();
  for(; set_cit!=set_of_row_col_indices.end(); set_cit++)
  {
    int row_index = (*set_cit).first;
    int col_index = (*set_cit).second;
    result->set(row_index,col_index,(*pointer_to_result_vector)[counter]*prefactor); //for n-dim periodicity: [eV*m^3/nm^n]

    counter++;
  }
  result->assemble();  
}

void Self_energy::scattering_deformation_potential_phonon(Propagator*& output_Propagator, const Propagator*& input_Propagator, const int root_address,
    const MPI_Comm& MPI_communicator, const int communication_number,
    const std::vector<NemoMeshPoint>& self_momentum_point, const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
    std::vector<std::complex<double> >*& pointer_to_result_vector,
    std::vector<std::complex<double> >*& /*pointer_to_result_vector2*/ ,
    std::vector<int>& group,
    bool save_local_sigma_v2)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::scattering_deformation_potential_phonon2 ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy("+this->get_name()+")::retarded_scattering_deformation_potential_phonon: ";
  int temp_address;
  MPI_Comm_rank(MPI_communicator,&temp_address);
  msg<<prefix<<"got root_address: "<<root_address<<"\twith my own address being: "<<temp_address<<"\n";
  msg<<"for the momentum:\n";
  for(unsigned int i=0; i<self_momentum_point.size(); i++)
    self_momentum_point[i].print();

  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //1. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {
    if(relevant_local_momenta->size()==0)
    {
      //std::map<std::vector<NemoMeshPoint>, int>::const_iterator c_it=global_job_list.find(self_momentum_point);
      //NEMO_ASSERT(c_it!=global_job_list.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
      Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
      NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
    }
    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }
  else
  {
    /*std::map<std::vector<NemoMeshPoint>, int>::const_iterator c_it=global_job_list.find(self_momentum_point);
    NEMO_ASSERT(c_it!=global_job_list.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");*/
    Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
    NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
  }
  //1. find the solver of the retarded Green's function
  NemoPhys::Propagator_type temp_type=NemoPhys::Fermion_retarded_Green;
  if(!particle_type_is_Fermion)
    temp_type=NemoPhys::Boson_retarded_Green;
  Simulation* solver = NULL;
  if(exact_GR_solver!=NULL)
    solver=exact_GR_solver;
  else if(exact_GL_solver!=NULL)
  {
    solver=exact_GL_solver;
    temp_type=NemoPhys::Fermion_lesser_Green;
    if(!particle_type_is_Fermion)
      temp_type=NemoPhys::Boson_lesser_Green;
  }
  else
    solver=find_source_of_data(input_Propagator);

  //1.1 get the local DOFs from the retarded Green's function (pick the first available matrix)
  PetscMatrixParallelComplex* Green=NULL;

  GreensfunctionInterface* green_source=dynamic_cast<GreensfunctionInterface*>(solver);
  NEMO_ASSERT(green_source!=NULL,prefix+solver->get_name()+"is not a GreensfunctionInterface\n");
  //solver->get_data(input_Propagator->get_name(),&momentum_point,Green);
  green_source->get_Greensfunction(momentum_point,Green,NULL,NULL, temp_type);

  bool is_diagonal_only=true;
  int off_diagonals = options.get_option("store_offdiagonals",0);
  bool store_atom_blockdiagonal = options.get_option("store_atom_blockdiagonal",false);
  PropagationOptionsInterface* opt_interface = dynamic_cast<PropagationOptionsInterface*>(this);
  if(off_diagonals>0 || store_atom_blockdiagonal || opt_interface->get_compute_blockdiagonal_self_energy())
    is_diagonal_only=false;
  if(Green->if_container() )//&& !is_diagonal_only)
    Green->assemble();
  int start_local_row;
  int end_local_row;
  Green->get_ownership_range(start_local_row,end_local_row);

  //2. define which section of the input Green's function to include for the scattering self-energy - if set_of_row_col_indices is not defined yet
  if(set_of_row_col_indices.size()==0)
    fill_set_of_row_col_indices(start_local_row,end_local_row);

  //3. call the integrate_submatrix method (covers the MPI_Reduce command)


  std::set<std::string> dummy_exclude_list; //no mesh to exclude from integral
  std::map<std::set<std::vector<NemoMeshPoint> > ,std::vector<std::complex<double> > >* pointer_to_local_data=NULL;
  if(local_sigma_maps.size()!=0)
  {
    NEMO_ASSERT(local_sigma_maps.size()==1,prefix+"found inconsistent size of local_sigma_maps\n");
    pointer_to_local_data=&((local_sigma_maps.begin()->second)[communication_number]);
  }
  else if(save_local_sigma_v2)
  {
    pointer_to_local_data=&local_sigma_maps_v2;
  }
  //4. find the local weight if using the triangular averaging
  std::map<std::vector<NemoMeshPoint>, double> weight_function;
  if(options.get_option("use_triangular_averaging",bool(false)))
  {
    if(relevant_local_momenta != NULL)
    {
      std::string energy_name;
     std::map<std::string, NemoPhys::Momentum_type>::const_iterator temp_cit=momentum_mesh_types.begin();
     for(; temp_cit!=momentum_mesh_types.end()&&energy_name.size()==0; ++temp_cit)
       if(temp_cit->second==NemoPhys::Energy)
         energy_name=temp_cit->first;
     double energy_scattering_interval = options.get_option("scattering_"+energy_name+"_interval",0.0); //[eV]
     //find the integration weight of this energy point
     double minimum_energy_interval=std::max(NemoMath::d_zero_tolerance,find_integration_weight(energy_name,self_momentum_point,input_Propagator));
     energy_scattering_interval=std::max(energy_scattering_interval,minimum_energy_interval);
      //get self-energy energy 
      double self_momentum_point_energy = PropagationUtilities::read_energy_from_momentum(this,self_momentum_point,output_Propagator);
     
      //loop through E' and prepare the weights
      std::set<std::vector<NemoMeshPoint> >::iterator it_final_momenta = relevant_local_momenta->begin();
      for(; it_final_momenta != relevant_local_momenta->end(); ++it_final_momenta)
      {
        double final_momentum_point_energy = PropagationUtilities::read_energy_from_momentum(this,*it_final_momenta, output_Propagator); 
       
        double ratio = 1-std::abs(final_momentum_point_energy-self_momentum_point_energy)/energy_scattering_interval;
        weight_function[*it_final_momenta] = ratio;
      }
        
   }
   std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*> local_weight_function_map;
   std::pair<int,int> local_index_pair(0,0);
   local_weight_function_map[local_index_pair]=&weight_function;
   const std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*>* pointer_to_weight_function=&local_weight_function_map;

   NemoPhys::Propagator_type propagator_type = get_Propagator_type(input_Propagator->get_name());
   PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices, relevant_local_momenta, solver, propagator_type, MPI_communicator,root_address, group,
                       pointer_to_result_vector, dummy_exclude_list, is_diagonal_only,pointer_to_weight_function,pointer_to_local_data); //for n-dim periodicity: [eV/eV/nm^n]   
  }
  else
  {
    //called with no weight
    const std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*>* pointer_to_weight_function=NULL;
    NemoPhys::Propagator_type input_type = get_Propagator_type(writeable_Propagator->get_name());
    NemoPhys::Propagator_type propagator_type=NemoPhys::Fermion_lesser_Green;
    if(input_type==NemoPhys::Fermion_retarded_self)
      propagator_type=NemoPhys::Fermion_retarded_Green;
    else if(input_type==NemoPhys::Boson_retarded_self)
      propagator_type=NemoPhys::Boson_retarded_Green;
    else if(input_type==NemoPhys::Boson_lesser_Green)
      propagator_type=NemoPhys::Boson_lesser_Green;
    PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices, relevant_local_momenta, solver, propagator_type, MPI_communicator,root_address, group,
        pointer_to_result_vector, dummy_exclude_list, is_diagonal_only,pointer_to_weight_function,pointer_to_local_data); //for n-dim periodicity: [eV/eV/nm^n]
  }
  NemoUtils::toc(tic_toc_prefix);
}
void Self_energy::save_and_postprocess_deformation_phonon(Propagator*& /*output_Propagator*/,
    const Propagator*& input_Propagator,
    const std::vector<NemoMeshPoint>& self_momentum_point,
    const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
    std::vector<std::complex<double> >*& pointer_to_result_vector,
    std::vector<std::complex<double> >*& /*pointer_to_result_vector2*/ ,
    PetscMatrixParallelComplex*& result)
{

  std::string prefix="Self_energy("+this->get_name()+")::save_and_postprocess_deformation_phonon: ";

  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //1. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {

    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }

  //1. find the solver of the retarded Green's function
  NemoPhys::Propagator_type temp_type=NemoPhys::Fermion_retarded_Green;
  Simulation* solver =NULL;
  if(exact_GR_solver!=NULL)
  {
    solver=exact_GR_solver;
    if(!particle_type_is_Fermion)
      temp_type=NemoPhys::Boson_retarded_Green;
  }
  else if(exact_GL_solver!=NULL)
  {
    solver=exact_GL_solver;
    if(particle_type_is_Fermion)
      temp_type=NemoPhys::Fermion_lesser_Green;
    else
      temp_type=NemoPhys::Boson_lesser_Green;
  }
  else
    throw std::runtime_error(prefix+"should not run into this\n");
  //solver=find_source_of_data(input_Propagator);
  //1.1 get the local DOFs from the retarded Green's function (pick the first available matrix)
  PetscMatrixParallelComplex* Green=NULL;
  GreensfunctionInterface* temp_interface=dynamic_cast<GreensfunctionInterface*>(solver);
  NEMO_ASSERT(temp_interface!=NULL,prefix+solver->get_name()+"is not a GreensfunctionInterface\n");
  temp_interface->get_Greensfunction(momentum_point,Green,NULL,NULL,temp_type);
  //solver->get_data(input_Propagator->get_name(),&momentum_point,Green);

  int start_local_row;
  int end_local_row;
  Green->get_ownership_range(start_local_row,end_local_row);

  bool is_diagonal_only=true;
  int off_diagonals = options.get_option("store_offdiagonals",0);
  bool store_atom_blockdiagonal = options.get_option("store_atom_blockdiagonal",false);
  PropagationOptionsInterface* opt_interface = dynamic_cast<PropagationOptionsInterface*>(this);
  if(off_diagonals>0 || store_atom_blockdiagonal || opt_interface->get_compute_blockdiagonal_self_energy())
    is_diagonal_only=false;
  //4. postprocessing if this MPI-rank is the root_rank

  if(options.check_option("store_atomic_offdiagonals"))
  {
    off_diagonals = options.get_option("store_atomic_offdiagonals",0);
    //find the number of orbitals per atom
    double temp_number_of_orbitals;
    Hamilton_Constructor->get_data("number_of_orbitals",temp_number_of_orbitals);
    int number_of_orbitals=int(temp_number_of_orbitals);
    off_diagonals *= number_of_orbitals;
  }
  NEMO_ASSERT(pointer_to_result_vector->size()==set_of_row_col_indices.size(),prefix+"mismatch in the result_vector size\n");

  //5. multiply with prefactors
  double deformation_potential = options.get_option("deformation_potential",1.0); //[eV]
  //double temperature=options.get_option("temperature",NemoPhys::temperature);
  //NEMO_ASSERT(options.check_option("material_density"),prefix+"please define \"material_density\" in [kg/m^3]\n");
  double material_density=options.get_option("material_density",1.0); //[kg/m^3]
  //NEMO_ASSERT(options.check_option("sound_velocity"),prefix+"please define \"sound velocity\" in [m/s]\n");
  double sound_velocity=options.get_option("sound_velocity",1.0);
  //5.1 if the user calls for elastic scattering - i.e. scattering_energy_interval<size of this energy channel, use the minimum as the interval
  //loop over all meshes to determine the energy mesh name
  std::string energy_name;
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator temp_cit=momentum_mesh_types.begin();
  for(; temp_cit!=momentum_mesh_types.end()&&energy_name.size()==0; ++temp_cit)
    if(temp_cit->second==NemoPhys::Energy)
      energy_name=temp_cit->first;
  double energy_scattering_interval = options.get_option("scattering_"+energy_name+"_interval",0.0); //[eV]
  //find the integration weight of this energy point
  double minimum_energy_interval=std::max(NemoMath::d_zero_tolerance,find_integration_weight(energy_name,self_momentum_point,input_Propagator));
  energy_scattering_interval=std::max(energy_scattering_interval,minimum_energy_interval);
  

  double prefactor=deformation_potential*deformation_potential*NemoPhys::boltzmann_constant*temperature/NemoPhys::elementary_charge; //[eV^3]
  prefactor/=2.0*energy_scattering_interval*material_density*sound_velocity*sound_velocity/NemoPhys::elementary_charge; //[eV*m^3]
  
  //5.2 take care of 1/(2*pi)^d factor from Fourier Transform
  const std::vector<vector<double> >reciprocal_basis = get_simulation_domain()->get_reciprocal_vectors();
  int rec_basis_size = reciprocal_basis.size();
  if(rec_basis_size > 0)
    prefactor /= std::pow(2*NemoMath::pi, rec_basis_size);


  //6. translate the result_vector into a matrix and store in result
  delete result;
  //6.1 set up the result matrix
  result = new PetscMatrixParallelComplex(Green->get_num_rows(),Green->get_num_cols(),
                                          get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
  if(!store_atom_blockdiagonal && !opt_interface->get_compute_blockdiagonal_self_energy())
  {
    //7.2 create the sparsity pattern for result
    result->set_num_owned_rows(Green->get_num_owned_rows());
    for (int i = 0; i < start_local_row; i++)
      result->set_num_nonzeros(i,0,0);
    for (int i = start_local_row; i < end_local_row; i++)
      result->set_num_nonzeros(i,2*off_diagonals+1,off_diagonals);
    for (unsigned int i = end_local_row; i < Green->get_num_rows(); i++)
      result->set_num_nonzeros(i,0,0);

  }
  else
  {
    //4.1 loop over the set_of_row_col_indices
    std::set<std::pair<int,int> >::const_iterator set_cit=set_of_row_col_indices.begin();
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

    }

    result->set_num_owned_rows(Green->get_num_owned_rows());
    for (int i = 0; i < start_local_row; i++)
      result->set_num_nonzeros(i,0,0);
    for (int i = start_local_row; i < end_local_row; i++)
    {
      nonzero_it = nonzero_map.find(i);
      //NEMO_ASSERT(nonzero_it != nonzero_map.end(),prefix + " could not find the number of nonzeros ");
      double num_nonzeros = 0;
      if(nonzero_it != nonzero_map.end())
        num_nonzeros = nonzero_it->second;
      result->set_num_nonzeros(i,num_nonzeros,0);
      //result->set_num_nonzeros(i,off_diagonals+1,off_diagonals);
    }
    for (unsigned int i = end_local_row; i < Green->get_num_rows(); i++)
      result->set_num_nonzeros(i,0,0);
  }
  result->allocate_memory();
  result->set_to_zero();
  if(!is_diagonal_only)
  {
    PetscMatrixParallelComplex* temp_matrix=NULL;
    Hamilton_Constructor->get_data(std::string("dimensional_S_matrix"),temp_matrix);//for n-dim periodicity: [1/nm^(3-n)]
    //everywhere gets the delta function .. this assumes all atoms are same volume.
    prefactor*=temp_matrix->get(0,0).real();
    delete temp_matrix;
    temp_matrix = NULL;
  }
  //7.3 loop over the set_of_row_col_indices (running index counter is the key of the result_vector)
  unsigned int counter=0;
  std::set<std::pair<int,int> >::const_iterator set_cit=set_of_row_col_indices.begin();
  for(; set_cit!=set_of_row_col_indices.end(); set_cit++)
  {
    int row_index = (*set_cit).first;
    int col_index = (*set_cit).second;
    result->set(row_index,col_index,(*pointer_to_result_vector)[counter]*prefactor); //for n-dim periodicity: [eV*m^3/nm^n]
    counter++;
  }

  if(is_diagonal_only)
  {
  //8. multiply with the Dirac delta function
  //8.1 get the Dirac delta function
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Hamilton_Constructor->get_data(std::string("dimensional_S_matrix"),temp_matrix);//for n-dim periodicity: [1/nm^(3-n)]
  if(options.get_option("debug_lambda",false))
    temp_matrix->save_to_matlab_file("dimensional_S_matrix.m");
  //8.2 perform the multiplication
  {
    PetscMatrixParallelComplex temp_matrix2(*result);
    delete result;
    result=NULL;
    PetscMatrixParallelComplex::mult(*temp_matrix,temp_matrix2,&result); //[eV*m^3/nm^n/nm^(3-n)]

    if(options.get_option("deformation_potential_tensor",false))
    {
      multiply_deformation_tensor("deformation_potential",result);
    }
  }
  delete temp_matrix;
  temp_matrix=NULL;
  }


  result->assemble();
  *result*=std::complex<double>(1.0/(NemoMath::nm_in_m*NemoMath::nm_in_m*NemoMath::nm_in_m),0.0); //[eV]
}

void Self_energy::multiply_deformation_tensor(std::string scattering_type, PetscMatrixParallelComplex*& )
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
   std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::multiply_deformation_tensor ";
   NemoUtils::tic(tic_toc_prefix);
   std::string prefix="Self_energy("+this->get_name()+")::multiply_deformation_tensor: ";

   //1. call get_data to get the nxn block from Schroedinger (Hamilton_Constructor->get_data(...,...,..)
   //PetscMatrixParallelComplex* deformation_potential_tensor = NULL;
   if(scattering_type == "deformation_potential")
   {

   }
   else if (scattering_type == "optical_deformation_potential")
   {

   }
   else
     throw std::runtime_error(prefix+"called with unknown scattering type\n");

   //2. for each atom block in result pre and post multiply by the deformation_potential_tensor.


   NemoUtils::toc(tic_toc_prefix);

}

void Self_energy::scattering_polar_optical_Froehlich_phonon(Propagator*& output_Propagator, const Propagator*& input_Propagator1,
    const Propagator*& input_Propagator2, Simulation* input_Propagator1_solver, Simulation* input_Propagator2_solver,
    const int root_address, const MPI_Comm& MPI_communicator,
    const std::vector<NemoMeshPoint>& self_momentum_point, const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
    std::vector<std::complex<double> >*& pointer_to_result_vector,
    std::vector<std::complex<double> >*& pointer_to_result_vector2,
    std::vector <int>& group,bool this_is_for_absorption)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::scattering_polar_optical_phonon ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy("+this->get_name()+")::scattering_polar_optical_phonon: ";

  //output for debugging:
  int temp_address;
  MPI_Comm_rank(MPI_communicator,&temp_address);
  msg<<prefix<<"got root_address: "<<root_address<<"\twith my own address being: "<<temp_address<<"\n";
  msg<<"for the momentum:\n";
  for(unsigned int i=0; i<self_momentum_point.size(); i++)
    self_momentum_point[i].print();

  //check the output_Propagator type
  NemoPhys::Propagator_type result_type=get_Propagator_type(output_Propagator->get_name());
  //check the needed input is given
  NemoPhys::Propagator_type input1_type=get_Propagator_type(input_Propagator1->get_name());
  if(result_type==NemoPhys::Fermion_retarded_self)
  {
    NEMO_ASSERT(input_Propagator2!=NULL,prefix+"second Propagator is missing\n");
    NemoPhys::Propagator_type input2_type=get_Propagator_type(input_Propagator2->get_name());
    NEMO_ASSERT(input1_type==NemoPhys::Fermion_retarded_Green&&input2_type==NemoPhys::Fermion_lesser_Green,
        prefix+"incorrect type of given input Propagators\n");
  }
  else if(result_type==NemoPhys::Fermion_lesser_self)
  {
    NEMO_ASSERT(input1_type==NemoPhys::Fermion_lesser_Green,prefix+"incorrect type of given input Propagator\n");
  }

  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //0. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {
    if(relevant_local_momenta->size()==0)
    {
      Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
      NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
    }
    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }
  else
  {
    Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
    NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
  }

  //1. Get the input matrice(s)
  PetscMatrixParallelComplex* Green1=NULL;
  PetscMatrixParallelComplex* Green2=NULL;
  input_Propagator1_solver->get_data(input_Propagator1->get_name(),&momentum_point,Green1);

  if(Green1->if_container() )
    Green1->assemble();
  if(input_Propagator1!=input_Propagator2)
  {
    input_Propagator2_solver->get_data(input_Propagator2->get_name(),&momentum_point,Green2);
    if(Green2->if_container())
      Green2->assemble();
  }
  int start_local_row;
  int end_local_row;
  Green1->get_ownership_range(start_local_row,end_local_row);
  //2. define which section of the input Green's function to include for the scattering self-energy (usually diagonal)
  //NEMO_ASSERT(store_offdiagonals<=(end_local_row/2-1),prefix+" error input store_offdiagonals > (number of rows)/2-1\n");
  /*int col_start = 0;
  int col_end = 0;
  if(set_of_row_col_indices.size()==0)
  {

    for(int i=start_local_row;i<end_local_row;i++)
    {
      col_start = i - store_offdiagonals;
      if(col_start<0)
      {
        col_start=0;
      }
      col_end = i + store_offdiagonals+1;
      if(col_end>end_local_row)
      {
        col_end=end_local_row;
      }
      for(int j=col_start;j<col_end;j++)
      {
        set_of_row_col_indices.insert(pair<int, int>(i,j));
      }
    }
  }*/

  //2. get the distance map for setting up the sparsity pattern
    EOMMatrixInterface* eom_source=dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
    NEMO_ASSERT(eom_source!=NULL,prefix+Hamilton_Constructor->get_name()+"is not an EOMMatrixInterface \n");

    std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, vector<double> > > distance_vector_map;
    eom_source->get_distance_vector_map(distance_vector_map,NULL,NULL);
    NEMO_ASSERT(!distance_vector_map.empty(), prefix + " distance map is empty. Did you set the interaction radius for the given material? \n");

   if(debug_output)
    {
      ofstream outfile;
      std::string filename = get_name() + "_distance_vector_map.dat";
      outfile.open(filename.c_str());
      outfile << "row \t column \t distance\n ";


      std::map<std::pair<unsigned int, unsigned int>,std::pair<unsigned int, vector<double> > >::iterator it = distance_vector_map.begin();
      for(; it != distance_vector_map.end(); ++it)
      {
        //std::pair<unsigned int, unsigned int> temp_pair = it->first;
        outfile << it->first.first << "\t" << it->first.second << "\t" ;
        for(unsigned int i =0; i <it->second.second.size(); i++)
          outfile << it->second.second[i] << "\t";

        outfile << "\n";
      }

      outfile.close();
    }
       //4. set up sparsity pattern from this map. This distance map does not include same atom (diagonal contribution)
      if(set_of_row_col_indices.empty())
      {
        std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, vector<double> > >::iterator it = distance_vector_map.begin();
        for(; it != distance_vector_map.end(); ++it)
          set_of_row_col_indices.insert(it->first);
      }
   bool   is_diagonal_only=false;
  //3. integrate as need over the Green's functions(covers the MPI_Reduce command) on matrices in 1.
  //3.1 (optional) determine whether the integrate_submatrix method needs to be called, i.e. whether there is a real integral to be done (i.e. not if there is no periodicity)
  //3.2 prepare the list of meshes to be excluded from integration (in particular energy, since this is scattering with discrete energies)
  std::set<std::string> exclude_integration;
  //find the energy mesh name
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
  std::string energy_name=std::string("");
  for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
    if(momentum_name_it->second==NemoPhys::Energy)
      energy_name=momentum_name_it->first;
  exclude_integration.insert(energy_name);

  //3.3 create the prefactor according to emission or absorption
  std::map<std::vector<NemoMeshPoint>, double> weight_function;
  //std::map<std::vector<NemoMeshPoint>, double>* pointer_to_weight_function=NULL;
  if(relevant_local_momenta!=NULL)
  {
    //pointer_to_weight_function = &weight_function;
    std::set<std::vector<NemoMeshPoint> >::const_iterator it_final_momenta = relevant_local_momenta->begin();
    for(; it_final_momenta != relevant_local_momenta->end(); ++it_final_momenta)
    {

      double final_weight=0;
      double self_energy_weight=0;

      // remove unused variables:
      //double final_momentum_energy = PropagationUtilities::read_energy_from_momentum(this,(*it_final_momenta),output_Propagator);
      //double self_momentum_point_energy = PropagationUtilities::read_energy_from_momentum(this,self_momentum_point,output_Propagator);
      //double final_k = read_kvector_from_momentum((*it_final_momenta),output_Propagator)[2];
      //double self_k = read_kvector_from_momentum(self_momentum_point,output_Propagator)[2];

      for(unsigned int i=0; i<(*it_final_momenta).size(); i++)
      {
        std::string momentum_mesh_name=output_Propagator->momentum_mesh_names[i];
        std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
        NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
        Simulation* mesh_constructor=temp_cit->second;

        InputOptions& mesh_options=mesh_constructor->get_reference_to_options();

        if(output_Propagator->momentum_mesh_names[i].find("energy")!=std::string::npos) // if momentum_mesh_names[i]= energy
        {
          if(!mesh_options.get_option(std::string("non_rectangular"),false))
          {
            //get local momentum integration weight
            mesh_constructor->get_data("integration_weight",(*it_final_momenta)[i],final_weight);  //Aryan what are the weight? k-vector weight?
            //get self momentum integration weight
            mesh_constructor->get_data("integration_weight",(self_momentum_point)[i],self_energy_weight); //Aryan what are the weight? k-vector weight?

          }
          else
          {
            //if integration weight for energy
            //if(output_Propagator->momentum_mesh_names[i].find("energy")!=std::string::npos)
            {
              //get the k-point from the momentum for local momentum
              std::vector<double> temp_vector=read_kvector_from_momentum(*it_final_momenta, output_Propagator);
              NemoMeshPoint temp_momentum(0,temp_vector);
              std::vector<NemoMeshPoint> temp_vector_momentum(1,temp_momentum);
              //mesh_constructor->get_data("integration_weight",only-k (1dvector<NemoMeshPoint>,only energy NemoMeshPoint,temp_double);
              mesh_constructor->get_data("integration_weight",temp_vector_momentum,(*it_final_momenta)[i],final_weight);

              //get the k-point from the momentum for self energy momentum
              temp_vector=read_kvector_from_momentum(self_momentum_point, output_Propagator);
              NemoMeshPoint temp_momentum2(0,temp_vector);
              std::vector<NemoMeshPoint> temp_vector_momentum2(1,temp_momentum2);
              //mesh_constructor->get_data("integration_weight",only-k (1dvector<NemoMeshPoint>,only energy NemoMeshPoint,temp_double);
              mesh_constructor->get_data("integration_weight",temp_vector_momentum2,(self_momentum_point)[i],self_energy_weight);

            }
          }


          double emiss_norm = 0.0;
          double absorp_norm = 0.0;

          std::set<unsigned int> Hamilton_momentum_indices;
          std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
          PropagationUtilities::find_Hamiltonian_momenta(this,writeable_Propagator,pointer_to_Hamilton_momentum_indices);

          //bool this_is_for_absorption=true; //Aryan todo

          if(this_is_for_absorption)
          {
            std::map<vector<NemoMeshPoint>,map<NemoMeshPoint,double> >::iterator it = emission_norm_factor.find(*it_final_momenta); //
            NEMO_ASSERT(it != emission_norm_factor.end(),prefix+"could not find emission norm factor for absorption");

            NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=(*it_final_momenta)[*(pointer_to_Hamilton_momentum_indices->begin())];
            NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(self_momentum_point)[*(pointer_to_Hamilton_momentum_indices->begin())];
            std::map<NemoMeshPoint,double>  temp = it->second;  // map of all absorption temp_NemoMeshPoint2 (self_mom point) norm factor for this final momenta
            std::map<NemoMeshPoint,double>::iterator map_it = temp.find(temp_NemoMeshPoint2);

            if(map_it!=temp.end())
              emiss_norm = map_it->second; // emission_norm_factor(point_final,self_point)
            else
              emiss_norm = self_energy_weight;

            it = absorption_norm_factor.find(self_momentum_point);
            NEMO_ASSERT(it != absorption_norm_factor.end(),prefix+"could not find absorption norm factor");

            //NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
            // if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(self_momentum_point)[*(pointer_to_Hamilton_momentum_indices->begin())];
            temp = it->second;
            map_it = temp.find(temp_NemoMeshPoint);
            if(map_it!=temp.end())
              absorp_norm = map_it->second;
            else
              absorp_norm =final_weight;
          }
          else
          {
            std::map<vector<NemoMeshPoint>,map<NemoMeshPoint,double> >::iterator it = emission_norm_factor.find(self_momentum_point);
            NEMO_ASSERT(it != emission_norm_factor.end(),prefix+"could not find emission norm factor");

            //const std::vector<NemoMeshPoint> temp_vector_pointer=it_comm_pair->first;

            NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=(self_momentum_point)[*(pointer_to_Hamilton_momentum_indices->begin())];
            NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(*it_final_momenta)[*(pointer_to_Hamilton_momentum_indices->begin())];
            std::map<NemoMeshPoint,double>  temp = it->second;
            std::map<NemoMeshPoint,double>::iterator map_it = temp.find(temp_NemoMeshPoint2);
            if(map_it!=temp.end())
              emiss_norm = map_it->second;
            else
              emiss_norm = final_weight;


            it = absorption_norm_factor.find(*it_final_momenta);
            NEMO_ASSERT(it != absorption_norm_factor.end(),prefix+"could not find absorption norm factor");

            //NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
            //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(*it_final_momenta)[*(pointer_to_Hamilton_momentum_indices->begin())];
            temp = it->second;
            map_it = temp.find(temp_NemoMeshPoint);
            if(map_it!=temp.end())
              absorp_norm = map_it->second;
            else
              absorp_norm = self_energy_weight;


          }
          NEMO_ASSERT(emiss_norm>0.0&&absorp_norm>0.0,prefix+"emission norm factor or absorption norm factor returned 0");
          double  ratio = 2.0*final_weight/(emiss_norm+absorp_norm);
          if(*it_final_momenta==self_momentum_point)
            ratio = 0.0;
          //if(ratio>2.0)
          //NEMO_ASSERT(ratio<=2.5,prefix + "absorption " + NemoUtils::nemo_to_string(this_is_for_absorption) + " ratio is too large " + NemoUtils::nemo_to_string(ratio) + " " + NemoUtils::nemo_to_string(emiss_norm) + " "
          //    + NemoUtils::nemo_to_string(absorp_norm) + "  " + NemoUtils::nemo_to_string(final_weight)+ " "
          //    + NemoUtils::nemo_to_string(final_momentum_energy) + " " + NemoUtils::nemo_to_string(self_momentum_point_energy));

          //cerr <<  prefix + "absorption " << this_is_for_absorption << " ratio is too large " + NemoUtils::nemo_to_string(ratio) + " " + NemoUtils::nemo_to_string(emiss_norm) + " "
          //  + NemoUtils::nemo_to_string(absorp_norm) + "  " + NemoUtils::nemo_to_string(final_weight)+ " "
          //   + NemoUtils::nemo_to_string(final_momentum_energy) + " " + NemoUtils::nemo_to_string(self_momentum_point_energy) << " \n";
          // ratio = 2.0*final_weight/(final_weight+self_energy_weight);

          weight_function[*it_final_momenta] = ratio;
        }
      }
    }
  }
  //3.4 call the integrate submatrix
  //bool is_diagonal_only=true;
  //int off_diagonals = options.get_option("store_offdiagonals",0);
  //if(off_diagonals>0)
  //  is_diagonal_only=false;
  std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*> local_weight_function_map;
  std::pair<int,int> local_index_pair(0,0);
  local_weight_function_map[local_index_pair]=&weight_function;
  const std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*>* pointer_to_weight_function=&local_weight_function_map;


  NemoPhys::Propagator_type propagator_type = get_Propagator_type(input_Propagator1->get_name());
  bool prefactor_flag = false;
  const std::vector<NemoMeshPoint> * self_momentum_point_ptr= NULL;
  const std::vector<vector<double> >reciprocal_basis = get_simulation_domain()->get_reciprocal_vectors();
  prefactor_flag = true;
  self_momentum_point_ptr = &self_momentum_point;
  //int rec_basis_size = reciprocal_basis.size();
  //if((rec_basis_size == 2)||(rec_basis_size == 0))
  //{
  //  self_momentum_point_ptr = &self_momentum_point;
  //  prefactor_flag = true;
  //}

  PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices, relevant_local_momenta, input_Propagator1_solver, propagator_type, MPI_communicator, root_address, group, pointer_to_result_vector, exclude_integration, is_diagonal_only, pointer_to_weight_function, NULL, false, prefactor_flag, self_momentum_point_ptr); //for n-dim periodicity: [eV/eV/nm^n]

  if (result_type == NemoPhys::Fermion_retarded_self)
  {
    NemoPhys::Propagator_type propagator_type = get_Propagator_type(input_Propagator2->get_name());
    PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices, relevant_local_momenta, input_Propagator2_solver, propagator_type, MPI_communicator, root_address, group, pointer_to_result_vector2, exclude_integration, is_diagonal_only, pointer_to_weight_function, NULL, false, prefactor_flag, self_momentum_point_ptr); //for n-dim periodicity: [eV/eV/nm^n]
  }

  NemoUtils::toc(tic_toc_prefix);

}

double Self_energy::utb_pop_mom_integral(unsigned int integration_points, double lattice_const, double r, double screening_length, std::vector<double> momentum_point1, std::vector<double> momentum_point2)
{
	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::theta_integral ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(\"" + this->get_name() + "\")::theta_integral() ";

	std::vector<double> q(integration_points, 0.0);
	std::vector<double> q_weights(integration_points, 0.0);

	double q_min = 0.0;
	double q_max = NemoMath::pi/lattice_const;

	for (unsigned int i = 0;i < q.size(); i++)
	{
		q[i] = q_min + (q_max - q_min)*i / (integration_points - 1);
	}

	for (unsigned int i = 0;i < q.size();i++)
	{
		if (i == 0)
			q_weights[i] = (q[i + 1] - q[i]) / 2;
		else
			if (i == q.size() - 1)
				q_weights[i] = (q[i] - q[i - 1]) / 2;
			else
				q_weights[i] = q[i] - q[i - 1];
	}

	double zeta = 1 / screening_length;
	double kinitial = 0.0;
	double kfinal = 0.0;

	for (unsigned int i = 0;i < momentum_point1.size();i++)
	{
		kinitial += std::pow(momentum_point1[i], 2);
		kfinal += std::pow(momentum_point2[i], 2);
	}

	kinitial = std::pow(kinitial, 0.5);
	kfinal = std::pow(kfinal, 0.5);

	double kminusl = kinitial - kfinal;
	double integrand = 0.0;
	double integral = 0.0;

	for (unsigned int i = 0;i < integration_points;i++)
	{
		integrand = q[i] * (q[i] * q[i] + kminusl*kminusl) / (std::pow((q[i] * q[i] + kminusl*kminusl + zeta*zeta), 2.0))*boost::math::cyl_bessel_j(0, q[i]*r);
		//integrand = q[i] * (q[i] * q[i] + kminusl*kminusl) / (std::pow((q[i] * q[i] + kminusl*kminusl + zeta*zeta), 2.0))*NemoMath::bessel_j(0, q[i] * r);
		integral += integrand*q_weights[i];
	}

	NemoUtils::toc(tic_toc_prefix);
	return integral;
}


double Self_energy::theta_integral(unsigned int integration_points, double r, double screening_length, std::vector<double> momentum_point1, std::vector<double> momentum_point2)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::theta_integral ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(\"" + this->get_name() + "\")::theta_integral() ";

	std::vector<double> theta(integration_points,0.0);
	std::vector<double> theta_weights(integration_points,0.0);

	double theta_min = 0.0;
	double theta_max = NemoMath::pi;

	for (unsigned int i = 0;i < theta.size(); i++)
	{
		theta[i] = theta_min + (theta_max - theta_min)*i / (integration_points-1);
	}

	for (unsigned int i = 0;i < theta.size();i++)
	{
		if (i == 0)
			theta_weights[i] = (theta[i + 1] - theta[i]) / 2;
		else
			if (i == theta.size() - 1)
				theta_weights[i] = (theta[i] - theta[i - 1]) / 2;
			else
				theta_weights[i] = theta[i] - theta[i - 1];
	}

	double zeta = 1 / screening_length;
	double kinitial = 0.0;
	double kfinal = 0.0;

	for (unsigned int i = 0;i < momentum_point1.size();i++)
	{
		kinitial += std::pow(momentum_point1[i],2);
		kfinal += std::pow(momentum_point2[i],2);
	}
    
	
    kinitial = std::pow(kinitial,0.5);
    kfinal = std::pow(kfinal,0.5);
    double alpha = 0.0;
	double integrand = 0.0;
	double integral = 0.0;

	for (unsigned int i = 0;i < integration_points;i++)
	{
		alpha = std::pow(kinitial, 2) + std::pow(kfinal, 2) - 2.0 * kinitial*kfinal*cos(theta[i]) + std::pow(zeta,2);
		alpha = std::pow(alpha, 0.5);
		integrand = (std::exp(-alpha*r) / alpha)*(1 - (zeta*zeta*r) / (2.0 * alpha) - (zeta*zeta) / (2.0 * alpha*alpha));
		integral += 2.0*integrand*theta_weights[i];
	}
	
	NemoUtils::toc(tic_toc_prefix);
	return integral;
}

/*double Self_energy::theta_integral(double r, double screening_length, std::vector<double> momentum_point1,std::vector<double> momentum_point2)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::polar_optical_prefactor ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::polar_optical_prefactor() ";
  double r2=std::abs(r);
  double q0=1/screening_length  ;
  double erg=0;
  int N_int=5;
  double a2=0;
  double b2=0;
  for(unsigned int i = 0; i<momentum_point1.size();i++)
  {
	a2 += std::pow(momentum_point1[i],2);
	b2 += std::pow(momentum_point2[i],2);
  }
  a2=std::pow(a2,0.5);
  b2=std::pow(b2,0.5);

  int ii=1;
  double h=NemoMath::pi/double(3)/double(N_int);
  double theta=0;
  double term1=0;
  double rterm1=0;
  double term2=0;
  double term3=0;
  for (;ii<N_int;ii++)
  {
	  theta=(3*ii-2)*h;
	  term1=std::pow(a2,2)+std::pow(b2,2)-double(2)*a2*b2*cos(theta)+std::pow(q0,2);
	  rterm1=std::pow(term1,double(.5));
	  term2=std::exp(-rterm1*r2);
	  term3=1-std::pow(q0,2)*r2/double(2)/rterm1-std::pow(q0,2)/double(2)/term1;
	  if(term2==double(1))
	  {
	    term2=term2*(9.5504e-1+4.5532e-1*rterm1-2.630677e1*term1+1.9867451e2*rterm1*term1-5.4926134e2*term1*term1);
	  }
	  erg=erg+3*(term2/rterm1*term3);
	  theta=(3*ii-1)*h;
	  term1=std::pow(a2,2)+std::pow(b2,2)-double(2)*a2*b2*cos(theta)+std::pow(q0,2);
	  rterm1=std::pow(term1,double(.5));
	  term2=std::exp(-rterm1*r2);
	  term3=1-std::pow(q0,2)*r2/double(2)/rterm1-std::pow(q0,2)/double(2)/term1;
	  if(term2==double(1))
	  {
        term2=term2*(9.5504e-1+4.5532-1*rterm1-2.630677e1*term1+1.9867451e2*rterm1*term1-5.4926134e2*term1*term1);
	  }
      erg=erg+3*(term2/rterm1*term3);
	  theta=3*ii*h;
      term1=std::pow(a2,2)+std::pow(b2,2)-double(2)*a2*b2*cos(theta)+std::pow(q0,2);
	  rterm1=std::pow(term1,double(.5));
	  term2=std::exp(-rterm1*r2);
	  term3=1-std::pow(q0,2)*r2/double(2)/rterm1-std::pow(q0,2)/double(2)/term1;
	  if(term2==double(1))
	  {
        term2=term2*(9.5504e-1+4.5532e-1*rterm1-2.630677e1*term1+1.9867451e2*rterm1*term1-5.4926134e2*term1*term1);
	  }
	  erg=erg+2*(term2/rterm1*term3);

  }
  ii=N_int;
  theta=(3*ii-2)*h;
  term1=std::pow(a2,2)+std::pow(b2,2)-double(2)*a2*b2*cos(theta)+std::pow(q0,2);
  rterm1=std::pow(term1,double(.5));
  term2=std::exp(-rterm1*r2);
  term3=1-std::pow(q0,2)*r2/double(2)/rterm1-std::pow(q0,2)/double(2)/term1;
  if(term2==double(1))
  {
    term2=term2*(9.5504e-1+4.5532e-1*rterm1-2.630677e1*term1+1.9867451e2*rterm1*term1-5.4926134e2*term1*term1);
  }
  erg=erg+double(3)*(term2/rterm1*term3);
  theta=(3*ii-1)*h;
  term1=std::pow(a2,2)+std::pow(b2,2)-double(2)*a2*b2*cos(theta)+std::pow(q0,2);
  rterm1=std::pow(term1,double(.5));
  term2=std::exp(-rterm1*r2);
  term3=1-std::pow(q0,2)*r2/double(2)/rterm1-std::pow(q0,2)/double(2)/term1;
  if(term2==double(1))
  {
	term2=term2*(9.5504e-1+4.5532e-1*rterm1-2.630677e1*term1+1.9867451e2*rterm1*term1-5.4926134e2*term1*term1);
	erg=erg+3*(term2/rterm1*term3);
  }
  theta=0;
  term1=std::pow(a2,2)+std::pow(b2,2)-double(2)*a2*b2*cos(theta)+std::pow(q0,2);
  rterm1=std::pow(term1,double(.5));
  term2=std::exp(-rterm1*r2);
  term3=1-std::pow(q0,2)*r2/double(2)/rterm1-std::pow(q0,2)/double(2)/term1;
  if(term1==1)
  {
   term2=term2*(9.5504e-1+4.5532e-1*rterm1-2.630677e1*term1+1.9867451e2*rterm1*term1-5.4926134e2*term1*term1);
  }
  erg=erg+(term2/rterm1*term3);

  theta=NemoMath::pi;
  term1=std::pow(a2,2)+std::pow(b2,2)-double(2)*a2*b2*cos(theta)+std::pow(q0,2);
  rterm1=std::pow(term1,double(.5));
  term2=std::exp(-rterm1*r2);
  term3=1-std::pow(q0,2)*r2/double(2)/rterm1-std::pow(q0,2)/double(2)/term1;
  if(term1==1)
  {
    term2=term2*(9.5504e-1+4.5532e-1*rterm1-2.630677e1*term1+1.9867451e2*rterm1*term1-5.4926134e2*term1*term1);
  }
  erg=erg+(term2/rterm1*term3);

  erg=erg*double(3)/double(8)*h*2;

  return erg;
}*/

double Self_energy::polar_optical_prefactor(const std::vector<NemoMeshPoint>& momentum1, const std::vector<NemoMeshPoint>& momentum2, unsigned int x1, unsigned int x2)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::polar_optical_prefactor ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::polar_optical_prefactor() ";

  //1 get the indices of the matrix
  NEMO_ASSERT(set_of_row_col_indices.size()!=0 , prefix+"Size of set_of_row_col_indices is zero \n");


  //2. get the position of each atom and build map
  static std::map<int,vector<double> > atom_row_index_position_map;
  double lattice_const = 0.0; //[nm]
  if(atom_row_index_position_map.size()==0)
  {
    //2.1 loop over all atoms
    const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (this->get_simulation_domain());
    const AtomicStructure&  atoms  = domain->get_atoms();
    ConstActiveAtomIterator it     = atoms.active_atoms_begin();
    ConstActiveAtomIterator end    = atoms.active_atoms_end();
    const DOFmapInterface&           dof_map = Hamilton_Constructor->get_dof_map(get_simulation_domain());
        //HamiltonConstructor* tb_ham = NULL;
	std::vector<std::string> lattice_str(1, "Lattice");

    for(; it!=end; ++it)
    {
      const AtomStructNode& nd        = it.node();
      const Atom*           atom      = nd.atom;
      const Material*       material  = atom->get_material();
	  MaterialProperties* material_properties = Hamilton_Constructor->get_material_properties(material);
	  std::string atom_tag = material->get_tag();
	  lattice_const = material_properties->query_database_for_material(material->get_name(), lattice_str, "a_lattice", atom_tag);
      /*const Crystal *       crystal   = material->get_crystal(); */
	    if(lattice_const == 0.0)
	    {
		    lattice_const = material_properties->query_database_for_material(material->get_name(), lattice_str, "a_lattice", atom_tag);
	    }
	    /*const Crystal *       crystal   = material->get_crystal(); */      
      HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (Hamilton_Constructor->get_material_properties(atom->get_material()));
      if(tb_ham->get_number_of_orbitals(atom)>1)
        atomic_output_only=true;
      const unsigned int    atom_id   = it.id();

      //position of the first atom
      std::vector<double> position(3,0.0);
      for(unsigned int i=0; i<3; i++)
        position[i]=nd.position[i];

      //2.2 loop over all orbitals
      const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
      map<short, unsigned int>::const_iterator c_it=atom_dofs->begin();
      for(; c_it!=atom_dofs->end(); c_it++)
      {
        atom_row_index_position_map.insert(pair< const unsigned int, std::vector<double> >(c_it->second , position));
      }
    }
  }


  //3. calculate real space distance
  std::vector<double> position1(3,0.0);
  position1=atom_row_index_position_map[x1];
  std::vector<double> position2(3,0.0);
  position2=atom_row_index_position_map[x2];
  double distance = double(0.0);
  if(x1!=x2)
  {
    distance= std::pow(std::pow(position1[0]-position2[0],2)+std::pow(position1[1]-position2[1],2)+std::pow(position1[2]-position2[2],2),0.5);
  }

  //5. get user input for prefactor
  double phonon_energy = options.get_option("optical_phonon_energy", 0.0); //[eV]
  double eps_r_0 = options.get_option("static_dielectric_constant",1.0); //[relative unit]
  double eps_r_inf = options.get_option("high_freq_dielectric_constant",1.0); //[relative unit]
  double screening_length = options.get_option("Debye_screening_length",1.0); //[nm]
  lattice_const = options.get_option("lattice_const",1.0); //[nm]
  double epsilon_0 = NemoPhys::epsilon;// [F/m] = [c^2/(J��m)]
  epsilon_0 = epsilon_0*NemoMath::nm_in_m*NemoPhys::elementary_charge;// [c^2/(ev��nm)]
  double gamma=1/eps_r_inf-1/eps_r_0 ; //[relative unit]
  gamma *= (NemoPhys::elementary_charge*NemoPhys::elementary_charge);//[c^2]
  gamma *= phonon_energy/(2.0*epsilon_0);// [ev^2��nm]

  const std::vector<vector<double> >reciprocal_basis = get_simulation_domain()->get_reciprocal_vectors();
  int rec_basis_size = reciprocal_basis.size();
  double prefactor = 0.0;
  std::vector<double> momentum_point1;
  std::vector<double> momentum_point2;
  if (rec_basis_size == 2)
  {
	bool rotational_symmetry = options.get_option("scattering_rate_rotational_symmetry",bool(false));
    //4. calculate the in plan phonon momentum
    double phonon_in_plane_momentum = 0.0;
    //check if there a single k point
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
    bool momentum_flag = false;
    for (; momentum_name_it!=momentum_mesh_types.end(); ++momentum_name_it)
       if(momentum_name_it->second==NemoPhys::Momentum_2D||momentum_name_it->second==NemoPhys::Momentum_1D||momentum_name_it->second==NemoPhys::Momentum_3D)
         momentum_flag = true ;

    if(momentum_flag==true)
    {
      momentum_point1=read_kvector_from_momentum(momentum1, writeable_Propagator);
      momentum_point2=read_kvector_from_momentum(momentum2, writeable_Propagator);

      for(unsigned int i = 0; i<momentum_point1.size();i++)
      {
        phonon_in_plane_momentum += std::pow(momentum_point1[i]-momentum_point2[i],2);
      }
    }
    //prefactot units = ev^2.*nm^2
    //after multiply by dK|| [1/nm^2] prefactor is ev^2
    //double momentum_screening = phonon_in_plane_momentum+double(1)/(screening_length*screening_length);//[1/nm/nm]
    if(rotational_symmetry==true)
    {
    	//If rotational symmetry is true and if device is quasi1D then we need to 
        //divide by 2*pi. This is needed as momentum integration weight already 
        //has a 2*pi in it whereas in POP this theta integral is done explicitly.
        prefactor = NemoMath::pi/std::pow(2*NemoMath::pi,3)*gamma*4.0/(2.0*NemoMath::pi);//[ev^2.nm]
    }
    else
    {
    	prefactor = NemoMath::pi/std::pow(2*NemoMath::pi,3)*gamma;//[ev^2.nm]
    }
    double  prefactor2=theta_integral( 20, distance,  screening_length, momentum_point1, momentum_point2);
	prefactor = prefactor*prefactor2;
   // {
    //double momentum_screening_root = std::pow(momentum_screening,.5);//[1/nm]
    //prefactor *= std::exp(double(-1)*momentum_screening_root*distance)/momentum_screening_root;//[ev^2.nm].[nm]=[ev^2.nm^2]
    //prefactor *= (double(1)-double(.5)*distance/screening_length/screening_length/momentum_screening_root-double(.5)/screening_length/screening_length/momentum_screening);
    //double temp1=distance/2/screening_length/screening_length/momentum_screening_root;
    //double temp2=1/2/screening_length/screening_length/momentum_screening;
   // std::cerr<<"Q1D polar optical\n";
  }
  else if((rec_basis_size == 0))
  {
    //prefactot units = ev^2
    if(x1==x2)
    {
      double prefactor_archtan = double(4)*NemoMath::pi* ( double(3)/ double(2)*NemoMath::pi/lattice_const-
            double(3)/double(2)/screening_length*NemoMath::atan(NemoMath::pi*screening_length/lattice_const)-
			double(1)/double(2)/lattice_const*(screening_length*screening_length*NemoMath::pi*NemoMath::pi*NemoMath::pi)/(lattice_const*lattice_const+NemoMath::pi*NemoMath::pi*screening_length*screening_length));//[1/nm]
        prefactor = double(1)/std::pow(2*NemoMath::pi,3)*gamma*prefactor_archtan; //[ev^2��nm]*[1/nm] =[ev^2]
    }
    else
    {
      prefactor=gamma*exp(double(-1)*distance/screening_length)/double(8)/NemoMath::pi*(double(2)/distance-double(1)/screening_length);//[ev^2]
    }
  }
  else
  {
	  prefactor = gamma / std::pow(2 * NemoMath::pi, 2);
	  double prefactor2 = utb_pop_mom_integral(50, lattice_const, distance, screening_length, momentum_point1, momentum_point2);
	  prefactor = prefactor*prefactor2;
	  //NEMO_ASSERT(false, prefix+" Polar optical is implemented only for wires or nonlocal Q1D");
  }
  //prefactor std::cerr<<"Prefactor "<<prefactor<<"\n";
  //NEMO_ASSERT(prefactor>0,prefix+"The prefactor is negative\n");
 if(debug_output)
  {
	 print_Pop_prefactor_log(momentum_point1, momentum_point2, x1,x2,position1, position2,prefactor);

  }

  NemoUtils::toc(tic_toc_prefix);
  return prefactor;
}


void Self_energy::print_Pop_prefactor_log(std::vector<double> momentum_point1,
		std::vector<double> momentum_point2, unsigned int x1,
			unsigned int x2,std::vector<double> position1,std::vector<double> position2,
			double prefactor)
{
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  if(myrank==0)
  {
	  if (debug_output)
	  {
		  std::string filename = get_name() + "_Pop_prefactor.dat";
		  ofstream  outfile;
		  outfile.open(filename.c_str(), ios_base::out | ios_base::app);
		  //outfile<<"momentum1 \t momentum2 \t index1 \t index2 \t Position1 x y z\t Position2 x y z\n";

		  if (momentum_point1.size() > 2)
		  {
			  outfile << momentum_point1[0] << "\t" << momentum_point1[1] << "\t" << momentum_point1[2] << "\t";
			  outfile << momentum_point2[0] << "\t" << momentum_point2[1] << "\t" << momentum_point2[2] << "\t";
		  }

		  outfile << x1 << "\t" << x2 << "\t";
		  outfile << position1[0] << "\t" << position1[1] << "\t" << position1[2] << "\t";
		  outfile << position2[0] << "\t" << position2[1] << "\t" << position2[2] << "\t" << prefactor << "\n";
		  outfile.close();
	  }

  }



}

void Self_energy::save_and_postprocess_polar_optical_Froehlich_phonon(Propagator*& /*output_Propagator*/, const Propagator*& input_Propagator, const Propagator*& input_Propagator2,
      Simulation* input_Propagator_solver, Simulation* input_Propagator2_solver,
      const std::vector<NemoMeshPoint>& self_momentum_point,
      const bool this_is_for_absorption, const bool /*coupling_to_same_energy*/,
      const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
      const std::set<vector<NemoMeshPoint> >* /*final_coupling_energies*/,
      std::vector<std::complex<double> >*& pointer_to_result_vector,
      std::vector<std::complex<double> >*& pointer_to_result_vector2,
      PetscMatrixParallelComplex*& result)
{

  std::string prefix="Self_energy("+this->get_name()+")::save_and_postprocess_polar_optical_phonon: ";
  //check the output_Propagator type
  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //0. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {
    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }
  //1. Get the input matrice(s)
  PetscMatrixParallelComplex* Green1=NULL;
  PetscMatrixParallelComplex* Green2=NULL;
  input_Propagator_solver->get_data(input_Propagator->get_name(),&momentum_point,Green1);
  if(input_Propagator!=input_Propagator2)
  {
    input_Propagator2_solver->get_data(input_Propagator2->get_name(),&momentum_point,Green2);

  }

  //4. postprocessing if this MPI-rank is the root_rank (i.e. stores the actual self-energy)
  //4.1 determine whether the energy of self_momentum_point is larger or smaller than the energy of relevant_local_momenta
  //double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_momentum_point, output_Propagator);
  //NEMO_ASSERT(relevant_local_momenta!=NULL,prefix+"received empty list of local momenta\n");

  //double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,*(relevant_local_momenta->begin()), output_Propagator);
  //bool this_is_for_absorption=true;
  //if(self_energy>representative_coupling_energy)
  //  this_is_for_absorption=false;

  //double temperature_in_eV=options.get_option("temperature",NemoPhys::temperature)*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  double phonon_energy = options.get_option("optical_phonon_energy", 0.0); //[eV]
  double phonon_number = NemoMath::bose_distribution(0.0, temperature_in_eV, phonon_energy);
  //double eps_r_0 = options.get_option("static_dielectric_constant",1.0); //[relative unit]
  //double eps_r_inf = options.get_option("high_freq_dielectric_constant",1.0); //[relative unit]
  //double screening_length = options.get_option("Debye_screening_length",1.0); //[nm]
  const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*> (this->get_simulation_domain());
  const AtomicStructure&  atoms = domain->get_atoms();
  ConstActiveAtomIterator it = atoms.active_atoms_begin();
  ConstActiveAtomIterator end = atoms.active_atoms_end();
  //const DOFmapInterface&           dof_map = Hamilton_Constructor->get_dof_map();
  //HamiltonConstructor* tb_ham = NULL;
  std::vector<std::string> lattice_str(1, "Lattice");
  double lattice_const = 0.0;
  for (; it != end; ++it)
  {
	  const AtomStructNode& nd = it.node();
	  const Atom*           atom = nd.atom;
	  const Material*       material = atom->get_material();
	  MaterialProperties* material_properties = Hamilton_Constructor->get_material_properties(material);
	  std::string atom_tag = material->get_tag();
	  if (lattice_const == 0.0)
	  {
	  lattice_const = material_properties->query_database_for_material(material->get_name(), lattice_str, "a_lattice", atom_tag);
  }
	  
  }
  //double lattice_const = options.get_option("lattice_const",1.0); //[nm] check_pop how to get it from the DB
  //double epsilon_0 = 4*NemoMath::pi*NemoPhys::epsilon;// [F/m]
  //// F/m = c^2/J/m convert to c^2/ev/nm
  //epsilon_0 = epsilon_0*NemoMath::nm_in_m*NemoPhys::elementary_charge;// [c^2/ev/nm]
  //double prefactor=1/eps_r_inf-1/eps_r_0 ; //[relative unit]
  //prefactor *= (NemoPhys::elementary_charge*NemoPhys::elementary_charge)/(std::pow(2*NemoMath::pi,3));//[c^2]
  //prefactor *= phonon_energy/(2.0*epsilon_0);//[c^2]*[ev]/[c^2/ev/nm] = [ev^2.nm]
  //double prefactor_archtan = 4*NemoMath::pi* ( NemoMath::pi/lattice_const-
  //    3/2/screening_length*NemoMath::atan(NemoMath::pi*screening_length/lattice_const)+
  //    lattice_const*NemoMath::pi/
  //    2/(screening_length*screening_length*NemoMath::pi*NemoMath::pi+lattice_const*lattice_const));//[1/nm]
  //prefactor *= prefactor_archtan; //[ev^2.nm]*[1/nm] =[ev^2]
  ////take care of 1/(2*pi)^d factor from Fourier Transform
  //const std::vector<vector<double> >reciprocal_basis = get_simulation_domain()->get_reciprocal_vectors();

  //if(rec_basis_size == 3) // 3D, no periodicity, e.g. nanowire case
  //prefactor*= -3*std::pow(2*NemoMath::pi,2)/ screening_length ; // factor 3 is for local point x3=x4 //todo implement a non local function of x3-x4

  double prefactor = 1;

  /*if (rec_basis_size == 2)//check_pop: I think it have to be same as optical deformation check_pop
  {
    prefactor = 1;
  }
  else if((rec_basis_size == 0))
  {
    prefactor = 1;
  }
  else
  {
    NEMO_ASSERT(false, prefix+" Polar optical is implemented only for wires or nonlocal Q1D");
  }*/

  /* else
     {
        NemoUtils::MsgLevel prev_level = msg.get_level();
        NemoUtils::msg.set_level(NemoUtils::MsgLevel(1));
        msg << "[polar phonon]: Warning polar phonon not yet implemented for reciprocal basis of size " << rec_basis_size << "\n";
        NemoUtils::msg.set_level(prev_level);
     }*/


  //if(rec_basis_size > 0)
  //prefactor /= std::pow(2*NemoMath::pi, rec_basis_size);


  //if this is a retarded self-energy
  if(pointer_to_result_vector2->size()>0)
  {
    //    if(coupling_to_same_energy)
    //    {
    //      NEMO_ASSERT(pointer_to_result_vector2->size()==pointer_to_result_vector->size(),prefix+"inconsistent result vector sizes\n");
    //      for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
    //      {
    //        (*pointer_to_result_vector)[i] = cplx(0.0,0.0);//*=std::complex<double>(((0.5+phonon_number)*prefactor),0.0)*cplx(0.0,0.0);
    //      }
    //    }
    //    else
    {
      if(this_is_for_absorption)
      {
        NEMO_ASSERT(pointer_to_result_vector2->size()==pointer_to_result_vector->size(),prefix+"inconsistent result vector sizes\n");
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
        {
          (*pointer_to_result_vector)[i]*=std::complex<double>(phonon_number*prefactor,0.0); //Sigmar= prefactor*no*Gr(E+Eo)
          (*pointer_to_result_vector)[i]-=((*pointer_to_result_vector2)[i]*prefactor*0.5); //-0.5*prefactor*G<(E+Eo)
        }
      }
      else
      {
        NEMO_ASSERT(pointer_to_result_vector2->size()==pointer_to_result_vector->size(),prefix+"inconsistent result vector sizes\n");
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
        {
          (*pointer_to_result_vector)[i]*=std::complex<double>((1.0+phonon_number)*prefactor,0.0);
          (*pointer_to_result_vector)[i]+=((*pointer_to_result_vector2)[i]*prefactor*0.5);
        }
      }
    }

  }
  //it this is a lesser self-energy
  else
  {
    //    if(coupling_to_same_energy)
    //    {
    //      for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
    //        (*pointer_to_result_vector)[i]= cplx(0.0,0.0);//*=(0.5+phonon_number)*prefactor*cplx(0.0,0.0);
    //    }
    //    else
    {
      if(this_is_for_absorption)
      {
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
          (*pointer_to_result_vector)[i]*=(1.0+phonon_number)*prefactor;
      }
      else
      {
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
          (*pointer_to_result_vector)[i]*=phonon_number*prefactor;
      }
    }
  }
  int start_local_row;
  int end_local_row;
  Green1->get_ownership_range(start_local_row,end_local_row);


  /*double store_offdiagonals = options.get_option("store_offdiagonals",0);

  store_offdiagonals=2*store_offdiagonals+1;
  if(store_offdiagonals>=end_local_row)
  {
    store_offdiagonals=end_local_row-1;
  }
  //4.2. translate the result_vector into temp_result and add to result
  //4.2.1 set up the temp_result matrix
  PetscMatrixParallelComplex* temp_result = new PetscMatrixParallelComplex(Green1->get_num_rows(),Green1->get_num_cols(),
      get_simulation_domain()->get_communicator());
  //4.2.2 create the sparsity pattern for temp_result
  temp_result->set_num_owned_rows(Green1->get_num_owned_rows());
  for (int i = 0; i < start_local_row; i++)
    temp_result->set_num_nonzeros(i,0,0);
  for (int i = start_local_row; i < end_local_row; i++)
    temp_result->set_num_nonzeros(i,store_offdiagonals,0);
  for (unsigned int i = end_local_row; i < Green1->get_num_rows(); i++)
    temp_result->set_num_nonzeros(i,0,0);
  temp_result->allocate_memory();
  temp_result->set_to_zero();
  */
  //4.1 loop over the set_of_row_col_indices
  std::set<std::pair<int,int> >::const_iterator set_cit=set_of_row_col_indices.begin();
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

  }
  //4.2.1 set up the temp_result matrix
  PetscMatrixParallelComplex* temp_result = new PetscMatrixParallelComplex(Green1->get_num_rows(),Green1->get_num_cols(),
      get_simulation_domain()->get_communicator());
  temp_result->set_num_owned_rows(Green1->get_num_owned_rows());
  for (int i = 0; i < start_local_row; i++)
    temp_result->set_num_nonzeros(i,0,0);
  for (int i = start_local_row; i < end_local_row; i++)
  {
    nonzero_it = nonzero_map.find(i);
    //NEMO_ASSERT(nonzero_it != nonzero_map.end(),prefix + " could not find the number of nonzeros ");
    double num_nonzeros = 0;
    if(nonzero_it != nonzero_map.end())
      num_nonzeros = nonzero_it->second;
    temp_result->set_num_nonzeros(i,num_nonzeros,0);
  }
  for (unsigned int i = end_local_row; i < Green1->get_num_rows(); i++)
    temp_result->set_num_nonzeros(i,0,0);
  temp_result->allocate_memory();
  temp_result->set_to_zero();

  //4.2.3 loop over the set_of_row_col_indices (running index counter is the key of the result_vector)

  unsigned int counter=0;
  set_cit=set_of_row_col_indices.begin();
  for(; set_cit!=set_of_row_col_indices.end(); set_cit++)
  {
    int row_index = (*set_cit).first;
    int col_index = (*set_cit).second;
    temp_result->set(row_index,col_index,(*pointer_to_result_vector)[counter]);
    counter++;
  }
   temp_result->assemble();

  //PetscMatrixParallelComplex* temp_matrix=NULL;
  //Hamilton_Constructor->get_data(std::string("dimensional_S_matrix"),temp_matrix);//for n-dim periodicity: [1/nm^(3-n)]

  //4.2.4.2 perform the multiplication
    //PetscMatrixParallelComplex* pointer_to_temp_result=NULL;
    //PetscMatrixParallelComplex::mult(*temp_matrix,*temp_result,&pointer_to_temp_result);
    //delete temp_result;
    //temp_result=NULL;

    //delete temp_matrix;
    //temp_matrix=NULL;

  //4.2.5 add temp_result to result
  if(result==NULL)
    result=new PetscMatrixParallelComplex(*temp_result);
  else
    result->add_matrix(*temp_result,SAME_NONZERO_PATTERN);
  //delete pointer_to_temp_result;
  //pointer_to_temp_result=NULL;

}

void Self_energy::scattering_optical_deformation_potential_phonon(Propagator*& output_Propagator, const Propagator*& input_Propagator,
    const Propagator*& input_Propagator2,
    Simulation* input_Propagator_solver, Simulation* input_Propagator2_solver,
    const int root_address,const MPI_Comm& MPI_communicator,
    const std::vector<NemoMeshPoint>& self_momentum_point,
    const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
    std::vector<std::complex<double> >*& pointer_to_result_vector,
    std::vector<std::complex<double> >*& pointer_to_result_vector2 ,
    std::vector<int>& group, bool this_is_for_absorption)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::scattering_optical_deformation_potential_phonon ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy("+this->get_name()+")::scattering_optical_deformation_potential_phonon: ";

  //output for debugging:
  int temp_address;
  MPI_Comm_rank(MPI_communicator,&temp_address);
  msg<<prefix<<"got root_address: "<<root_address<<"\twith my own address being: "<<temp_address<<"\n";
  msg<<"for the momentum:\n";
  for(unsigned int i=0; i<self_momentum_point.size(); i++)
    self_momentum_point[i].print();

  //check the output_Propagator type
  NemoPhys::Propagator_type result_type=get_Propagator_type(output_Propagator->get_name());
  //check the needed input is given
  NemoPhys::Propagator_type input1_type=get_Propagator_type(input_Propagator->get_name());
  if(result_type==NemoPhys::Fermion_retarded_self)
  {
    NEMO_ASSERT(input_Propagator2!=NULL,prefix+"second Propagator is missing\n");
    NemoPhys::Propagator_type input2_type=get_Propagator_type(input_Propagator2->get_name());
    NEMO_ASSERT(input1_type==NemoPhys::Fermion_retarded_Green&&input2_type==NemoPhys::Fermion_lesser_Green,
                prefix+"incorrect type of given input Propagators\n");
  }
  else if(result_type==NemoPhys::Fermion_lesser_self)
  {
    NEMO_ASSERT(input1_type==NemoPhys::Fermion_lesser_Green,prefix+"incorrect type of given input Propagator\n");
  }

  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //0. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {
    if(relevant_local_momenta->size()==0)
    {
      Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
      NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
    }
    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }
  else
  {
    Propagator::PropagatorMap::const_iterator temp_cit=output_Propagator->propagator_map.find(self_momentum_point);
    NEMO_ASSERT(temp_cit!=output_Propagator->propagator_map.end(),prefix+"called without task: empty relevant_local_momenta and foreign self_momentum_point\n");
  }

  //1. Get the input matrice(s)
  PetscMatrixParallelComplex* Green1=NULL;
  PetscMatrixParallelComplex* Green2=NULL;
  input_Propagator_solver->get_data(input_Propagator->get_name(),&momentum_point,Green1);
  bool is_diagonal_only=true;
  int off_diagonals = options.get_option("store_offdiagonals",0);
  bool store_atom_blockdiagonal = options.get_option("store_atom_blockdiagonal",false);
  PropagationOptionsInterface* opt_interface = dynamic_cast<PropagationOptionsInterface*>(this);
  if(off_diagonals>0 || store_atom_blockdiagonal || opt_interface->get_compute_blockdiagonal_self_energy())
    is_diagonal_only=false;
  if(Green1->if_container() /*&& !is_diagonal_only*/)
    Green1->assemble();
  if(input_Propagator!=input_Propagator2)
  {
    input_Propagator2_solver->get_data(input_Propagator2->get_name(),&momentum_point,Green2);
    if(Green2->if_container()/*&&!is_diagonal_only*/)
      Green2->assemble();
  }
  int start_local_row;
  int end_local_row;
  Green1->get_ownership_range(start_local_row,end_local_row);
  //2. define which section of the input Green's function to include for the scattering self-energy (usually diagonal)
  if(set_of_row_col_indices.size()==0)
    fill_set_of_row_col_indices(start_local_row,end_local_row);

  //3. integrate as need over the Green's functions(covers the MPI_Reduce command) on matrices in 1.
  //3.1 (optional) determine whether the integrate_submatrix method needs to be called, i.e. whether there is a real integral to be done (i.e. not if there is no periodicity)
  //3.2 prepare the list of meshes to be excluded from integration (in particular energy, since this is scattering with discrete energies)
  std::set<std::string> exclude_integration;
  //find the energy mesh name
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
  std::string energy_name=std::string("");
  for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
    if(momentum_name_it->second==NemoPhys::Energy)
      energy_name=momentum_name_it->first;
  exclude_integration.insert(energy_name);

  //3.3 create the prefactor according to emission or absorption
  std::map<std::vector<NemoMeshPoint>, double> weight_function;
  //std::map<std::vector<NemoMeshPoint>, double>* pointer_to_weight_function=NULL;
  if(relevant_local_momenta!=NULL)
  {
    //pointer_to_weight_function = &weight_function;
    std::set<std::vector<NemoMeshPoint> >::const_iterator it_final_momenta = relevant_local_momenta->begin();
    for(; it_final_momenta != relevant_local_momenta->end(); ++it_final_momenta)
    {

      double final_weight=0;
      double self_energy_weight=0;

      // remove unused variables:
      //double final_momentum_energy = PropagationUtilities::read_energy_from_momentum(this,(*it_final_momenta),output_Propagator);
      //double self_momentum_point_energy = PropagationUtilities::read_energy_from_momentum(this,self_momentum_point,output_Propagator);
      //double final_k = read_kvector_from_momentum((*it_final_momenta),output_Propagator)[2];
      //double self_k = read_kvector_from_momentum(self_momentum_point,output_Propagator)[2];

      for(unsigned int i=0; i<(*it_final_momenta).size(); i++)
      {
        std::string momentum_mesh_name=output_Propagator->momentum_mesh_names[i];
        std::map<std::string, Simulation*>::const_iterator temp_cit=Mesh_Constructors.find(momentum_mesh_name);
        NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
        Simulation* mesh_constructor=temp_cit->second;

        InputOptions& mesh_options=mesh_constructor->get_reference_to_options();

        if(output_Propagator->momentum_mesh_names[i].find("energy")!=std::string::npos)
        {
          if(!mesh_options.get_option(std::string("non_rectangular"),false))
          {
            //get local momentum integration weight
            mesh_constructor->get_data("integration_weight",(*it_final_momenta)[i],final_weight);
            //get self momentum integration weight
            mesh_constructor->get_data("integration_weight",(self_momentum_point)[i],self_energy_weight);

          }
          else
          {
            //if integration weight for energy
            //if(output_Propagator->momentum_mesh_names[i].find("energy")!=std::string::npos)
            {
              //get the k-point from the momentum for local momentum
              std::vector<double> temp_vector=read_kvector_from_momentum(*it_final_momenta, output_Propagator);
              NemoMeshPoint temp_momentum(0,temp_vector);
              std::vector<NemoMeshPoint> temp_vector_momentum(1,temp_momentum);
              //mesh_constructor->get_data("integration_weight",only-k (1dvector<NemoMeshPoint>,only energy NemoMeshPoint,temp_double);
              mesh_constructor->get_data("integration_weight",temp_vector_momentum,(*it_final_momenta)[i],final_weight);

              //get the k-point from the momentum for self energy momentum
              temp_vector=read_kvector_from_momentum(self_momentum_point, output_Propagator);
              NemoMeshPoint temp_momentum2(0,temp_vector);
              std::vector<NemoMeshPoint> temp_vector_momentum2(1,temp_momentum2);
              //mesh_constructor->get_data("integration_weight",only-k (1dvector<NemoMeshPoint>,only energy NemoMeshPoint,temp_double);
              mesh_constructor->get_data("integration_weight",temp_vector_momentum2,(self_momentum_point)[i],self_energy_weight);

            }
          }


          double emiss_norm = 0.0;
          double absorp_norm = 0.0;

          std::set<unsigned int> Hamilton_momentum_indices;
          std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
          PropagationUtilities::find_Hamiltonian_momenta(this,writeable_Propagator,pointer_to_Hamilton_momentum_indices);

          if(this_is_for_absorption)
          {
            std::map<vector<NemoMeshPoint>,map<NemoMeshPoint,double> >::iterator it = emission_norm_factor.find(*it_final_momenta); //
            NEMO_ASSERT(it != emission_norm_factor.end(),prefix+"could not find emission norm factor for absorption");

            NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=(*it_final_momenta)[*(pointer_to_Hamilton_momentum_indices->begin())];
            NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(self_momentum_point)[*(pointer_to_Hamilton_momentum_indices->begin())];
            std::map<NemoMeshPoint,double>  temp = it->second;  // map of all absorption temp_NemoMeshPoint2 (self_mom point) norm factor for this final momenta
            std::map<NemoMeshPoint,double>::iterator map_it = temp.find(temp_NemoMeshPoint2);

            if(map_it!=temp.end())
              emiss_norm = map_it->second; // emission_norm_factor(point_final,self_point)
            else
              emiss_norm = self_energy_weight;

            it = absorption_norm_factor.find(self_momentum_point);
            NEMO_ASSERT(it != absorption_norm_factor.end(),prefix+"could not find absorption norm factor");

            //NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
           // if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(self_momentum_point)[*(pointer_to_Hamilton_momentum_indices->begin())];
            temp = it->second;
            map_it = temp.find(temp_NemoMeshPoint);
            if(map_it!=temp.end())
              absorp_norm = map_it->second;
            else
              absorp_norm =final_weight;
          }
          else
          {
            std::map<vector<NemoMeshPoint>,map<NemoMeshPoint,double> >::iterator it = emission_norm_factor.find(self_momentum_point);
            NEMO_ASSERT(it != emission_norm_factor.end(),prefix+"could not find emission norm factor");

            //const std::vector<NemoMeshPoint> temp_vector_pointer=it_comm_pair->first;

            NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=(self_momentum_point)[*(pointer_to_Hamilton_momentum_indices->begin())];
            NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
            if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(*it_final_momenta)[*(pointer_to_Hamilton_momentum_indices->begin())];
            std::map<NemoMeshPoint,double>  temp = it->second;
            std::map<NemoMeshPoint,double>::iterator map_it = temp.find(temp_NemoMeshPoint2);
            if(map_it!=temp.end())
              emiss_norm = map_it->second;
            else
              emiss_norm = final_weight;


            it = absorption_norm_factor.find(*it_final_momenta);
            NEMO_ASSERT(it != absorption_norm_factor.end(),prefix+"could not find absorption norm factor");

            //NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
            //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=(*it_final_momenta)[*(pointer_to_Hamilton_momentum_indices->begin())];
            temp = it->second;
            map_it = temp.find(temp_NemoMeshPoint);
            if(map_it!=temp.end())
              absorp_norm = map_it->second;
            else
              absorp_norm = self_energy_weight;

          }
    
          NEMO_ASSERT(emiss_norm>0.0&&absorp_norm>0.0,prefix+"emission norm factor or absorption norm factor returned 0");
          double  ratio = 2.0*final_weight/(emiss_norm+absorp_norm);
          //ratio = 2.0*final_weight/(final_weight+self_energy_weight);
          if(*it_final_momenta==self_momentum_point)
                      ratio = 0.0;
          //if(ratio>2.0)
          //NEMO_ASSERT(ratio<=2.0,prefix + "absorption " + NemoUtils::nemo_to_string(this_is_for_absorption) + " ratio is too large " + NemoUtils::nemo_to_string(ratio) + " " + NemoUtils::nemo_to_string(emiss_norm) + " "
          //    + NemoUtils::nemo_to_string(absorp_norm) + "  " + NemoUtils::nemo_to_string(final_weight)+ " "
          //    + NemoUtils::nemo_to_string(final_momentum_energy) + " " + NemoUtils::nemo_to_string(self_momentum_point_energy));

           // cerr <<  prefix + "absorption " << this_is_for_absorption << " ratio is too large " + NemoUtils::nemo_to_string(ratio) + " " + NemoUtils::nemo_to_string(emiss_norm) + " "
           //   + NemoUtils::nemo_to_string(absorp_norm) + "  " + NemoUtils::nemo_to_string(final_weight)+ " "
           //  + NemoUtils::nemo_to_string(final_momentum_energy) + " " + NemoUtils::nemo_to_string(self_momentum_point_energy) << " \n";
          

          weight_function[*it_final_momenta] = ratio;
        }
      }
    }
  }
  //3.4 call the integrate submatrix
  //bool is_diagonal_only=true;
  //int off_diagonals = options.get_option("store_offdiagonals",0);
  //if(off_diagonals>0)
  //  is_diagonal_only=false;
  std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*> local_weight_function_map;
  std::pair<int,int> local_index_pair(0,0);
  local_weight_function_map[local_index_pair]=&weight_function;
  const std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*>* pointer_to_weight_function=&local_weight_function_map;

  NemoPhys::Propagator_type propagator_type = get_Propagator_type(input_Propagator->get_name());
  PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices, relevant_local_momenta, input_Propagator_solver, propagator_type, MPI_communicator,root_address,group,
                      pointer_to_result_vector, exclude_integration, is_diagonal_only,pointer_to_weight_function); //for n-dim periodicity: [eV/eV/nm^n]


  if(result_type==NemoPhys::Fermion_retarded_self)
  {
    NemoPhys::Propagator_type propagator_type = get_Propagator_type(input_Propagator2->get_name());
    PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices, relevant_local_momenta, input_Propagator2_solver, propagator_type, MPI_communicator,root_address,group,
                        pointer_to_result_vector2, exclude_integration, is_diagonal_only,pointer_to_weight_function); //for n-dim periodicity: [eV/eV/nm^n]
  }
  NemoUtils::toc(tic_toc_prefix);
}
void Self_energy::save_and_postprocess_deformation_optical_phonon(Propagator*& /*output_Propagator*/, const Propagator*& input_Propagator,
    const Propagator*& input_Propagator2,
    Simulation* input_Propagator_solver, Simulation* input_Propagator2_solver,
    const std::vector<NemoMeshPoint>& self_momentum_point,
    const bool this_is_for_absorption, const bool /*coupling_to_same_energy*/,
    const std::set<std::vector<NemoMeshPoint> >*& relevant_local_momenta,
    const std::set<vector<NemoMeshPoint> >* /*final_coupling_energies*/,
    std::vector<std::complex<double> >*& pointer_to_result_vector,
    std::vector<std::complex<double> >*& pointer_to_result_vector2 ,
    PetscMatrixParallelComplex*& result)
{
  std::string prefix="Self_energy("+this->get_name()+")::save_and_postprocess_deformation_optical_phonon: ";
  //check the output_Propagator type
  std::vector<NemoMeshPoint> momentum_point=self_momentum_point;
  //0. check that the relevant_local_momenta (i.e. those momenta that are relevant for the self-energy and calculated on this MPI-process) is not empty or self_momentum_point is solved here
  if(relevant_local_momenta!=NULL)
  {
    if(relevant_local_momenta->size()>0)
      momentum_point = (*(relevant_local_momenta->begin()));
  }
  //1. Get the input matrice(s)
  PetscMatrixParallelComplex* Green1=NULL;
  PetscMatrixParallelComplex* Green2=NULL;
  input_Propagator_solver->get_data(input_Propagator->get_name(),&momentum_point,Green1);
  if(input_Propagator!=input_Propagator2)
  {
    input_Propagator2_solver->get_data(input_Propagator2->get_name(),&momentum_point,Green2);

  }

  //4. postprocessing if this MPI-rank is the root_rank (i.e. stores the actual self-energy)
  //4.1 determine whether the energy of self_momentum_point is larger or smaller than the energy of relevant_local_momenta
  //double self_energy=PropagationUtilities::read_energy_from_momentum(this,self_momentum_point, output_Propagator);
  //NEMO_ASSERT(relevant_local_momenta!=NULL,prefix+"received empty list of local momenta\n");

  //double representative_coupling_energy=PropagationUtilities::read_energy_from_momentum(this,*(relevant_local_momenta->begin()), output_Propagator);
  //bool this_is_for_absorption=true;
  //if(self_energy>representative_coupling_energy)
  //  this_is_for_absorption=false;

  //double temperature_in_eV=options.get_option("temperature",NemoPhys::temperature)*NemoPhys::boltzmann_constant/NemoPhys::elementary_charge;
  double phonon_energy=options.get_option("optical_phonon_energy", 0.0); //[eV]
  double phonon_number=NemoMath::bose_distribution(0.0, temperature_in_eV, phonon_energy);
  double optical_deformation_constant=options.get_option("optical_deformation_constant",1.0); //[eV/nm]
  double material_density=options.get_option("material_density",1.0); //[kg/m^3]
  double prefactor=optical_deformation_constant*optical_deformation_constant*NemoPhys::planck_constant*NemoPhys::planck_constant;
  prefactor/=2.0*material_density*phonon_energy*NemoPhys::nm_in_m*NemoPhys::nm_in_m*4.0*NemoMath::pi*NemoMath::pi*NemoPhys::elementary_charge;
  //prefactor/=2.0*material_density*phonon_energy*NemoPhys::nm_in_m*NemoPhys::nm_in_m*NemoPhys::elementary_charge;
  //if this is a retarded self-energy

  //take care of 1/(2*pi)^d factor from Fourier Transform
  const std::vector<vector<double> >reciprocal_basis = get_simulation_domain()->get_reciprocal_vectors();
  int rec_basis_size = reciprocal_basis.size();
  if(rec_basis_size > 0)
    prefactor /= std::pow(2*NemoMath::pi, rec_basis_size);

  if(pointer_to_result_vector2->size()>0)
  {
    //    if(coupling_to_same_energy)
    //    {
    //      NEMO_ASSERT(pointer_to_result_vector2->size()==pointer_to_result_vector->size(),prefix+"inconsistent result vector sizes\n");
    //      for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
    //      {
    //        (*pointer_to_result_vector)[i] = cplx(0.0,0.0);//*=std::complex<double>(((0.5+phonon_number)*prefactor),0.0)*cplx(0.0,0.0);
    //      }
    //    }
    //    else
    {
      if(this_is_for_absorption)
      {
        NEMO_ASSERT(pointer_to_result_vector2->size()==pointer_to_result_vector->size(),prefix+"inconsistent result vector sizes\n");
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
        {
          (*pointer_to_result_vector)[i]*=std::complex<double>(phonon_number*prefactor,0.0);
          (*pointer_to_result_vector)[i]-=((*pointer_to_result_vector2)[i]*prefactor*0.5);
        }
      }
      else
      {
        NEMO_ASSERT(pointer_to_result_vector2->size()==pointer_to_result_vector->size(),prefix+"inconsistent result vector sizes\n");
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
        {
          (*pointer_to_result_vector)[i]*=std::complex<double>((1.0+phonon_number)*prefactor,0.0);
          (*pointer_to_result_vector)[i]+=((*pointer_to_result_vector2)[i]*prefactor*0.5);
        }
      }
    }

  }
  //it this is a lesser self-energy
  else
  {
    //    if(coupling_to_same_energy)
    //    {
    //      for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
    //        (*pointer_to_result_vector)[i]= cplx(0.0,0.0);//*=(0.5+phonon_number)*prefactor*cplx(0.0,0.0);
    //    }
    //    else
    {
      if(this_is_for_absorption)
      {
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
          (*pointer_to_result_vector)[i]*=(1.0+phonon_number)*prefactor;
      }
      else
      {
        for(unsigned int i=0; i<pointer_to_result_vector->size(); i++)
          (*pointer_to_result_vector)[i]*=phonon_number*prefactor;
      }
    }
  }
  int start_local_row;
  int end_local_row;
  Green1->get_ownership_range(start_local_row,end_local_row);

  bool is_diagonal_only=true;
  int off_diagonals = options.get_option("store_offdiagonals",0);
  bool store_atom_blockdiagonal = options.get_option("store_atom_blockdiagonal",false);
  PropagationOptionsInterface* opt_interface = dynamic_cast<PropagationOptionsInterface*>(this);
  if(off_diagonals>0 || store_atom_blockdiagonal || opt_interface->get_compute_blockdiagonal_self_energy())
    is_diagonal_only=false;

   if(options.check_option("store_atomic_offdiagonals"))
   {
     off_diagonals = options.get_option("store_atomic_offdiagonals",0);
     //find the number of orbitals per atom
     double temp_number_of_orbitals;
     Hamilton_Constructor->get_data("number_of_orbitals",temp_number_of_orbitals);
     int number_of_orbitals=int(temp_number_of_orbitals);
     off_diagonals *= number_of_orbitals;
   }

  //4.2. translate the result_vector into temp_result and add to result
  //4.2.1 set up the temp_result matrix
  PetscMatrixParallelComplex* temp_result = new PetscMatrixParallelComplex(Green1->get_num_rows(),Green1->get_num_cols(),
      get_simulation_domain()->get_communicator());
  if(!store_atom_blockdiagonal)
  {
    //4.2.2 create the sparsity pattern for temp_result
    temp_result->set_num_owned_rows(Green1->get_num_owned_rows());
    for (int i = 0; i < start_local_row; i++)
      temp_result->set_num_nonzeros(i,0,0);
    for (int i = start_local_row; i < end_local_row; i++)
      temp_result->set_num_nonzeros(i,2*off_diagonals+1,off_diagonals);
    for (unsigned int i = end_local_row; i < Green1->get_num_rows(); i++)
      temp_result->set_num_nonzeros(i,0,0);
  }
  else
  {
    //4.1 loop over the set_of_row_col_indices
    std::set<std::pair<int,int> >::const_iterator set_cit=set_of_row_col_indices.begin();
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

    }

    temp_result->set_num_owned_rows(Green1->get_num_owned_rows());
    for (int i = 0; i < start_local_row; i++)
      temp_result->set_num_nonzeros(i,0,0);
    for (int i = start_local_row; i < end_local_row; i++)
    {
      nonzero_it = nonzero_map.find(i);
      //NEMO_ASSERT(nonzero_it != nonzero_map.end(),prefix + " could not find the number of nonzeros ");
      double num_nonzeros = 0;
      if(nonzero_it != nonzero_map.end())
        num_nonzeros = nonzero_it->second;
      temp_result->set_num_nonzeros(i,num_nonzeros,0);
      //result->set_num_nonzeros(i,off_diagonals+1,off_diagonals);
    }
    for (unsigned int i = end_local_row; i < Green1->get_num_rows(); i++)
      temp_result->set_num_nonzeros(i,0,0);
  }


  temp_result->allocate_memory();
  temp_result->set_to_zero();

  double delta_fun_prefactor = 1.0;
  if(!is_diagonal_only)
  {
    PetscMatrixParallelComplex* temp_matrix=NULL;
    Hamilton_Constructor->get_data(std::string("dimensional_S_matrix"),temp_matrix);//for n-dim periodicity: [1/nm^(3-n)]
    delta_fun_prefactor *= temp_matrix->get(0,0).real();
    delete temp_matrix;
    temp_matrix=NULL;
  }


  //4.2.3 loop over the set_of_row_col_indices (running index counter is the key of the result_vector)
  unsigned int counter=0;
  std::set<std::pair<int,int> >::const_iterator set_cit=set_of_row_col_indices.begin();
  for(; set_cit!=set_of_row_col_indices.end(); set_cit++)
  {
    int row_index = (*set_cit).first;
    int col_index = (*set_cit).second;
    temp_result->set(row_index,col_index,(*pointer_to_result_vector)[counter]*delta_fun_prefactor);
    counter++;
  }
  temp_result->assemble();

  if(is_diagonal_only)
  {
  //4.2.4 multiply with the Dirac delta function
  //4.2.4.1 get the Dirac delta function
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Hamilton_Constructor->get_data(std::string("dimensional_S_matrix"),temp_matrix);//for n-dim periodicity: [1/nm^(3-n)]

  //4.2.4.2 perform the multiplication
  PetscMatrixParallelComplex* pointer_to_temp_result=NULL;
  PetscMatrixParallelComplex::mult(*temp_matrix,*temp_result,&pointer_to_temp_result);
  delete temp_result;
  temp_result=NULL;
  *pointer_to_temp_result*=std::complex<double>(1.0/(NemoMath::nm_in_m*NemoMath::nm_in_m*NemoMath::nm_in_m),0.0);

  delete temp_matrix;
  temp_matrix=NULL;
  //4.2.5 add temp_result to result
  if(result==NULL)
    result=new PetscMatrixParallelComplex(*pointer_to_temp_result);
  else
    result->add_matrix(*pointer_to_temp_result,SAME_NONZERO_PATTERN);
    delete pointer_to_temp_result;
    pointer_to_temp_result = NULL;

  }
  else //e.g. atom block diagonal
  {
     *temp_result*=std::complex<double>(1.0/(NemoMath::nm_in_m*NemoMath::nm_in_m*NemoMath::nm_in_m),0.0);
     //temp_result=PetscMatrixParallelComplex(*pointer_to_temp_result);

     //4.2.5 add temp_result to result
     if(result==NULL)
       result=new PetscMatrixParallelComplex(*temp_result);
     else
       result->add_matrix(*temp_result,SAME_NONZERO_PATTERN);

     delete temp_result;
     temp_result = NULL;

  }

  if(options.get_option("optical_deformation_potential_tensor",false))
  {
    multiply_deformation_tensor("optical_deformation_potential",result);
  }

}


double Self_energy::analytical_1D_LOphonon_scattering_potential(const double abs_distance, const double delta_k, const double average_xi) const
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::analytical_1D_LOphonon_scattering_potential ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy("+this->get_name()+")::analytical_1D_LOphonon_scattering_potential: ";
  //this method calculates Eq.(3.5.9) of  Christoph Kubis (2010), "Quantum transport in semiconductor nanostructures," 
  NEMO_ASSERT(std::abs(average_xi)>NemoMath::d_zero_tolerance,prefix+"received too small averaged screening length\n");
  double inv_xi = 1.0/average_xi;
  double temp0 = delta_k*delta_k+inv_xi*inv_xi;
  double temp_sqrt = std::sqrt(temp0);
  double temp1 = std::exp(-temp_sqrt*abs_distance);
  double temp2 = 1.0 - inv_xi*inv_xi*abs_distance/2.0/temp_sqrt - inv_xi*inv_xi/2.0/temp0;
  NemoUtils::toc(tic_toc_prefix);
  return NemoMath::pi*temp1*temp2;
}

double Self_energy::analytical_3D_LOphonon_scattering_potential(const double abs_distance, const double /*delta_k*/, const double average_xi) const
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::analytical_3D_LOphonon_scattering_potential ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy("+this->get_name()+")::analytical_3D_LOphonon_scattering_potential: ";

  NEMO_ASSERT(std::abs(average_xi)>NemoMath::d_zero_tolerance,prefix+"received too small averaged screening length\n");
  if(abs_distance>NemoMath::d_zero_tolerance)
  {
    double inv_xi = 1.0/average_xi;
    double temp0=std::exp(-abs_distance*inv_xi)*inv_xi/16.0/NemoMath::pi;
    double temp1=temp0*(2*average_xi/abs_distance-1.0);
    NemoUtils::toc(tic_toc_prefix);
    return temp1;
  }
  else
  {
    double temp1=(2*average_xi/abs_distance-1.0);
    NemoUtils::toc(tic_toc_prefix);
    return temp1;
  }
}

void Self_energy::update_scattering_coupling_table(const double input_phonon_energy)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::update_scattering_coupling_table ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy("+this->get_name()+")::update_scattering_coupling_table: ";
  //1. get the energy mesh
  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
  std::string energy_name=std::string("");
  for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
    if(momentum_name_it->second==NemoPhys::Energy||momentum_name_it->second==NemoPhys::Complex_energy)
      energy_name=momentum_name_it->first;
  std::map<std::string, NemoMesh*>::const_iterator temp_cit=Momentum_meshes.find(energy_name);
  NEMO_ASSERT(temp_cit!=Momentum_meshes.end(),prefix+"have not found mesh \""+energy_name+"\" in the Momentum_meshes\n");
  NemoMesh* energy_mesh=temp_cit->second;
  if(energy_mesh==NULL)
  {
    std::map<std::string, Simulation*>::const_iterator temp_c_it2=Mesh_Constructors.find(temp_cit->first);
    NEMO_ASSERT(temp_c_it2!=Mesh_Constructors.end(),prefix+"no mesh constructor found for \""+temp_cit->first+"\"\n");
    Simulation* mesh_source = temp_c_it2->second;
    mesh_source->get_data("e_space_"+temp_cit->first,energy_mesh);
  }
  NEMO_ASSERT(energy_mesh!=NULL,prefix+"energy mesh is NULL\n");

  //2. identify the pairs of energy mesh points that couple due to the given phonon energy
  std::set<std::pair<double,double> > coupling_energies;
  //2.1 get all energy points
  vector<NemoMeshPoint*>& all_energies=energy_mesh->get_mesh_points();
  //we store this vector into a map to order the energy values
  std::map<double,NemoMeshPoint*> all_energies_map;
  for(unsigned int i=0; i<all_energies.size(); i++)
  {
    double energy1=all_energies[i]->get_x();
    all_energies_map[energy1]=all_energies[i];
  }

  //2.2 loop over all energy points (i.e. initial energies)
  std::map<double,NemoMeshPoint*>::const_iterator energy_cit=all_energies_map.begin();
  for(; energy_cit!=all_energies_map.end(); energy_cit++)
  {
    double energy1=energy_cit->first;
    //find the energy closest to energy1-input_phonon_energy
    double energy2=energy1-input_phonon_energy;
    std::map<double,NemoMeshPoint*>::const_iterator lower_energy_cit=all_energies_map.lower_bound(energy2);
    std::map<double,NemoMeshPoint*>::const_iterator upper_energy_cit=all_energies_map.upper_bound(energy2);
    if(lower_energy_cit!=all_energies_map.end())
    {
      std::pair<double,double> temp_pair(energy1,energy2);
      coupling_energies.insert(temp_pair);
      //msg<<prefix<<"inserted coupling pair: "<<temp_pair.first <<", "<<temp_pair.second<<"\n";
      //the swichting is done in 4.
      //temp_pair.first=energy2; temp_pair.second=energy1;
      //coupling_energies.insert(temp_pair);
    }
    else if(upper_energy_cit!=all_energies_map.end())
    {
      std::pair<double,double> temp_pair(energy1,energy2);
      coupling_energies.insert(temp_pair);
      //msg<<prefix<<"inserted coupling pair: "<<temp_pair.first <<", "<<temp_pair.second<<"\n";
      //the swichting is done in 4.
      //temp_pair.first=energy2; temp_pair.second=energy1;
      //coupling_energies.insert(temp_pair);
    }
    //find the energy closest to energy1+input_phonon_energy
    double energy3=energy1+input_phonon_energy;
    lower_energy_cit=all_energies_map.lower_bound(energy3);
    upper_energy_cit=all_energies_map.upper_bound(energy3);
    if(upper_energy_cit!=all_energies_map.end())
    {
      std::pair<double,double> temp_pair(energy1,energy3);
      coupling_energies.insert(temp_pair);
      //msg<<prefix<<"inserted coupling pair: "<<temp_pair.first <<", "<<temp_pair.second<<"\n";
      //the swichting is done in 4.
      //temp_pair.first=energy3; temp_pair.second=energy1;
      //coupling_energies.insert(temp_pair);
    }
    lower_energy_cit=all_energies_map.upper_bound(energy3);
    if(lower_energy_cit!=all_energies_map.end())
    {
      std::pair<double,double> temp_pair(energy1,energy3);
      coupling_energies.insert(temp_pair);
      //msg<<prefix<<"inserted coupling pair: "<<temp_pair.first <<", "<<temp_pair.second<<"\n";
      //the swichting is done in 4.
      //temp_pair.first=energy3; temp_pair.second=energy1;
      //coupling_energies.insert(temp_pair);
    }
  }

  //3. find a translation map between the energy value and all possible momentum configurations with this energy
  std::map<double,std::set<std::vector<NemoMeshPoint> > > energy_to_full_momentum_map;
  std::set<std::vector<NemoMeshPoint> >::const_iterator all_momenta_cit=pointer_to_all_momenta->begin();
  //3.1 loop over all momentum configurations
  for(; all_momenta_cit!=pointer_to_all_momenta->end(); all_momenta_cit++)
  {
    const Propagator* temp_propagator=writeable_Propagator; //s.begin()->second;
    //3.2 find the energy of this momentum
    double temp_energy = PropagationUtilities::read_energy_from_momentum(this,*all_momenta_cit,temp_propagator);
    std::map<double,std::set<std::vector<NemoMeshPoint> > >::iterator temp_it=energy_to_full_momentum_map.find(temp_energy);
    //3.3. store the momentum due to its energy
    if(temp_it!=energy_to_full_momentum_map.end())
      temp_it->second.insert(*all_momenta_cit);
    else
    {
      std::set<std::vector<NemoMeshPoint> > temp_set;
      temp_set.insert(*all_momenta_cit);
      energy_to_full_momentum_map[temp_energy]=temp_set;
    }
  }

  //4. store the results in the scattering_coupling_table
  std::map<double, std::set<scattering_coupling_set_pair> >::iterator result_it=scattering_coupling_table.find(input_phonon_energy);
  //4.1 loop over all energy pairs
  std::set<std::pair<double,double> >::iterator coupling_set_it=coupling_energies.begin();
  for(; coupling_set_it!=coupling_energies.end(); coupling_set_it++)
  {
    //4.2 find the set of momenta that belongs to this energy pair
    double temp_energy1 = (*coupling_set_it).first;
    double temp_energy2 = (*coupling_set_it).second;
    std::map<double,std::set<std::vector<NemoMeshPoint> > >::iterator temp_it1=energy_to_full_momentum_map.find(temp_energy1);
    NEMO_ASSERT(temp_it1!=energy_to_full_momentum_map.end(),prefix+"have not found energy1 in energy_to_full_momentum_map\n");
    NEMO_ASSERT(temp_it1->second.size()>0,prefix+"no momentum with given energy1 found\n");
    std::map<double,std::set<std::vector<NemoMeshPoint> > >::const_iterator temp_it2=energy_to_full_momentum_map.find(temp_energy2);
    NEMO_ASSERT(temp_it2!=energy_to_full_momentum_map.end(),prefix+"have not found energy2 in energy_to_full_momentum_map\n");
    NEMO_ASSERT(temp_it2->second.size()>0,prefix+"no momentum with given energy2 found\n");

    scattering_coupling_set_pair temp_momentum_pair(temp_it1->second,temp_it2->second);
    //4.3 store the coupling momentum sets
    if(result_it==scattering_coupling_table.end())
    {
      std::set<scattering_coupling_set_pair> temp_set;
      temp_set.insert(temp_momentum_pair);
      //temp_set.insert(temp_momentum_pair2);
      scattering_coupling_table[input_phonon_energy]=temp_set;
      result_it=scattering_coupling_table.find(input_phonon_energy);
    }
    else
    {
      //result_it->second=temp_momentum_pair;
      result_it->second.insert(temp_momentum_pair);
      //result_it->second.insert(temp_momentum_pair2);
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_lesser(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                  PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_lesser ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy("+this->get_name()+")::do_solve_lesser: ";
  std::string Propagator_name=output_Propagator->get_name();
  std::map<std::string, NemoPhys::Propagator_type>::const_iterator type_c_it=Propagator_types.find(Propagator_name);
  NEMO_ASSERT(type_c_it!=Propagator_types.end(),prefix+"have not found propagator \""+Propagator_name+"\" in the type map\n");
  if(type_c_it->second==NemoPhys::Fermion_lesser_self || type_c_it->second==NemoPhys::Boson_lesser_self)
  {
    //call the appropiate do_solve_lesser method:
    if(Propagator_name.find(std::string("contact"))!=std::string::npos)
    {
      do_solve_lesser_contact(output_Propagator,momentum_point,result);
    }
    else if(Propagator_name.find(std::string("scattering"))!=std::string::npos)
    {
      do_solve_scattering_lesser(output_Propagator,momentum_point,result);
    }
    else if(Propagator_name.find(std::string("Buettiker_probe"))!=std::string::npos)
    {
      do_solve_Buettiker_lesser(output_Propagator,momentum_point,result);
    }
    else
      throw std::runtime_error(prefix+"called with unknown option\n");
  }
  else
  {
    std::string prop_type=Propagator_type_map.find(type_c_it->second)->second;
    throw std::runtime_error(prefix+"called with unknown Propagator type:\""+prop_type+"\"\n");
  }
  set_job_done_momentum_map(&(output_Propagator->get_name()), &momentum_point, true);
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_scattering_lesser(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_scattering_lesser ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy("+this->get_name()+")::do_solve_scattering_lesser: ";

  //1. get the type of the output_Propagator (either Boson or Fermion retarded self-energy)
  std::map<std::string, NemoPhys::Propagator_type>::const_iterator Propagator_type_c_it=Propagator_types.find(output_Propagator->get_name());
  NEMO_ASSERT(Propagator_type_c_it!=Propagator_types.end(),prefix+"have not found the propagator type of \""+output_Propagator->get_name()+"\"\n");
  NemoPhys::Propagator_type output_propagator_type=Propagator_type_c_it->second;

  //2. get the solver of the lesser Green's function of this momentum that as the same particle type as the output_Propagator (lesser_Green)
  //we assume that there is exactly one lesser Green's function in the list of the readable Propagators that has the same particle type as the output_Propagator
  //this is a prerequisite for all scattering self energies in the self-consistent Born approximation (NOTE: add a flag - Buettiker probe later)
  const Propagator* lesser_Green=NULL;
  Simulation* lesser_Green_solver=NULL;
  if(output_propagator_type==NemoPhys::Fermion_lesser_self)
  {
    find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,lesser_Green,lesser_Green_solver);
    NEMO_ASSERT(lesser_Green!=NULL,prefix+"pointer to propagator of type \"Fermion_lesser_Green\" is NULL\n");
  }
  else
  {
    find_solver_and_readable_propagator_of_type(NemoPhys::Boson_lesser_Green,lesser_Green,lesser_Green_solver);
    NEMO_ASSERT(lesser_Green!=NULL,prefix+"pointer to propagator of type \"Boson_lesser_Green\" is NULL\n");
  }

  //3. and get the solver of the retarded Green's function of this momentum that as different particle type as the output_Propagator (ret_scattering_Green)
  //and we assume that there is exactly one retarded Green's function in the list of the readable Propagators that has the different particle type as the output_Propagator
  //if this Propagator and its solver is given, we assume Sigma = D*G*potential
  //otherwise we will use Sigma = F*G, with F being a scattering type specific function
  const Propagator* lesser_scattering_Green=NULL;
  Simulation* lesser_scattering_Green_solver=NULL;
  std::string scattering_type=std::string("");
  if(!options.check_option("scattering_type"))
  {
    if(output_propagator_type==NemoPhys::Fermion_lesser_self)
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Boson_lesser_Green,lesser_scattering_Green,lesser_scattering_Green_solver);
      NEMO_ASSERT(lesser_Green!=NULL,prefix+"have not found a propagator of type \"Boson_lesser_Green\"\n");
    }
    else
    {
      find_solver_and_readable_propagator_of_type(NemoPhys::Fermion_lesser_Green,lesser_scattering_Green,lesser_scattering_Green_solver);
      NEMO_ASSERT(lesser_Green!=NULL,prefix+"have not found a propagator of type \"Fermion_lesser_Green\"\n");
    }
  }
  else
    scattering_type = options.get_option("scattering_type",std::string(""));

  if(scattering_type=="lambda_G_scattering")
  {
    //scattering self-energy is the Green's function multiplied with lambda
    //read in the proportionality constant lambda
   //check whether material specific lambda is specified
   if(options.get_option("material_specific_lambda",false))
   {
      scattering_lambda_proportional_to_G(output_Propagator, lesser_Green, lesser_Green_solver, momentum_point,result);
   }
   else
   {
      double lambda=options.get_option("lambda",0.0);
      scattering_lambda_proportional_to_G(output_Propagator, lesser_Green, lesser_Green_solver, lambda, momentum_point,result);
   }
  }
  else
    throw std::runtime_error(prefix+"scattering of type \""+scattering_type+"\" not implemented, yet\n");
  NemoUtils::toc(tic_toc_prefix);
}


void Self_energy::do_solve_lesser_contact(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum,
    PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_lesser_contact ";
  std::string prefix = tic_toc_prefix;
  NemoUtils::tic(tic_toc_prefix);
  if(options.get_option("equilibrium_model",true))
  {
    PropagationUtilities::do_solve_lesser_equilibrium(this,output_Propagator,momentum, result);
  }
  else
  {
    //this version solves the lesser contact self-energy using a contact Green's function (required for RGF-version2)
    //get the domain pointer corresponding to the desired contact self-energy
    std::string variable_name=output_Propagator->get_name()+std::string("_lead_domain");
    std::string neighbor_domain_name;
    if (options.check_option(variable_name))
      neighbor_domain_name=options.get_option(variable_name,std::string(""));
    else
      throw std::invalid_argument(prefix+" define \""+variable_name+"\"\n");
    const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);
    NEMO_ASSERT(neighbor_domain!=NULL,prefix+"Domain \""+neighbor_domain_name+"\" has not been found!\n");
    //direct_iterative_leads(neighbor_domain,momentum, result);
    PropagationUtilities::direct_iterative_leads(this, Hamilton_Constructor, momentum_mesh_types,Propagators, Propagator_types,
                                                  neighbor_domain, momentum,result);

  }
  set_job_done_momentum_map(&(output_Propagator->get_name()), &momentum, true);
  //symmetrize(result,NemoMath::antihermitian);
  //{
  //  //debug: make result antihermitian
  //  PetscMatrixParallelComplex temp_result=PetscMatrixParallelComplex(result->get_num_cols(),
  //    result->get_num_rows(),
  //    result->get_communicator());
  //  result->hermitian_transpose_matrix(temp_result,MAT_INITIAL_MATRIX);
  //  result->add_matrix(temp_result, DIFFERENT_NONZERO_PATTERN,std::complex<double> (-1.0,0.0));
  //  *result*=std::complex<double>(0.5,0.0);
  //}

  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_lesser_contact(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum)
{
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_lesser_contact";
  NemoUtils::tic(tic_toc_prefix);
  activate_regions();
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=output_Propagator->propagator_map.find(momentum);
  if(matrix_it!=output_Propagator->propagator_map.end())
    temp_matrix=matrix_it->second;
  do_solve_lesser_contact(output_Propagator,momentum,temp_matrix);
  NEMO_ASSERT(is_ready(output_Propagator->get_name(),momentum),tic_toc_prefix+"still not ready\n");
  write_propagator(output_Propagator->get_name(),momentum, temp_matrix);
  conclude_after_do_solve(output_Propagator,momentum);
  NemoUtils::toc(tic_toc_prefix);
}

//this is a new version of the do_solve_constant_eta - it is allocating matrices for all requested eta matrices
//minimizing the storage is the responsibility of the calling Propagation or module
void Self_energy::do_solve_constant_eta(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
                                        PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_constant_eta2 ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::do_solve_constant_eta: ";
  const DOFmapInterface* defining_DOFmap=&(Hamilton_Constructor->get_dof_map(get_const_simulation_domain()));
  //allocate only one diagonal matrix and let the MatrixPointers of other momenta point to it
  std::map<const std::vector<NemoMeshPoint>,bool>::iterator alloc_it=output_Propagator->allocated_momentum_Propagator_map.find(momentum_point);
  alloc_it->second=true;
  delete result;
  result=NULL;
  result=new PetscMatrixParallelComplex(defining_DOFmap->get_global_dof_number(),defining_DOFmap->get_global_dof_number(),get_simulation_domain()->get_communicator());
  result->set_num_owned_rows(result->get_num_rows());
  for(unsigned int i=0;i<result->get_num_rows();i++)
    result->set_num_nonzeros_for_local_row(i,1,0);
  result->allocate_memory();
  /*if(alloc_it==output_Propagator->allocated_momentum_Propagator_map.end())
  {
    output_Propagator->allocated_momentum_Propagator_map[momentum_point]=false;
    alloc_it=output_Propagator->allocated_momentum_Propagator_map.find(momentum_point);
  }

  if(!alloc_it->second)
  {
    allocate_propagator_matrices(defining_DOFmap,&momentum_point);
    alloc_it->second=true;
  }*/
  Propagator::PropagatorMap::iterator prop_it=output_Propagator->propagator_map.find(momentum_point);
  NEMO_ASSERT(prop_it!=output_Propagator->propagator_map.end(),prefix+"have not found the propagator matrix for this momentum\n");
  //prop_it->second->assemble();
  //PetscMatrixParallelComplex* diagonal_eta = prop_it->second;
  //diagonal_eta->set_to_zero();
  const double eta=options.get_option(std::string("constant_eta"),0.0);
  //diagonal_eta->matrix_diagonal_shift(std::complex<double> (0.0,eta));
  for(unsigned int i=0;i<result->get_num_rows();i++)
    result->set(i,i,std::complex<double> (0.0,eta));
  result->assemble();
  //result->matrix_diagonal_shift(std::complex<double> (0.0,eta));
  //result->assemble();
  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum_point,true);
  //result = prop_it->second;
  prop_it->second=result;
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_retarded_contact(Propagator*& output_Propagator,const std::vector<NemoMeshPoint>& momentum_point)
{
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_retarded_contact2 ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::do_solve_retarded_contact(): ";
  activate_regions();
  PetscMatrixParallelComplex* temp_matrix=NULL;
  Propagator::PropagatorMap::iterator matrix_it=output_Propagator->propagator_map.find(momentum_point);
  if(matrix_it!=output_Propagator->propagator_map.end())
    temp_matrix=matrix_it->second;
  do_solve_retarded_contact(output_Propagator,momentum_point,temp_matrix);
  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum_point,true);

  NEMO_ASSERT(is_ready(output_Propagator->get_name(),momentum_point),prefix+"still not ready\n");
  write_propagator(output_Propagator->get_name(),momentum_point, temp_matrix);
  conclude_after_do_solve(output_Propagator,momentum_point);
  NemoUtils::toc(tic_toc_prefix);
}


void Self_energy::do_solve_retarded_contact(Propagator*& output_Propagator,const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_retarded_contact2 ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::do_solve_retarded_contact(): ";


  //get the domain pointer corresponding to the desired contact self-energy
  std::string variable_name=output_Propagator->get_name()+std::string("_lead_domain");
  std::string neighbor_domain_name;
  if (options.check_option(variable_name))
    neighbor_domain_name=options.get_option(variable_name,std::string(""));
  else
    throw std::invalid_argument(prefix+" define \""+variable_name+"\"\n");
  const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);
  NEMO_ASSERT(neighbor_domain!=NULL,prefix+"Domain \""+neighbor_domain_name+"\" has not been found!\n");
  do_solve_retarded_contact(output_Propagator,neighbor_domain,momentum_point,result);
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_retarded_contact(Propagator*& output_Propagator, const Domain* neighbor_domain, const std::vector<NemoMeshPoint>& momentum,
    PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_retarded_contact3 ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::do_solve_retarded_contact ";
  std::string retarded_lead_method=options.get_option("retarded_lead_method", std::string("direct_iterations"));
  if (retarded_lead_method=="Sancho_Rubio")
  {
    Sancho_Rubio_lead(output_Propagator,neighbor_domain,momentum,result);
  }
  else if(retarded_lead_method=="direct_iterations")
  {
    std::string variable = output_Propagator->get_name();
    //direct_iterative_leads(neighbor_domain,momentum, result);
    PropagationUtilities::direct_iterative_leads(this, Hamilton_Constructor, momentum_mesh_types,Propagators, Propagator_types,
                                                  neighbor_domain, momentum,result);

  }
  //else if(retarded_lead_method=="transfer_matrix")
  else if(retarded_lead_method.find("transfer_matrix")!=std::string::npos)
  {
#ifndef NO_MKL
    int previous_num_threads=mkl_get_max_threads();
    int dynamic_state=mkl_get_dynamic();
    if(options.check_option("bc_mkl_threads"))
    {
      int num_threads = options.get_option("bc_mkl_threads",1);
      mkl_set_dynamic(false);
      mkl_set_num_threads(num_threads);
    }
#endif
    PropagationUtilities::transfer_matrix_leads(this,output_Propagator, neighbor_domain, momentum, result);
#ifndef NO_MKL
    if(options.check_option("bc_mkl_threads"))
    {
      mkl_set_num_threads(previous_num_threads);
      mkl_set_dynamic(dynamic_state);
    }
#endif
  }
  else
    throw std::invalid_argument("Self_energy::do_solve_contact called with unknown method: "+retarded_lead_method+"\n");
  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum,true);
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::Sancho_Rubio_lead(Propagator*& output_Propagator, const Domain* neighbor_domain, const std::vector<NemoMeshPoint>& momentum,
                                    PetscMatrixParallelComplex*& result)
{
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::Sancho_Rubio_lead: ";
  NemoUtils::tic(tic_toc_prefix);
  msg<<"Self_energy(\""+this->get_name()+"\")::Sancho_Rubio_lead()"<<std::endl;

  //std::set<unsigned int>Hamilton_momentum_indices;
  //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices = &Hamilton_momentum_indices;
  //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
  //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
  //if(pointer_to_Hamilton_momentum_indices!=NULL)
  //  temp_NemoMeshPoint = momentum[*(pointer_to_Hamilton_momentum_indices->begin())];

  PetscMatrixParallelComplex* surface_g = NULL;
  PropagationUtilities::Sancho_Rubio_retarded_green(this,output_Propagator, neighbor_domain, momentum,surface_g);


  // sigma = t*gr*t'
  //6.1 get coupling Hamiltonian between device and contact
  // current domain is device, neighbor_domain is contact
  PetscMatrixParallelComplex* coupling_device = NULL;
  PetscMatrixParallelComplex* coupling_overlap_device = NULL;
  const DOFmapInterface& device_DOFmap = Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain());
  unsigned int number_of_rows2 = device_DOFmap.get_global_dof_number();
  unsigned int number_of_cols2 = number_of_rows2;
  //DOFmapInterface* coupling_DOFmap_device=NULL;
  DOFmapInterface* coupling_Over_DOFmap_device=NULL;

  Simulation* temp_simulation=find_simulation_on_domain(neighbor_domain,Hamilton_Constructor->get_type());
  if (temp_simulation == NULL && (Hamilton_Constructor->get_type().find("module") != std::string::npos || Hamilton_Constructor->get_type().find("Module") != std::string::npos))
    throw std::runtime_error(tic_toc_prefix + "can't find simulation of type " + Hamilton_Constructor->get_type() + " on domain " + neighbor_domain->get_name() + "\n");
  NEMO_ASSERT(temp_simulation != NULL, tic_toc_prefix + "have not found Hamilton_Constructor for neighbor_domain\n");
  DOFmapInterface& temp_DOFmap=temp_simulation->get_dof_map(neighbor_domain);
  DOFmapInterface* coupling_DOFmap_device=&temp_DOFmap;
  DOFmapInterface* temp_pointer=coupling_DOFmap_device;

  std::vector<NemoMeshPoint> sorted_momentum;
  QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,momentum_mesh_types,Hamilton_Constructor);

  Hamilton_Constructor->get_data(std::string("Hamiltonian"),sorted_momentum, neighbor_domain, coupling_device, coupling_DOFmap_device);

  //coupling_device->assemble();
  //coupling_device->save_to_matlab_file("coupling_H.m");

  Hamilton_Constructor->get_data(std::string("overlap_matrix_coupling"), sorted_momentum, neighbor_domain, coupling_overlap_device, coupling_Over_DOFmap_device);

  if(coupling_DOFmap_device!=temp_pointer)
  {
    delete coupling_DOFmap_device;
    coupling_DOFmap_device =NULL;
  }




  unsigned int number_of_coupling1_rows = coupling_device->get_num_rows();
  unsigned int number_of_rows1 = number_of_coupling1_rows - number_of_rows2;
  unsigned int number_of_cols1 = number_of_rows1;
  //coupling_device        ->save_to_matlab_file("coupling_deviceH.m");
  //coupling_overlap_device->save_to_matlab_file("coupling_deviceS.m");

  // get upper right block
  unsigned int start_row = 0;
  unsigned int end_row = 0;
  unsigned int nonzero_index = 0;

  for(unsigned int i=0; i<number_of_coupling1_rows; i++)
  {
    nonzero_index = coupling_device->get_nz_diagonal(i);
    if(nonzero_index>0)
    {
      start_row = i;
      break;
    }
  }
  nonzero_index = 0;
  for(unsigned int i=number_of_coupling1_rows; i>0; i--)
  {
    nonzero_index = coupling_device->get_nz_diagonal(i-1);
    if(nonzero_index>0)
    {
      end_row = i-1;
      break;
    }
  }
  unsigned int number_of_couple_rows1 = end_row + 1 - start_row;

  //6.2 get sub block
  std::vector<int> temp_rows1(number_of_couple_rows1);
  std::vector<int> temp_cols1(number_of_cols1);
  for(unsigned int i=0; i<number_of_couple_rows1; i++)
    temp_rows1[i]=i+start_row;
  for(unsigned int i=0; i<number_of_cols1; i++)
    temp_cols1[i]=i+number_of_cols2;

  //! Get Hamiltonian sub_block
  PetscMatrixParallelComplex* small_coupling_device = NULL;
  small_coupling_device= new PetscMatrixParallelComplex(number_of_couple_rows1,number_of_cols1,get_simulation_domain()->get_communicator() );
  small_coupling_device->set_num_owned_rows(number_of_couple_rows1);
  vector<int> rows_diagonal1(number_of_couple_rows1,0);
  vector<int> rows_offdiagonal1(number_of_couple_rows1,0);
  unsigned int number_of_nonzero_cols_local1=0;
  unsigned int number_of_nonzero_cols_nonlocal1=0;

  for(unsigned int i=0; i<number_of_couple_rows1; i++)
  {
    rows_diagonal1[i]=coupling_device->get_nz_diagonal(i+start_row);
    rows_offdiagonal1[i]=coupling_device->get_nz_offdiagonal(i+start_row);
    if(rows_diagonal1[i]>0)
      number_of_nonzero_cols_local1++;
    if(rows_offdiagonal1[i]>0)
      number_of_nonzero_cols_nonlocal1++;
  }

  for(unsigned int i=0; i<number_of_couple_rows1; i++)
    small_coupling_device->set_num_nonzeros_for_local_row(i,rows_diagonal1[i],rows_offdiagonal1[i]);
  coupling_device->get_submatrix(temp_rows1,temp_cols1,MAT_INITIAL_MATRIX,small_coupling_device);
  small_coupling_device->assemble();
  //small_coupling_device->save_to_matlab_file("tauH_red.m");

  //! Get Overlap sub_block
  if(coupling_overlap_device!=NULL)
  {
    PetscMatrixParallelComplex* small_coupling_overlap_device = NULL;
    small_coupling_overlap_device= new PetscMatrixParallelComplex(number_of_couple_rows1,number_of_cols1,get_simulation_domain()->get_communicator() );
    small_coupling_overlap_device->set_num_owned_rows(number_of_couple_rows1);
    vector<int> rows_diagonal1s(number_of_couple_rows1,0);
    vector<int> rows_offdiagonal1s(number_of_couple_rows1,0);
    unsigned int number_of_nonzero_cols_local1s=0;
    unsigned int number_of_nonzero_cols_nonlocal1s=0;

    for(unsigned int i=0; i<number_of_couple_rows1; i++)
    {
      rows_diagonal1s[i]    = coupling_overlap_device->get_nz_diagonal(i+start_row);
      rows_offdiagonal1s[i] = coupling_overlap_device->get_nz_offdiagonal(i+start_row);
      if(rows_diagonal1s[i]>0)
        number_of_nonzero_cols_local1s++;
      if(rows_offdiagonal1s[i]>0)
        number_of_nonzero_cols_nonlocal1s++;
    }

    for(unsigned int i=0; i<number_of_couple_rows1; i++)
      small_coupling_overlap_device->set_num_nonzeros_for_local_row(i,rows_diagonal1s[i],rows_offdiagonal1s[i]);
    coupling_overlap_device->get_submatrix(temp_rows1,temp_cols1,MAT_INITIAL_MATRIX,small_coupling_overlap_device);
    small_coupling_overlap_device->assemble();
    //small_coupling_overlap_device->save_to_matlab_file("tauS_red.m");

    const Propagator* temp_propagator=Propagators.begin()->second;
    const double energy = PropagationUtilities::read_energy_from_momentum(this,momentum, temp_propagator);

    *small_coupling_overlap_device *= energy; //E*S
    // H-ES
    small_coupling_device->add_matrix(*small_coupling_overlap_device,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));
    *small_coupling_device *= std::complex<double>(-1.0,0.0); //ES-H
    delete small_coupling_overlap_device;
    small_coupling_overlap_device=NULL;
  }

  //small_coupling_device       ->save_to_matlab_file("Tau.m");
  PetscMatrixParallelComplex* temp_matrix1=NULL;
  PetscMatrixParallelComplex::mult(*small_coupling_device,*surface_g,&temp_matrix1); //t*gr
  //temp_matrix1       ->save_to_matlab_file("T_gr.m");
  PetscMatrixParallelComplex transpose_coupling(small_coupling_device->get_num_cols(),small_coupling_device->get_num_rows(),
      small_coupling_device->get_communicator());
  small_coupling_device->hermitian_transpose_matrix(transpose_coupling, MAT_INITIAL_MATRIX); //t'
  delete small_coupling_device;
  small_coupling_device=NULL;
  PetscMatrixParallelComplex* temp_result=NULL;
  PetscMatrixParallelComplex::mult(*temp_matrix1,transpose_coupling,&temp_result); //t*gr*t'
  //temp_result       ->save_to_matlab_file("T_gr_T.m");

  delete temp_matrix1;
  temp_matrix1=NULL;

  //6.3. copy result into a sparse matrix
  vector<int> result_diagonal(number_of_rows2,0);
  vector<int> result_offdiagonal(number_of_rows2,0);

  for(unsigned int i=start_row; i<=end_row; i++)
  {
    if(rows_diagonal1[i-start_row]>0)
      result_diagonal[i]=number_of_nonzero_cols_local1;
    if(rows_offdiagonal1[i-start_row]>0)
      result_offdiagonal[i]=number_of_nonzero_cols_nonlocal1;
  }

  PropagationUtilities::transfer_matrix_initialize_temp_matrix(this,number_of_rows2,number_of_rows2,result_diagonal,result_offdiagonal,result);

  const std::complex<double>* pointer_to_data= NULL;
  vector<cplx> data_vector;
  vector<int> col_index;
  int n_nonzeros=0;
  const int* n_col_nums=NULL;

  for(unsigned int i=0; i<temp_result->get_num_rows(); i++)
  {
    if(rows_diagonal1[i]>0)
    {
      temp_result->get_row(i,&n_nonzeros,n_col_nums,pointer_to_data);
      // petsc tells us that temp_result is always full
      //n_nonzeros must be the same as it number of columns
      col_index.resize(number_of_nonzero_cols_local1,0);
      data_vector.resize(number_of_nonzero_cols_local1,cplx(0.0,0.0));
      unsigned int result_i=0;
      for(int j=0; j<n_nonzeros; j++)
      {
        if(rows_diagonal1[j]>0)
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
  temp_result = NULL;
  //delete coupling_DOFmap;
  //coupling_DOFmap = NULL;
  //delete coupling_DOFmap_device;
  coupling_DOFmap_device = NULL;
  delete coupling_device;
  coupling_device = NULL;
  //result->save_to_matlab_file("sigma_SR.m");
  NemoUtils::toc(tic_toc_prefix);

  delete surface_g;
  surface_g = NULL;

}




//void Self_energy::transfer_matrix_get_self_energy(PetscMatrixParallelComplex*& E_minus_H_matrix, PetscMatrixParallelComplex*& T_matrix,
//    unsigned int* number_of_wave, unsigned int* /*number_of_vectors_size*/, PetscMatrixParallelComplex*& phase_factor,
//    PetscMatrixParallelComplex*& Wave_matrix, PetscMatrixParallelComplex*& result_matrix,
//    unsigned int number_of_sub_rows, unsigned int number_of_sub_cols)
//{
//  //calculate self-energy with given modes 
//  //result with subdomain size -- NEGF or QTBM that doesn't use smart dofmap
//  tic_toc_name = options.get_option("tic_toc_name",get_name());
//  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::transfer_matrix_get_self_energy ";
//  NemoUtils::tic(tic_toc_prefix);
//  std::string prefix = "Self_energy(\""+get_name()+"\")::transfer_matrix_get_self_energy ";
//
//  unsigned int number_of_device_rows = T_matrix->get_num_rows(); //coupling Hamiltonian matrix 
//  unsigned int number_of_device_cols = number_of_device_rows;
//  if(*number_of_wave==0) //if number of modes is 0, fake the result matrix with a number 0
//  { 
//    //result defined in subdomain size
//    result_matrix = new PetscMatrixParallelComplex(number_of_sub_rows,number_of_sub_cols,get_simulation_domain()->get_communicator() );
//    result_matrix->set_num_owned_rows(number_of_sub_rows);
//    for(unsigned int i=0; i<number_of_sub_rows; i++)
//      result_matrix->set_num_nonzeros(i,1,0);
//    result_matrix->allocate_memory();
//    result_matrix->set_to_zero();
//    result_matrix->assemble();
//  }
//  else 
//  {
//    PetscMatrixParallelComplex WaveT(Wave_matrix->get_num_cols(),Wave_matrix->get_num_rows(),get_simulation_domain()->get_communicator() );
//    PetscMatrixParallelComplex* GR_inv_temp1 = NULL;
//    PetscMatrixParallelComplex* GR_inv1 = NULL;
//    PetscMatrixParallelComplex* GR_inv_temp2 = NULL;
//    PetscMatrixParallelComplex* GR_inv2 = NULL;
//
//    std::string tic_toc_mult1 = tic_toc_prefix+": invg = VL'*(E-H)*VL+VL'*T*VL*expmikdelta";
//    NemoUtils::tic(tic_toc_mult1);
//    Wave_matrix->hermitian_transpose_matrix(WaveT,MAT_INITIAL_MATRIX);//VL'
//    PetscMatrixParallelComplex::mult(*Wave_matrix,*phase_factor,&GR_inv2); //GR_inv2=VL*expmikdelta
//    PetscMatrixParallelComplex::mult(*T_matrix,*GR_inv2,&GR_inv_temp2); //GR_inv_temp2=T*VL*expmikdelta //delete GR_inv2 after this line
//    delete GR_inv2;
//    PetscMatrixParallelComplex::mult(*E_minus_H_matrix,*Wave_matrix,&GR_inv_temp1); //GR_inv_temp1=(E-H)*VL
//    GR_inv_temp1->add_matrix(*GR_inv_temp2,DIFFERENT_NONZERO_PATTERN); //GR_inv_temp1 = (E-H)*VL+T*VL*expmikdelta //delete GR_inv_temp2 after this line
//    delete GR_inv_temp2;
//    PetscMatrixParallelComplex::mult(WaveT,*GR_inv_temp1,
//                                     &GR_inv1); //GR_inv1=GR_inv1=VL'*(E-H)*VL+VL'*T*VL*expmikdelta //delete GR_inv_temp1 after this line
//    delete GR_inv_temp1;
//    NemoUtils::toc(tic_toc_mult1);
//
//    std::string tic_toc_VLmultT= tic_toc_prefix+": VL'*T01, T10*VL ";
//    NemoUtils::tic(tic_toc_VLmultT);
//
//    vector<int> T_rows_diagonal(number_of_sub_rows,0);
//    vector<int> T_rows_offdiagonal(number_of_sub_rows,0);
//    for(unsigned int i=0; i<number_of_sub_rows; i++)
//    {
//      T_rows_diagonal[i]=T_matrix->get_nz_diagonal(i);
//      T_rows_offdiagonal[i]=T_matrix->get_nz_offdiagonal(i);
//    }
//    PetscMatrixParallelComplex* couple_temp = NULL;
//    //get only the sub rows that are allocated -- this is important for efficient solution of the linear equation
//    PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_sub_rows,number_of_device_cols,0,0,T_rows_diagonal,T_rows_offdiagonal,T_matrix,couple_temp);
//    PetscMatrixParallelComplex* couplemultwave = NULL;
//    PetscMatrixParallelComplex::mult(*couple_temp,*Wave_matrix,
//                                     &couplemultwave); //T10*VL // petsc assumes sparse*dense = dense, even though it has tons of zeros!
//    delete couple_temp;
//    PetscMatrixParallelComplex wavemultcouple(couplemultwave->get_num_cols(),couplemultwave->get_num_rows(),get_simulation_domain()->get_communicator() );
//    couplemultwave->hermitian_transpose_matrix(wavemultcouple,MAT_INITIAL_MATRIX);//(T10*VL)' = VL'*T01
//    NemoUtils::toc(tic_toc_VLmultT);
//
//    PetscMatrixParallelComplex* temp_self = NULL;
//
//    if(*number_of_wave<=1) //if only 1 mode, it is a number, do the simple calculation
//    {
//      std::string tic_toc_mult2= tic_toc_prefix+": sigma = (T10*VL)*inv(GR_inv1)*V:'*T01 ";
//      NemoUtils::tic(tic_toc_mult2);
//      //obtain GR = inv(GR_inv1)
//      PetscMatrixParallelComplex* GR_matrix = NULL;
//      std::complex<double> GR_matrix_1(0.0,0.0);
//      if(*number_of_wave==1)
//        GR_matrix_1 = 1.0/(GR_inv1->get(0,0));
//      GR_matrix = new PetscMatrixParallelComplex(1,1,get_simulation_domain()->get_communicator() );
//      GR_matrix->set_num_owned_rows(1);
//      GR_matrix->set_num_nonzeros(0,1,0);
//      GR_matrix->allocate_memory();
//      GR_matrix->set_to_zero();
//      GR_matrix->set(0,0,GR_matrix_1);
//      GR_matrix->assemble();
//
//      PetscMatrixParallelComplex* temp1= NULL;
//      PetscMatrixParallelComplex::mult(*couplemultwave,*GR_matrix,&temp1);
//      PetscMatrixParallelComplex::mult(*temp1,wavemultcouple,&temp_self); //self-energy
//      NemoUtils::toc(tic_toc_mult2);
//      delete GR_matrix;
//      delete temp1;
//    }
//    else
//    {
//      // ------------------------------------
//      // solve a linear equation as OMEN did
//      // ------------------------------------
//      //set up the Linear solver
//
//      PetscMatrixParallelComplex temp_solution(GR_inv1->get_num_rows(),wavemultcouple.get_num_cols(),
//          get_simulation_domain()->get_communicator() );
//      temp_solution.consider_as_full(); // petsc require solution as dense
//      temp_solution.allocate_memory();
//
//      // set A, B, X -- petsc require B and X as dense!
//      {
//        LinearSolverPetscComplex solver(*GR_inv1,wavemultcouple,&temp_solution);
//
//        //prepare input options for the linear solver
//        InputOptions options_for_linear_solver;
//        string value("petsc"); //default petsc preconditioner
//        string key("solver");
//        options_for_linear_solver[key] = value;
//        solver.set_options(options_for_linear_solver);
//
//        std::string tic_toc_prefix_linear= tic_toc_prefix+": solve_linear_equation GR_inv1*X=VL'*T01";
//        NemoUtils::tic(tic_toc_prefix_linear);
//        solver.solve(); //GR_inv1*X=VL'*T01 solve X
//        NemoUtils::toc(tic_toc_prefix_linear);
//      }
//      PetscMatrixParallelComplex* temp_G=NULL;
//      temp_G=&temp_solution;
//
//      //delete wavemultcouple_temp;
//      std::string tic_toc_mult2= tic_toc_prefix+": sigma = T10*VL*X";
//      NemoUtils::tic(tic_toc_mult2);
//      PetscMatrixParallelComplex::mult(*couplemultwave,*temp_G,&temp_self); //sigma = T10*VL*X
//      NemoUtils::toc(tic_toc_mult2);
//    }
//
//    //need to copy result to a sparse matrix, for following usage
//    //the problem is petsc doesn't allow adding dense matrix to sparse matrix directly!
//    std::string tic_toc_copy1= tic_toc_prefix+": copy result";
//    NemoUtils::tic(tic_toc_copy1);
//    vector<int> result_diagonal(number_of_sub_rows,0); //result stored in subdomain
//    vector<int> result_offdiagonal(number_of_sub_rows,0);
//    for(unsigned int i=0; i<number_of_sub_rows; i++)
//    {
//      {
//        result_diagonal[i]=temp_self->get_nz_diagonal(i);
//        result_offdiagonal[i]=temp_self->get_nz_offdiagonal(i);
//      }
//    }
//    //initialize result matrix as sparse matrix
//    PropagationUtilities::transfer_matrix_initialize_temp_matrix(this,number_of_sub_rows,number_of_sub_rows,result_diagonal,result_offdiagonal,result_matrix);
//    const std::complex<double>* pointer_to_data= NULL;
//    vector<cplx> data_vector;
//    vector<int> col_index;
//    int n_nonzeros=0;
//    const int* n_col_nums=NULL;
//    for(unsigned int i=0; i<number_of_sub_rows; i++) //copy row-wise, this is most efficient way for copying elements to sparse matrix 
//    {
//      {
//        temp_self->get_row(i,&n_nonzeros,n_col_nums,pointer_to_data);
//        col_index.resize(n_nonzeros,0);
//        data_vector.resize(n_nonzeros,cplx(0.0,0.0));
//        for(int j=0; j<n_nonzeros; j++)
//        {
//          col_index[j]=n_col_nums[j];
//          data_vector[j]=pointer_to_data[j];
//        }
//        result_matrix->set(i,col_index,data_vector);
//        temp_self->store_row(i,&n_nonzeros,n_col_nums,pointer_to_data);
//      }
//    }
//    result_matrix->assemble();
//    NemoUtils::toc(tic_toc_copy1);
//    delete GR_inv1;
//    delete couplemultwave;
//    delete temp_self;
//  }
//  NemoUtils::toc(tic_toc_prefix);
//}
//

//void Self_energy::transfer_matrix_get_new_self_energy(PetscMatrixParallelComplex*& E_minus_H_matrix, PetscMatrixParallelComplex*& T_matrix,
//    unsigned int* number_of_wave, unsigned int* /*number_of_vectors_size*/, PetscMatrixParallelComplex*& phase_factor,
//    PetscMatrixParallelComplex*& Wave_matrix, PetscMatrixParallelComplex*& result_matrix,
//    PetscMatrixParallelComplex*& coupling_device)
//{
//  //solve self-energy, result store with device size -- OMEN-QTBM or RGF
//  tic_toc_name = options.get_option("tic_toc_name",get_name());
//  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::transfer_matrix_get_self_energy ";
//  NemoUtils::tic(tic_toc_prefix);
//  std::string prefix = "Self_energy(\""+get_name()+"\")::transfer_matrix_get_self_energy ";
//
//  unsigned int number_of_device_rows = coupling_device->get_num_rows(); //coupling of contact/device
//  unsigned int number_of_device_cols = number_of_device_rows;
//  if(*number_of_wave==0) //number of modes is 0, fake result with number 0
//  {
//    result_matrix = new PetscMatrixParallelComplex(number_of_device_rows,number_of_device_cols,
//        get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
//    result_matrix->set_num_owned_rows(number_of_device_rows);
//    for(unsigned int i=0; i<number_of_device_rows; i++)
//      result_matrix->set_num_nonzeros(i,1,0);
//    result_matrix->allocate_memory();
//    result_matrix->set_to_zero();
//    result_matrix->assemble();
//  }
//  else
//  {
//    PetscMatrixParallelComplex Wave = (*Wave_matrix);
//    PetscMatrixParallelComplex WaveT(Wave.get_num_cols(),Wave.get_num_rows(),get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
//    PetscMatrixParallelComplex* GR_inv_temp1 = NULL;
//    PetscMatrixParallelComplex* GR_inv1 = NULL;
//    PetscMatrixParallelComplex* GR_inv_temp2 = NULL;
//    PetscMatrixParallelComplex* GR_inv2 = NULL;
//
//    std::string tic_toc_mult1 = tic_toc_prefix+": invg = VL'*(E-H)*VL+VL'*T*VL*expmikdelta";
//    NemoUtils::tic(tic_toc_mult1);
//    (&Wave)->hermitian_transpose_matrix(WaveT,MAT_INITIAL_MATRIX);//VL'
//    PetscMatrixParallelComplex::mult(Wave,*phase_factor,&GR_inv2); //GR_inv2=VL*expmikdelta
//    PetscMatrixParallelComplex::mult(*T_matrix,*GR_inv2,&GR_inv_temp2); //GR_inv_temp2=T*VL*expmikdelta
//    PetscMatrixParallelComplex::mult(*E_minus_H_matrix,Wave,&GR_inv_temp1); //GR_inv_temp1=(E-H)*VL
//    GR_inv_temp1->add_matrix(*GR_inv_temp2,DIFFERENT_NONZERO_PATTERN); //GR_inv_temp1 = (E-H)*VL+T*VL*expmikdelta
//    PetscMatrixParallelComplex::mult(WaveT,*GR_inv_temp1,&GR_inv1); //GR_inv1=GR_inv1=VL'*(E-H)*VL+VL'*T*VL*expmikdelta
//    NemoUtils::toc(tic_toc_mult1);
//
//    PetscMatrixParallelComplex* GR_matrix_inv = NULL;
//    //get the left upper nonzeros of GR_inv1
//    vector<int> GR_rows_diagonal(*number_of_wave,0);
//    vector<int> GR_rows_offdiagonal(*number_of_wave,0);
//    for(unsigned int i=0; i<*number_of_wave; i++)
//    {
//      GR_rows_diagonal[i]=GR_inv1->get_nz_diagonal(i);
//      GR_rows_offdiagonal[i]=GR_inv1->get_nz_offdiagonal(i);
//    }
//    transfer_matrix_get_full_submatrix(*number_of_wave,*number_of_wave,0,0,GR_rows_diagonal,GR_rows_offdiagonal,GR_inv1,GR_matrix_inv);
//
//    std::string tic_toc_VLmultT= tic_toc_prefix+": VL'*T01, T10*VL ";
//    NemoUtils::tic(tic_toc_VLmultT);
//    unsigned int start_row = 0;
//    unsigned int end_row = 0;
//    unsigned int nonzero_index = 0;
//    //extract the nonzero rows from coupling matrix -- this is important for efficient solution of the linear equation
//    for(unsigned int i=0; i<number_of_device_rows; i++)
//    {
//      nonzero_index = coupling_device->get_nz_diagonal(i);
//      if(nonzero_index>0)
//      {
//        start_row = i; //starting row of nonzeros
//        break;
//      }
//    }
//    nonzero_index = 0;
//    for(unsigned int i=number_of_device_rows; i>0; i--)
//    {
//      nonzero_index = coupling_device->get_nz_diagonal(i-1);
//      if(nonzero_index>0)
//      {
//        end_row = i-1; //end row of nonzeros
//        break;
//      }
//    }
//    unsigned int number_of_sub_rows = end_row+1-start_row; //operations only to those nonzero rows; this is important for efficiency
//    vector<int> T_rows_diagonal(number_of_sub_rows,0);
//    vector<int> T_rows_offdiagonal(number_of_sub_rows,0);
//    for(unsigned int i=0; i<number_of_sub_rows; i++)
//    {
//      T_rows_diagonal[i]=coupling_device->get_nz_diagonal(i+start_row);
//      T_rows_offdiagonal[i]=coupling_device->get_nz_offdiagonal(i+start_row);
//    }
//    PetscMatrixParallelComplex* couple_temp = NULL;
//    unsigned int number_of_contact_cols = E_minus_H_matrix->get_num_cols();
//    //extract the nonzero rows and put into couple_temp
//    transfer_matrix_get_submatrix(number_of_sub_rows,number_of_contact_cols,start_row,0,T_rows_diagonal,T_rows_offdiagonal,coupling_device,couple_temp);
//    PetscMatrixParallelComplex small_coupling_device(*couple_temp);
//    delete couple_temp;
//
//    PetscMatrixParallelComplex* couplemultwave = NULL;
//    PetscMatrixParallelComplex::mult(small_coupling_device,Wave,
//                                     &couplemultwave); //T10*VL // petsc assumes sparse*dense = dense, even though it has tons of zeros!
//    PetscMatrixParallelComplex wavemultcouple(couplemultwave->get_num_cols(),couplemultwave->get_num_rows(),
//        get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
//    couplemultwave->hermitian_transpose_matrix(wavemultcouple,MAT_INITIAL_MATRIX);//(T10*VL)' = VL'*T01
//    NemoUtils::toc(tic_toc_VLmultT);
//
//    PetscMatrixParallelComplex* temp_self = NULL;
//
//    if(*number_of_wave<=1) //if only 1 mode, use simple calculation
//    {
//      std::string tic_toc_mult2= tic_toc_prefix+": sigma = (T10*VL)*inv(GR_inv1)*V:'*T01 ";
//      NemoUtils::tic(tic_toc_mult2);
//      //obtain GR = inv(GR_inv1)
//      PetscMatrixParallelComplex* GR_matrix = NULL;
//      std::complex<double> GR_matrix_1(0.0,0.0);
//      if(*number_of_wave==1)
//        GR_matrix_1 = 1.0/(GR_matrix_inv->get(0,0));
//      GR_matrix = new PetscMatrixParallelComplex(1,1,get_simulation_domain()->get_communicator() /*holder.geometry_communicator*/);
//      GR_matrix->set_num_owned_rows(1);
//      GR_matrix->set_num_nonzeros(0,1,0);
//      GR_matrix->allocate_memory();
//      GR_matrix->set_to_zero();
//      GR_matrix->set(0,0,GR_matrix_1);
//      GR_matrix->assemble();
//
//      PetscMatrixParallelComplex* temp1= NULL;
//      PetscMatrixParallelComplex::mult(*couplemultwave,*GR_matrix,&temp1);
//      PetscMatrixParallelComplex::mult(*temp1,wavemultcouple,&temp_self);
//      NemoUtils::toc(tic_toc_mult2);
//      delete GR_matrix;
//      delete temp1;
//    }
//    else
//    {
//      // ------------------------------------
//      // solve a linear equation as OMEN did
//      // ------------------------------------
//      //set up the Linear solver
//      PetscMatrixParallelComplex temp_solution(GR_matrix_inv->get_num_rows(),wavemultcouple.get_num_cols(),
//          get_simulation_domain()->get_communicator() );
//      temp_solution.consider_as_full(); // petsc require solution as dense
//      temp_solution.allocate_memory();
//
//      // set A, B, X -- petsc require B and X as dense!
//      LinearSolverPetscComplex solver(*GR_matrix_inv,wavemultcouple,&temp_solution);
//
//      //prepare input options for the linear solver
//      InputOptions options_for_linear_solver;
//      string value("petsc");
//      string key("solver");
//      options_for_linear_solver[key] = value;
//      solver.set_options(options_for_linear_solver);
//
//      std::string tic_toc_prefix_linear= tic_toc_prefix+": solve_linear_equation GR_inv1*X=VL'*T01";
//      NemoUtils::tic(tic_toc_prefix_linear);
//      solver.solve(); // GR_inv1*X=VL'*T01 solve X
//      NemoUtils::toc(tic_toc_prefix_linear);
//
//      PetscMatrixParallelComplex* temp_G=NULL;
//      temp_G=&temp_solution;
//      std::string tic_toc_mult2= tic_toc_prefix+": sigma = T10*VL*X";
//      NemoUtils::tic(tic_toc_mult2);
//      PetscMatrixParallelComplex::mult(*couplemultwave,*temp_G,&temp_self); //sigma = T10*VL*X
//      NemoUtils::toc(tic_toc_mult2);
//    }
//    
//    //copy result to sparse matrix since petsc doesn't allow adding dense matrix to sparse matrix!
//    std::string tic_toc_copy1= tic_toc_prefix+": copy result";
//    NemoUtils::tic(tic_toc_copy1);
//    vector<int> result_diagonal(number_of_device_rows,0);
//    vector<int> result_offdiagonal(number_of_device_rows,0);
//    for(unsigned int i=start_row; i<=end_row; i++)
//    {
//      result_diagonal[i]=temp_self->get_nz_diagonal(i-start_row);
//      result_offdiagonal[i]=temp_self->get_nz_offdiagonal(i-start_row);
//    }
//    //the result matrix is defined in device dimension for QTBM requirement
//    transfer_matrix_initialize_temp_matrix(number_of_device_rows,number_of_device_rows,result_diagonal,result_offdiagonal,
//                                           result_matrix);
//    const std::complex<double>* pointer_to_data= NULL;
//    vector<cplx> data_vector;
//    vector<int> col_index;
//    int n_nonzeros=0;
//    const int* n_col_nums=NULL;
//    for(unsigned int i=start_row; i<=end_row; i++) //row-wise, only for those nonzero rows
//    {
//      temp_self->get_row(i-start_row,&n_nonzeros,n_col_nums,pointer_to_data);
//      col_index.resize(n_nonzeros,0);
//      data_vector.resize(n_nonzeros,cplx(0.0,0.0));
//      for(int j=0; j<n_nonzeros; j++)
//      {
//        col_index[j]=n_col_nums[j]+start_row;
//        data_vector[j]=pointer_to_data[j];
//      }
//      result_matrix->set(i,col_index,data_vector);
//      temp_self->store_row(i-start_row,&n_nonzeros,n_col_nums,pointer_to_data);
//    }
//    result_matrix->assemble();
//    NemoUtils::toc(tic_toc_copy1);
//    delete GR_inv_temp1;
//    delete GR_inv1;
//    delete GR_inv_temp2;
//    delete GR_inv2;
//    delete GR_matrix_inv;
//    delete couplemultwave;
//    delete temp_self;
//  }
//  NemoUtils::toc(tic_toc_prefix);
//}
//void Self_energy::transfer_matrix_get_energy(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum, std::complex<double>* energy)
//{
// //obtain energy from momentum
//  tic_toc_name = options.get_option("tic_toc_name",get_name());
//  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::transfer_matrix_get_energy ";
//  NemoUtils::tic(tic_toc_prefix);
//  //figure out which index of momentum is the energy
//  bool complex_energy=false;
//  std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
//  std::string energy_name=std::string("");
//  for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
//    if(momentum_name_it->second==NemoPhys::Energy)
//      energy_name=momentum_name_it->first;
//    else if (momentum_name_it->second==NemoPhys::Complex_energy)
//    {
//      energy_name=momentum_name_it->first;
//      complex_energy=true;
//    }
//  unsigned int energy_index=0;
//  for (unsigned int i=0; i<output_Propagator->momentum_mesh_names.size(); i++)
//    if(output_Propagator->momentum_mesh_names[i].find(std::string("energy"))!=std::string::npos)
//      energy_index=i;
//  //-------------------------------------------------------
//  NemoMeshPoint enery_point=momentum[energy_index];
//  if(!complex_energy)
//    *energy=std::complex<double> (enery_point.get_x(),0.0);
//  else
//    *energy=std::complex<double> (enery_point.get_x(),enery_point.get_y());
//  msg<<"Selfenergy(\""+this->get_name()+"\")::transfer_matrix_leads() for energy"<<*energy<<std::endl;
//  NemoUtils::toc(tic_toc_prefix);
//}

//void Self_energy::transfer_matrix_leads(Propagator*& output_Propagator,const Domain* neighbor_domain, const std::vector<NemoMeshPoint>& momentum,
//                                        PetscMatrixParallelComplex*& result)
//{
//  //main function for transfer matrix method
//  //i.prepare matrix for eigenvalue problem
//  //ii.solve eigenvalue problem for eigenvalues and modes
//  //iii.solve self-energy
//  //ref: PRB 74, 205323 (2006)
//  tic_toc_name = options.get_option("tic_toc_name",get_name());
//  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::transfer_matrix_leads ";
//  NemoUtils::tic(tic_toc_prefix);
//  std::string prefix = "Self_energy(\""+this->get_name()+"\")::transfer_matrix_leads()";
//  msg<<prefix<<std::endl;
//
//  std::string time_for_transfer_matrix_leads = get_name()+"_transfer_matrix_leads()";
//  NemoUtils::tic(time_for_transfer_matrix_leads);
//
//  std::string temp_name2;
//  const std::vector<NemoMeshPoint>* temp_vector_pointer=&(momentum);
//  translate_momentum_vector(temp_vector_pointer,temp_name2);
//
//  //set_valley(writeable_Propagators.begin()->second, momentum, Hamilton_Constructor); //
//
//  //process flow: e.g. 4 atomic layers described in PRB 74, 205323 (2006)
//  //1. Hamiltonian Hs and coupling_Hamiltonian Ts for atomic layer
//  //H=|H00 H01 0     0|; T=|0 0 0 H43|; T'=|0   0 0 0|
//  //  |H10 H11 H12   0|    |0 0 0 0  |     |0   0 0 0|
//  //  |0   H21 H22 H23|    |0 0 0 0  |     |0   0 0 0|
//  //  |0   0   H32 H33|    |0 0 0 0  |     |H34 0 0 0|
//  Domain* neighbor_domain2=NULL;
//  bool new_self_energy = false;
//  std::string retarded_lead_method=options.get_option("retarded_lead_method", std::string("direct_iterations"));
//  if(retarded_lead_method.find("avoid_surface_greens_function")!=std::string::npos) new_self_energy = true; //for OMEN-QTBM this is used
//
//  //1a. get Hamiltonian and coupling term (and DOF) for each subdomain 0~3, 3 is always labelled as the one attached to device
//  PetscMatrixParallelComplex* D00 = NULL;
//  PetscMatrixParallelComplex* H01 = NULL;
//  PetscMatrixParallelComplex* S00 = NULL;
//  PetscMatrixParallelComplex* S01 = NULL;
//
//  const DOFmapInterface* subdomain_DOF0;
//  DOFmapInterface* subdomain_coupling_DOF0=NULL;
//  PropagationUtilities::transfer_matrix_get_Hamiltonian(this,std::string("0"),output_Propagator,neighbor_domain2, momentum, D00, H01,S00, S01, subdomain_DOF0, subdomain_coupling_DOF0);
//  delete subdomain_coupling_DOF0;
//
//  PetscMatrixParallelComplex* D11 = NULL;
//  PetscMatrixParallelComplex* H12 = NULL;
//  PetscMatrixParallelComplex* S11 = NULL;
//  PetscMatrixParallelComplex* S12 = NULL;
//
//  const DOFmapInterface* subdomain_DOF1;
//  DOFmapInterface* subdomain_coupling_DOF1=NULL;
//  PropagationUtilities::transfer_matrix_get_Hamiltonian(this,std::string("1"),output_Propagator,neighbor_domain2, momentum, D11, H12, S11, S12, subdomain_DOF1, subdomain_coupling_DOF1);
//  delete subdomain_coupling_DOF1;
//
//  PetscMatrixParallelComplex* D22 = NULL;
//  PetscMatrixParallelComplex* H23 = NULL;
//  PetscMatrixParallelComplex* S22 = NULL;
//  PetscMatrixParallelComplex* S23 = NULL;
//
//  const DOFmapInterface* subdomain_DOF2;
//  DOFmapInterface* subdomain_coupling_DOF2=NULL;
//  PropagationUtilities::transfer_matrix_get_Hamiltonian(this,std::string("2"),output_Propagator,neighbor_domain2, momentum, D22, H23,S22, S23, subdomain_DOF2, subdomain_coupling_DOF2);
//  delete subdomain_coupling_DOF2;
//
//
//  PetscMatrixParallelComplex* D33 = NULL;
//  PetscMatrixParallelComplex* H34 = NULL;
//  PetscMatrixParallelComplex* S33 = NULL;
//  PetscMatrixParallelComplex* S34 = NULL;
//
//  const DOFmapInterface* subdomain_DOF3;
//  DOFmapInterface* subdomain_coupling_DOF3=NULL;
//
//  PetscMatrixParallelComplex* Dtmp = NULL;
//  PetscMatrixParallelComplex* Htmp = NULL;
//  PetscMatrixParallelComplex* Stmp = NULL;
//  PetscMatrixParallelComplex* SStmp = NULL;
//
//  const DOFmapInterface* subdomain_DOF4;
//  DOFmapInterface* subdomain_coupling_DOF4=NULL;
//
//  if(new_self_energy) //for OMEN type QTBM
//  {
//    PropagationUtilities::transfer_matrix_get_Hamiltonian(this,std::string("3"),output_Propagator,neighbor_domain2, momentum, D33, Htmp,S33, Stmp, subdomain_DOF3, subdomain_coupling_DOF3);
//    PropagationUtilities::transfer_matrix_get_Hamiltonian(this,std::string("4"),output_Propagator,neighbor_domain2, momentum, Dtmp, H34, SStmp, S34, subdomain_DOF4, subdomain_coupling_DOF4);
//    delete Htmp;
//  }
//  else //for RGF/NEGF
//    PropagationUtilities::transfer_matrix_get_Hamiltonian(this,std::string("3"),output_Propagator,neighbor_domain2, momentum, D33, H34, S33, S34, subdomain_DOF3, subdomain_coupling_DOF3);
//
//  delete subdomain_coupling_DOF3;
//  delete subdomain_coupling_DOF4;
//
//  //Hermitian transpose of coupling matrix for subdomains
//  PetscMatrixParallelComplex* H10 = new PetscMatrixParallelComplex(*H01);
//  H10->hermitian_transpose_matrix(*H10,MAT_REUSE_MATRIX);
//  PetscMatrixParallelComplex* H21 = new PetscMatrixParallelComplex(*H12);
//  H21->hermitian_transpose_matrix(*H21,MAT_REUSE_MATRIX);
//  PetscMatrixParallelComplex* H32 = new PetscMatrixParallelComplex(*H23);
//  H32->hermitian_transpose_matrix(*H32,MAT_REUSE_MATRIX);
//  PetscMatrixParallelComplex* H43 = new PetscMatrixParallelComplex(*H34);
//  H43->hermitian_transpose_matrix(*H43,MAT_REUSE_MATRIX);
//
//  //1b. obtain matrix information
//  //global rows
//  unsigned int number_of_00_rows = D00->get_num_rows();
//  unsigned int number_of_00_cols = D00->get_num_cols();
//  unsigned int number_of_11_rows = D11->get_num_rows();
//  unsigned int number_of_11_cols = D11->get_num_cols();
//  unsigned int number_of_22_rows = D22->get_num_rows();
//  unsigned int number_of_22_cols = D22->get_num_cols();
//  unsigned int number_of_33_rows = D33->get_num_rows();
//  unsigned int number_of_33_cols = D33->get_num_cols();
//
//  //local rows
//  unsigned int number_of_00_local_rows = D00->get_num_owned_rows();
//  unsigned int number_of_11_local_rows = D11->get_num_owned_rows();
//  unsigned int number_of_22_local_rows = D22->get_num_owned_rows();
//  unsigned int number_of_33_local_rows = D33->get_num_owned_rows();
//  int start_own_00_rows;
//  int end_own_00_rows_p1;
//  D00->get_ownership_range(start_own_00_rows,end_own_00_rows_p1);
//  int start_own_11_rows;
//  int end_own_11_rows_p1;
//  D11->get_ownership_range(start_own_11_rows,end_own_11_rows_p1);
//  int start_own_22_rows;
//  int end_own_22_rows_p1;
//  D22->get_ownership_range(start_own_22_rows,end_own_22_rows_p1);
//  int start_own_33_rows;
//  int end_own_33_rows_p1;
//  D33->get_ownership_range(start_own_33_rows,end_own_33_rows_p1);
//
//  //get information of coupling Hamiltonian
//  int start_own_01_rows;
//  int end_own_01_rows_p1;
//  H01->get_ownership_range(start_own_01_rows,end_own_01_rows_p1);
//  int start_own_12_rows;
//  int end_own_12_rows_p1;
//  H12->get_ownership_range(start_own_12_rows,end_own_12_rows_p1);
//  int start_own_23_rows;
//  int end_own_23_rows_p1;
//  H23->get_ownership_range(start_own_23_rows,end_own_23_rows_p1);
//  int start_own_34_rows;
//  int end_own_34_rows_p1;
//  H34->get_ownership_range(start_own_34_rows,end_own_34_rows_p1);
//
//  //1c. set local nonzeros and nonlocal nonzeros
//  vector<int> D01_rows_diagonal(number_of_00_local_rows,0);
//  vector<int> D01_rows_offdiagonal(number_of_00_local_rows,0);
//  for(unsigned int i=0; i<number_of_00_local_rows; i++)
//  {
//    D01_rows_diagonal[i]=H01->get_nz_diagonal_by_global_index(i+start_own_01_rows);
//    D01_rows_offdiagonal[i]=H01->get_nz_offdiagonal_by_global_index(i+start_own_01_rows);
//  }
//  vector<int> D12_rows_diagonal(number_of_11_local_rows,0);
//  vector<int> D12_rows_offdiagonal(number_of_11_local_rows,0);
//  for(unsigned int i=0; i<number_of_11_local_rows; i++)
//  {
//    D12_rows_diagonal[i]=H12->get_nz_diagonal_by_global_index(i+start_own_12_rows);
//    D12_rows_offdiagonal[i]=H12->get_nz_offdiagonal_by_global_index(i+start_own_12_rows);
//  }
//  vector<int> D23_rows_diagonal(number_of_22_local_rows,0);
//  vector<int> D23_rows_offdiagonal(number_of_22_local_rows,0);
//  for(unsigned int i=0; i<number_of_22_local_rows; i++)
//  {
//    D23_rows_diagonal[i]=H23->get_nz_diagonal_by_global_index(i+start_own_23_rows);
//    D23_rows_offdiagonal[i]=H23->get_nz_offdiagonal_by_global_index(i+start_own_23_rows);
//  }
//  vector<int> D34_rows_diagonal(number_of_33_local_rows,0);
//  vector<int> D34_rows_offdiagonal(number_of_33_local_rows,0);
//  for(unsigned int i=0; i<number_of_33_local_rows; i++)
//  {
//    D34_rows_diagonal[i]=H34->get_nz_diagonal_by_global_index(i+start_own_34_rows);
//    D34_rows_offdiagonal[i]=H34->get_nz_offdiagonal_by_global_index(i+start_own_34_rows);
//  }
//  vector<int> D10_rows_diagonal(number_of_11_local_rows,0);
//  vector<int> D10_rows_offdiagonal(number_of_11_local_rows,0);
//  for(unsigned int i=0; i<number_of_11_local_rows; i++)
//  {
//    D10_rows_diagonal[i]=H10->get_nz_diagonal(i+number_of_00_local_rows);
//    D10_rows_offdiagonal[i]=H10->get_nz_offdiagonal(i+number_of_00_local_rows);
//  }
//  vector<int> D21_rows_diagonal(number_of_22_local_rows,0);
//  vector<int> D21_rows_offdiagonal(number_of_22_local_rows,0);
//  for(unsigned int i=0; i<number_of_22_local_rows; i++)
//  {
//    D21_rows_diagonal[i]=H21->get_nz_diagonal(i+number_of_11_local_rows);
//    D21_rows_offdiagonal[i]=H21->get_nz_offdiagonal(i+number_of_11_local_rows);
//  }
//  vector<int> D32_rows_diagonal(number_of_33_local_rows,0);
//  vector<int> D32_rows_offdiagonal(number_of_33_local_rows,0);
//  for(unsigned int i=0; i<number_of_33_local_rows; i++)
//  {
//    D32_rows_diagonal[i]=H32->get_nz_diagonal(i+number_of_22_local_rows);
//    D32_rows_offdiagonal[i]=H32->get_nz_offdiagonal(i+number_of_22_local_rows);
//  }
//  vector<int> D43_rows_diagonal(number_of_00_local_rows,0);
//  vector<int> D43_rows_offdiagonal(number_of_00_local_rows,0);
//  for(unsigned int i=0; i<number_of_00_local_rows; i++)
//  {
//    D43_rows_diagonal[i]=H43->get_nz_diagonal(i+number_of_33_local_rows);
//    D43_rows_offdiagonal[i]=H43->get_nz_offdiagonal(i+number_of_33_local_rows);
//  }
//
//  //1d. obtain the submatrix -- Dij is the upper right block of super-coupling matrix Hij
//  PetscMatrixParallelComplex* D01 = NULL;
//  PetscMatrixParallelComplex* D12 = NULL;
//  PetscMatrixParallelComplex* D23 = NULL;
//  PetscMatrixParallelComplex* D34 = NULL;
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_00_rows,number_of_11_cols,0,number_of_00_cols,D01_rows_diagonal,D01_rows_offdiagonal,H01,D01);
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_11_rows,number_of_22_cols,0,number_of_11_cols,D12_rows_diagonal,D12_rows_offdiagonal,H12,D12);
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_22_rows,number_of_33_cols,0,number_of_22_cols,D23_rows_diagonal,D23_rows_offdiagonal,H23,D23);
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_33_rows,number_of_00_cols,0,number_of_33_cols,D34_rows_diagonal,D34_rows_offdiagonal,H34,D34);
//
//  PetscMatrixParallelComplex* D10 = NULL;
//  PetscMatrixParallelComplex* D21 = NULL;
//  PetscMatrixParallelComplex* D32 = NULL;
//  PetscMatrixParallelComplex* D43 = NULL;
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_11_rows,number_of_00_cols,number_of_00_rows,0,D10_rows_diagonal,D10_rows_offdiagonal,H10,D10);
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_22_rows,number_of_11_cols,number_of_11_rows,0,D21_rows_diagonal,D21_rows_offdiagonal,H21,D21);
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_33_rows,number_of_22_cols,number_of_22_rows,0,D32_rows_diagonal,D32_rows_offdiagonal,H32,D32);
//  PropagationUtilities::transfer_matrix_get_submatrix(this,number_of_00_rows,number_of_33_cols,number_of_33_rows,0,D43_rows_diagonal,D43_rows_offdiagonal,H43,D43);
//  delete H01;
//  delete H12;
//  delete H23;
//  delete H34;
//  delete H10;
//  delete H21;
//  delete H32;
//  delete H43;
//
//  //2. get the full Hamiltonian for the lead domain
//  //E-H=|D00 D01 0     0|; T=|0 0 0 D43|
//  //    |D10 D11 D12   0|    |0 0 0 0  |
//  //    |0   D21 D22 D23|    |0 0 0 0  |
//  //    |0   0   D32 D33|    |0 0 0 0  |
//  std::complex<double> energy;
//  PropagationUtilities::transfer_matrix_get_energy(this,output_Propagator, momentum, &energy);
//  double eta=options.get_option("constant_lead_eta",1e-12);//small i*eta adding to energy for van Hove singularity
//  //differentiate the fermion and boson in the transfer matrix method.
//  if((output_Propagator->get_name()).find("Fermion")!=std::string::npos)
//  {
//    energy+=cplx(0.0,1.0)*eta;
//    *D00 *=std::complex<double>(-1.0,0.0); //Dii=E-Hii
//    D00->matrix_diagonal_shift(energy);
//    *D11 *=std::complex<double>(-1.0,0.0);
//    D11->matrix_diagonal_shift(energy);
//    *D22 *=std::complex<double>(-1.0,0.0);
//    D22->matrix_diagonal_shift(energy);
//    *D33 *=std::complex<double>(-1.0,0.0);
//    D33->matrix_diagonal_shift(energy);
//  }
//  else if ((output_Propagator->get_name()).find("Boson")!=std::string::npos)
//  {
//    energy=energy*energy+cplx(0.0,1.0)*eta;
//    *D00 *=std::complex<double>(-1.0,0.0);
//    D00->matrix_diagonal_shift(energy);
//    *D11 *=std::complex<double>(-1.0,0.0);
//    D11->matrix_diagonal_shift(energy);
//    *D22 *=std::complex<double>(-1.0,0.0);
//    D22->matrix_diagonal_shift(energy);
//    *D33 *=std::complex<double>(-1.0,0.0);
//    D33->matrix_diagonal_shift(energy);
//  }
//  *D01 *=std::complex<double>(-1.0,0.0); //Dij=-Hij
//  *D10 *=std::complex<double>(-1.0,0.0);
//  *D12 *=std::complex<double>(-1.0,0.0);
//  *D21 *=std::complex<double>(-1.0,0.0);
//  *D23 *=std::complex<double>(-1.0,0.0);
//  *D32 *=std::complex<double>(-1.0,0.0);
//  *D34 *=std::complex<double>(-1.0,0.0);
//  *D43 *=std::complex<double>(-1.0,0.0);
//
//  PetscMatrixParallelComplex* E_minus_H_matrix = NULL; //E-H matrix
//  PetscMatrixParallelComplex* T43 = NULL; //T matrix
//  PetscMatrixParallelComplex* T34 = NULL; //T' matrix
//
//  //construct the E-H matrix for the contact domain
//  unsigned int number_of_EmH_rows = number_of_00_rows+number_of_11_rows+number_of_22_rows+number_of_33_rows;
//  unsigned int number_of_EmH_cols = number_of_00_cols+number_of_11_cols+number_of_22_cols+number_of_33_cols;
//  unsigned int number_of_EmH_local_rows = number_of_00_local_rows+number_of_11_local_rows+number_of_22_local_rows+number_of_33_local_rows;
//  unsigned int offset_rows = start_own_00_rows+start_own_11_rows+start_own_22_rows+start_own_33_rows;
//
//  //-------------------------------------------------------
//  vector<int> EmH_rows_diagonal(number_of_EmH_local_rows,0);
//  vector<int> EmH_rows_offdiagonal(number_of_EmH_local_rows,0);
//  for(unsigned int i=0; i<number_of_00_local_rows; i++)
//  {
//    EmH_rows_diagonal[i]=D00->get_nz_diagonal(i)+D01->get_nz_diagonal(i);
//    EmH_rows_offdiagonal[i]=D00->get_nz_offdiagonal(i)+D01->get_nz_offdiagonal(i);
//  }
//  for(unsigned int i=0; i<number_of_11_local_rows; i++)
//  {
//    EmH_rows_diagonal[i+number_of_00_local_rows]=D11->get_nz_diagonal(i)+
//        D12->get_nz_diagonal(i)+D10->get_nz_diagonal(i);
//    EmH_rows_offdiagonal[i+number_of_00_local_rows]=D11->get_nz_offdiagonal(i)+
//        D12->get_nz_offdiagonal(i)+D10->get_nz_offdiagonal(i);
//  }
//  for(unsigned int i=0; i<number_of_22_local_rows; i++)
//  {
//    EmH_rows_diagonal[i+number_of_00_local_rows+number_of_11_local_rows]=D22->get_nz_diagonal(i)+
//        D23->get_nz_diagonal(i)+D21->get_nz_diagonal(i);
//    EmH_rows_offdiagonal[i+number_of_00_local_rows+number_of_11_local_rows]=D22->get_nz_offdiagonal(i)+
//        D23->get_nz_offdiagonal(i)+D21->get_nz_offdiagonal(i);
//  }
//  for(unsigned int i=0; i<number_of_33_local_rows; i++)
//  {
//    EmH_rows_diagonal[i+number_of_EmH_local_rows-number_of_33_local_rows]=D33->get_nz_diagonal(i)+
//        D32->get_nz_diagonal(i);
//    EmH_rows_offdiagonal[i+number_of_EmH_local_rows-number_of_33_local_rows]=D33->get_nz_offdiagonal(i)+
//        D32->get_nz_offdiagonal(i);
//  }
//
//  vector<int> T43_rows_diagonal(number_of_EmH_local_rows,0);
//  vector<int> T43_rows_offdiagonal(number_of_EmH_local_rows,0);
//  for(unsigned int i=0; i<number_of_00_local_rows; i++)
//  {
//    T43_rows_diagonal[i]=D43->get_nz_diagonal(i);
//    T43_rows_offdiagonal[i]=D43->get_nz_offdiagonal(i);
//  }
//
//  //set the sparsity pattern
//  PropagationUtilities::transfer_matrix_initialize_temp_matrix(this,number_of_EmH_rows,number_of_EmH_cols,EmH_rows_diagonal,EmH_rows_offdiagonal,E_minus_H_matrix);
//
//  //fill E-H elements from the sub blocks 0~3
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows,0,0,D00,E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows,number_of_00_cols,0,D01,E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_00_local_rows,0,0,D10,E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_00_local_rows,number_of_00_cols,0,D11,E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_00_local_rows,number_of_00_cols+number_of_11_cols,0,D12,E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_00_local_rows+number_of_11_local_rows,number_of_00_cols,0,D21,E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_00_local_rows+number_of_11_local_rows,number_of_00_cols+number_of_11_cols,0,D22,
//                                      E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_00_local_rows+number_of_11_local_rows,number_of_EmH_cols-number_of_33_cols,0,D23,
//                                      E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,number_of_00_cols+number_of_11_cols,0,D32,
//                                      E_minus_H_matrix);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows+number_of_EmH_local_rows-number_of_33_local_rows,number_of_EmH_cols-number_of_33_cols,0,D33,
//                                      E_minus_H_matrix);
//  E_minus_H_matrix->assemble();
//
//  //fill the elements for T matrix
//  PropagationUtilities::transfer_matrix_initialize_temp_matrix(this,number_of_EmH_rows,number_of_EmH_cols,T43_rows_diagonal,T43_rows_offdiagonal,T43);
//  PropagationUtilities::transfer_matrix_set_matrix_elements(this,offset_rows,number_of_EmH_cols-number_of_33_cols,0,D43,T43);
//  T43->assemble();
//  T34 = new PetscMatrixParallelComplex(*T43);
//  T34->hermitian_transpose_matrix(*T34,MAT_REUSE_MATRIX);
//
//  bool LRA=options.get_option("LRA", bool(false)); //LRA in contact used or not
//  double number_of_set_num_ratio=options.get_option("ratio_of_eigenvalues_to_be_solved",double(1.0)); //LRA ratio
//  double shift=-0.5; //real part of eigenvalues =-0.5eV is for the propagating modes
//
//  unsigned int number_of_eigenvalues; //number of eigenvalues that are solved and stored
//  unsigned int number_of_vectors_size; //dimension of contact modes
//  unsigned int number_of_wave_left; //number of out of device modes
//  unsigned int number_of_wave_right; //number of into device modes
//
//  // from step 3 to step 5 we customize the code for real-type matrix and complex-type matrix
//  bool wireNonSO=options.get_option("wire_nonSO_basis", bool(false)); //wire without spin orbit coupling in the tb model, matrix is real
//  if(wireNonSO)
//  {
//    //if matrix is real
//    //3. derivation of eigenvalue problem
//    //HP=|D00 D01 0   D43|; P=|0 0 0   D43|; M=inv(HP)*P
//    //   |D10 D11 D12   0|    |0 0 0   0  |
//    //   |0   D21 D22   0|    |0 0 0   0  |
//    //   |D34 0   D32 D33|    |0 0 D32 D33|
//    PetscMatrixParallel<double>* M1_matrix = NULL; //upper right block of M
//    PetscMatrixParallel<double>* M2_matrix = NULL; //lower right block of M
//
//    //obtain M matrix for eigenproblem
//    PropagationUtilities::transfer_matrix_get_M_matrix_double(this,LRA,D00,D11,D22,D33,D32,D43,E_minus_H_matrix,T34,T43,M1_matrix,M2_matrix);
//    delete D01;
//    delete D10;
//    delete D12;
//    delete D21;
//    delete D23;
//    delete D34;
//    delete D32;
//    delete D43;
//    //Hack to avoid memory leaks for Phonon transport"
//    //reason: StrainVFF solver creates the "Hamiltonian", i.e. D11, D22 etc under the assumption
//    //that the get_data calling solver deletes it afterwards.
//    if(Hamilton_Constructor->get_type()=="VFFStrain")
//    {
//      delete D11;
//      D11=NULL;
//      delete D22;
//      D22=NULL;
//      delete D33;
//      D33=NULL;
//    }
//
//    //4. solve the eigenvalue problem (or part of it if use LRA)
//    int number_of_set_num=M2_matrix->get_num_rows();
//    number_of_set_num*=number_of_set_num_ratio;
//    vector<std::complex<double> > M_values; //eigenvalues
//    vector< PetscVectorNemo<double> > M_vectors_real; //real part of eigenvector
//    vector< PetscVectorNemo<double> > M_vectors_imag; //imag part of eigenvector
//
//    PropagationUtilities::transfer_matrix_solve_eigenvalues_double(this,M2_matrix,LRA,shift,number_of_set_num,M_values,
//        M_vectors_real,M_vectors_imag,&number_of_eigenvalues,&number_of_vectors_size);
//
//    //5. find out which waves are needed
//    PropagationUtilities::transfer_matrix_get_wave_direction_double(this,M1_matrix,T43,M_values,M_vectors_real,M_vectors_imag,&number_of_eigenvalues,
//        &number_of_vectors_size,&number_of_wave_left,&number_of_wave_right);
//    delete M1_matrix;
//    delete M2_matrix;
//  }
//  else
//  {
//    //if matrix is complex
//    //3. derivation of eigenvalue problem
//    //HP=|D00 D01 0   D43|; P=|0 0 0   D43|; M=inv(HP)*P
//    //   |D10 D11 D12   0|    |0 0 0   0  |
//    //   |0   D21 D22   0|    |0 0 0   0  |
//    //   |D34 0   D32 D33|    |0 0 D32 D33|
//    PetscMatrixParallelComplex* M1_matrix = NULL; //upper right block of M
//    PetscMatrixParallelComplex* M2_matrix = NULL; //lower right block of M
//
//    //obtain M matrix for eigenproblem
//    PropagationUtilities::transfer_matrix_get_M_matrix(this,LRA,D00,D11,D22,D33,D32,D43,E_minus_H_matrix,T34,T43,M1_matrix,M2_matrix);
//    delete D01;
//    delete D10;
//    delete D12;
//    delete D21;
//    delete D23;
//    delete D34;
//    delete D32;
//    delete D43;
//
//    //4. solve the eigenvalue problem (or part of it)
//    int number_of_set_num=M2_matrix->get_num_rows();
//    number_of_set_num*=number_of_set_num_ratio;
//    vector<std::complex<double> > M_values; //eigenvalues
//    vector< vector< std::complex<double> > > M_vectors_temp; //eigenvectors
//
//    PropagationUtilities::transfer_matrix_solve_eigenvalues(this,M2_matrix,LRA,shift,number_of_set_num,M_values,
//                                      M_vectors_temp,&number_of_eigenvalues,&number_of_vectors_size);
//
//    //5. find out which waves are needed
//    PropagationUtilities::transfer_matrix_get_wave_direction(this,M1_matrix,T43,M_values,M_vectors_temp,&number_of_eigenvalues,
//                                       &number_of_vectors_size,&number_of_wave_left,&number_of_wave_right);
//    delete M1_matrix;
//    delete M2_matrix;
//  }
//
//  //6. get the self energy
//  if(new_self_energy) //OMEN-type QTBM, self-energy with the same size as device domain
//  { 
//    //std::set<unsigned int> Hamilton_momentum_indices;
//    //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
//    //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
//    //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
//    //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=momentum[*(pointer_to_Hamilton_momentum_indices->begin())];
//
//    PetscMatrixParallelComplex* coupling_Hamiltonian=NULL;
//    DOFmapInterface* coupling_DOFmap=NULL;
//    const DOFmapInterface& device_DOFmap = Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain());
//
//    NemoUtils::tic(tic_toc_prefix + "obtain Hamiltonian");
//    std::vector<NemoMeshPoint> sorted_momentum;
//    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,momentum_mesh_types,Hamilton_Constructor);
//    Hamilton_Constructor->get_data(std::string("Hamiltonian"),sorted_momentum, neighbor_domain, coupling_Hamiltonian, coupling_DOFmap);
//    delete coupling_DOFmap;
//    NemoUtils::toc(tic_toc_prefix + "obtain Hamiltonian");
//
//    // get the Hamilton constructor of the neighboring domain
//    Simulation* neighbor_HamiltonConstructor =  NULL;
//    std::string neighbor_Hamilton_constructor_name;
//    std::string variable_name="Hamilton_constructor_"+neighbor_domain->get_name();
//    InputOptions& opt = Hamilton_Constructor->get_reference_to_options();
//    if(opt.check_option(variable_name))
//      neighbor_Hamilton_constructor_name=opt.get_option(variable_name,std::string(""));
//    else
//      throw std::invalid_argument("Schroedinger("+this->get_name()+")::assemble_hamiltonian: define \""+variable_name+"\"\n");
//    neighbor_HamiltonConstructor = this->find_simulation(neighbor_Hamilton_constructor_name);
//    NEMO_ASSERT(neighbor_HamiltonConstructor!=NULL,
//                "Schroedinger("+this->get_name()+")::assemble_hamiltonian: Simulation \""+neighbor_Hamilton_constructor_name+"\" has not been found!\n");
//    PetscMatrixParallelComplex* coupling_Hamiltonian_lead=NULL;
//    Domain* neighbor_domain_lead = this->get_simulation_domain();
//    DOFmapInterface* coupling_DOFmap_lead=NULL;
//
//    // (to set multivalley effective mass Hamiltonian)
//    //set_valley(writeable_Propagators.begin()->second, momentum, neighbor_HamiltonConstructor);
//    sorted_momentum.clear();
//    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,momentum_mesh_types,neighbor_HamiltonConstructor);
//    neighbor_HamiltonConstructor->get_data(std::string("Hamiltonian"),sorted_momentum, neighbor_domain_lead, coupling_Hamiltonian_lead,
//                                           coupling_DOFmap_lead);
//    
//
//
//    NemoUtils::tic(tic_toc_prefix + "coupling_Hamiltonian_get_ownership_range");
//    //construct the domain3=domain1+domain2 matrices
//    unsigned int number_of_super_rows = coupling_Hamiltonian->get_num_rows();
//    unsigned int number_of_super_cols = coupling_Hamiltonian->get_num_cols();
//    int start_own_super_rows;
//    int end_own_super_rows_p1;
//    coupling_Hamiltonian->get_ownership_range(start_own_super_rows,end_own_super_rows_p1);
//    NemoUtils::toc(tic_toc_prefix + "coupling_Hamiltonian_get_ownership_range");
//
//    //these are the dimensions of the device matrices:
//    unsigned int number_of_rows2 = device_DOFmap.get_global_dof_number();
//    unsigned int number_of_cols2 = number_of_rows2;
//    unsigned int local_number_of_rows2 = device_DOFmap.get_number_of_dofs();
//
//    //these are the dimensions of the lead matrices:
//    unsigned int number_of_rows1      = number_of_super_rows-number_of_rows2;
//    unsigned int number_of_cols1      = number_of_super_cols-number_of_cols2;
//
//    if(opt.check_option("atom_order_from") || number_of_rows1>number_of_rows2)
//    {
//      //if smartdof used in device domain or device size < lead size (e.g RGF)
//      //----------------------------------------------------------
//      //Method I. get self energy that is defined on device domain
//      // this maybe slow since the matrix is of device size
//      // get the submatrix (upper right block) from coupling Hamiltonian
//      std::vector<int> temp_rows(local_number_of_rows2);
//      std::vector<int> temp_cols(number_of_cols1);
//      for(unsigned int i=0; i<local_number_of_rows2 && i<number_of_rows2; i++)
//        temp_rows[i]=i;
//      for(unsigned int i=0; i<number_of_cols1; i++)
//        temp_cols[i]=i+number_of_cols2;
//
//      PetscMatrixParallelComplex* coupling = NULL;
//      coupling= new PetscMatrixParallelComplex(number_of_rows2,number_of_cols1,get_simulation_domain()->get_communicator());
//      coupling->set_num_owned_rows(local_number_of_rows2);
//      vector<int> rows_diagonal(local_number_of_rows2,0);
//      vector<int> rows_offdiagonal(local_number_of_rows2,0);
//      for(unsigned int i=0; i<local_number_of_rows2; i++)
//      {
//        rows_diagonal[i]=coupling_Hamiltonian->get_nz_diagonal(i);
//        rows_offdiagonal[i]=coupling_Hamiltonian->get_nz_offdiagonal(i);
//      }
//      for(unsigned int i=0; i<local_number_of_rows2; i++)
//        coupling->set_num_nonzeros_for_local_row(i,rows_diagonal[i],rows_offdiagonal[i]);
//      coupling_Hamiltonian->get_submatrix(temp_rows,temp_cols,MAT_INITIAL_MATRIX,coupling);
//      coupling->assemble();
//
//      PropagationUtilities::transfer_matrix_get_new_self_energy(this,E_minus_H_matrix,T43,&number_of_wave_left,&number_of_vectors_size,
//                                          out_of_device_phase,out_of_device_modes,result,coupling);
//      delete coupling;
//      //----------------------------------------------------------
//      delete coupling_Hamiltonian;
//      delete coupling_Hamiltonian_lead;
//    }
//    else
//    {
//      //----------------------------------------------------------
//      //Method II. get self energy that is defined on contact domain, then translate into device
//      // this maybe faster than I, however requires a translation from contact into device
//      //1. get self energy on contact -- only upper left block is full
//      PetscMatrixParallelComplex* result_temp = NULL;
//      transfer_matrix_get_self_energy(E_minus_H_matrix,T43,&number_of_wave_left,&number_of_vectors_size,
//                                      out_of_device_phase,out_of_device_modes,result_temp,number_of_00_rows,number_of_00_cols);
//
//      //2. translate self energy into the correct atomic order
//      NemoUtils::tic(tic_toc_prefix + " translate self energy into device domain");
//      result = new PetscMatrixParallelComplex(number_of_rows2,number_of_cols2,get_simulation_domain()->get_communicator() );
//      result->set_num_owned_rows(local_number_of_rows2);
//      std::map<unsigned int, unsigned int> self_subindex_map;
//      coupling_DOFmap_lead->get_sub_DOF_index_map(subdomain_DOF0,self_subindex_map);
//      delete coupling_DOFmap_lead;
//      delete coupling_Hamiltonian_lead;
//      std::map<unsigned int, unsigned int>::iterator temp_it1=self_subindex_map.begin();
//
//      //figure out the nonzero rows of coupling
//      int row_index = 0;
//      for(unsigned int i=0; i<local_number_of_rows2; i++)
//      {
//        row_index = coupling_Hamiltonian->get_nz_diagonal(i);
//        if(row_index>0)
//        {
//          row_index = i;
//          break;
//        }
//      }
//      delete coupling_Hamiltonian;
//      //which block is this nonzero rows located?
//      int contact_index=0;
//      int start_point = 0;
//      unsigned int number_of_rows_to_set=0;
//      if(row_index!=0)
//      {
//        contact_index = number_of_rows2/row_index;
//      }
//      start_point = number_of_rows2-contact_index*number_of_rows1;
//      number_of_rows_to_set = number_of_rows1;
//      if(start_point<0 || row_index==0)
//        start_point = 0;
//
//      //set nonzero pattern
//      for(unsigned int i=0; i<number_of_rows2; i++)
//        result->set_num_nonzeros(i,0,0);
//      for(unsigned int i=0; i<number_of_rows_to_set; i++)
//      {
//        temp_it1=self_subindex_map.find(i);
//        if(temp_it1!=self_subindex_map.end())
//        {
//          unsigned int j = temp_it1->second;//translated matrix index
//          if(j>number_of_00_rows)
//            result->set_num_nonzeros(i+start_point,0,0);
//          else
//            result->set_num_nonzeros(i+start_point,result_temp->get_nz_diagonal(j),result_temp->get_nz_offdiagonal(j));
//        }
//        else
//          result->set_num_nonzeros(i+start_point,0,0);
//      }
//      result->allocate_memory();
//      result->set_to_zero();
//
//      //set matrix elements
//      const std::complex<double>* pointer_to_data= NULL;
//      vector<cplx> data_vector;
//      vector<int> col_index;
//      int n_nonzeros=0;
//      const int* n_col_nums=NULL;
//      temp_it1=self_subindex_map.begin();
//      for(; temp_it1!=self_subindex_map.end(); temp_it1++)
//      {
//        unsigned int i = temp_it1->first;
//        unsigned int j = temp_it1->second;//translated row index
//        unsigned int nonzero_cols = result_temp->get_nz_diagonal(j);
//        if(j<=number_of_00_rows && nonzero_cols>0)
//        {
//          result_temp->get_row(j,&n_nonzeros,n_col_nums,pointer_to_data);
//          col_index.resize(n_nonzeros,0);
//          data_vector.resize(n_nonzeros,cplx(0.0,0.0));
//          std::map<unsigned int, unsigned int>::iterator temp_it2=self_subindex_map.begin();
//          for(int jj=0; jj<n_nonzeros; jj++)
//          {
//            data_vector[jj]=pointer_to_data[jj];
//            col_index[jj]=temp_it2->first+start_point;
//            temp_it2++;
//          }
//          result->set(i+start_point,col_index,data_vector);
//          result_temp->store_row(j,&n_nonzeros,n_col_nums,pointer_to_data);
//        }
//      }
//      result->assemble();
//      NemoUtils::toc(tic_toc_prefix + " translate self energy into device domain");
//      delete result_temp;
//      //----------------------------------------------------------
//    }
//  }
//  else
//  {
//    transfer_matrix_get_self_energy(E_minus_H_matrix,T43,&number_of_wave_left,&number_of_vectors_size,
//                                    out_of_device_phase,out_of_device_modes,result,number_of_00_rows,number_of_00_cols);
//  }
//  bool output_cplxEk=options.get_option("output_complexEk", bool(false));
//  if(output_cplxEk&&!no_file_output) //complex band structure output
//  {
//    vector<cplx> kdelta(out_of_device_phase->get_num_rows(),0.0);
//    for(unsigned int i=0; i<out_of_device_phase->get_num_rows(); i++)
//    {
//      cplx ktmp=cplx(0.0,1.0)*log(out_of_device_phase->get(i,i));
//      kdelta[i]=ktmp;
//    }
//    std::ostringstream strs_e;
//    strs_e << energy.real();
//    std::string str = strs_e.str();
//    std::string Ekname="cplxEk.dat";
//    const char* filename = Ekname.c_str();
//    ofstream fEk;
//    fEk.open(filename,ios_base::app);
//    fEk<< out_of_device_phase->get_num_rows() << " ";
//    fEk<< energy.real() << " ";
//    for(unsigned int i=0; i<out_of_device_phase->get_num_rows(); i++)
//      fEk<<kdelta[i].real()<<" "<<kdelta[i].imag()<<" ";
//    fEk<<"\n";
//    fEk.close();
//  }
//
//  bool output_Ek=options.get_option("output_Ek", bool(false));
//  if(output_Ek&&!no_file_output) //normal band structure output
//  {
//    vector<double> kdelta(out_of_device_propagating_phase->get_num_rows(),0.0);
//    for(unsigned int i=0; i<out_of_device_propagating_phase->get_num_rows(); i++)
//    {
//      cplx ktmp=cplx(0.0,1.0)*log(out_of_device_propagating_phase->get(i,i));
//      kdelta[i]=ktmp.real();
//    }
//    std::stable_sort(kdelta.begin(),kdelta.end());
//    std::ostringstream strs_e;
//    strs_e << energy.real();
//    std::string str = strs_e.str();
//    std::string Ekname="Ek.dat";
//    const char* filename = Ekname.c_str();
//    ofstream fEk;
//    fEk.open(filename,ios_base::app);
//    fEk<< energy.real() << " ";
//    for(unsigned int i=0; i<out_of_device_propagating_phase->get_num_rows(); i++)
//      fEk<<kdelta[i]<<" ";
//    fEk<<"\n";
//    fEk.close();
//
//    vector<double> kdeltam(into_device_propagating_phase->get_num_rows(),0.0);
//    for(unsigned int i=0; i<into_device_propagating_phase->get_num_rows(); i++)
//    {
//      cplx ktmp=cplx(0.0,1.0)*log(into_device_propagating_phase->get(i,i));
//      kdeltam[i]=ktmp.real();
//    }
//    std::stable_sort(kdeltam.begin(),kdeltam.end());
//    std::ostringstream strs_em;
//    strs_em << energy.real();
//    std::string strm = strs_em.str();
//    std::string Eknamem="Ekm.dat";
//    const char* filenamem = Eknamem.c_str();
//    ofstream fEkm;
//    fEkm.open(filenamem,ios_base::app);
//    fEkm<< energy.real() << " ";
//    for(unsigned int i=0; i<into_device_propagating_phase->get_num_rows(); i++)
//      fEkm<<kdeltam[i]<<" ";
//    fEkm<<"\n";
//    fEkm.close();
//
//    //std::set<unsigned int> Hamilton_momentum_indices;
//    //std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
//    //find_Hamiltonian_momenta(writeable_Propagators.begin()->second,pointer_to_Hamilton_momentum_indices);
//    //NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
//    //if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=momentum[*(pointer_to_Hamilton_momentum_indices->begin())];
//    Simulation* neighbor_HamiltonConstructor =  NULL;
//    std::string neighbor_Hamilton_constructor_name;
//    std::string variable_name="Hamilton_constructor_"+neighbor_domain->get_name();
//    InputOptions& opt = Hamilton_Constructor->get_reference_to_options();
//    if(opt.check_option(variable_name))
//      neighbor_Hamilton_constructor_name=opt.get_option(variable_name,std::string(""));
//    else
//      throw std::invalid_argument("Schroedinger("+this->get_name()+")::assemble_hamiltonian: define \""+variable_name+"\"\n");
//    neighbor_HamiltonConstructor = this->find_simulation(neighbor_Hamilton_constructor_name);
//    PetscMatrixParallelComplex* contact_H=NULL;
//
//    // (to set multivalley effective mass Hamiltonian)
//    //set_valley(writeable_Propagators.begin()->second, momentum, neighbor_HamiltonConstructor);
//    std::vector<NemoMeshPoint> sorted_momentum;
//    QuantumNumberUtils::sort_quantum_number(momentum,sorted_momentum,options,momentum_mesh_types,neighbor_HamiltonConstructor);
//    neighbor_HamiltonConstructor->get_data(std::string("Hamiltonian"),contact_H,sorted_momentum);
//
//    contact_H->save_to_matlab_file(this->get_name()+"contact_Hamiltonian.m");
//    E_minus_H_matrix->save_to_matlab_file(this->get_name()+"EmH.m");
//  }
//  delete T43;
//  delete T34;
//  delete E_minus_H_matrix;
//
//  //if(options.get_option("avoid_copying_hamiltonian",false))
//  if(avoid_copying_hamiltonian)
//  {
//    delete D00;
//    delete D11;
//    delete D22;
//    delete D33;
//    if(new_self_energy)
//      delete Dtmp;
//  }
//
//  NemoUtils::toc(time_for_transfer_matrix_leads);
//
//  NemoUtils::toc(tic_toc_prefix);
//
//}
//

/* This function used to divide the ranks involved in calculating
 * the self energy to set of non overlapping
 * groups. Each group contains set of independent ranks that run together.
 *
 * The algorithm is divide to two main  steps:
 *
 * 1. Calculate the adjacent matrix. Each element of this matrix represents
 * the dependence between two communication pairs. The matrix element is zero if no
 * dependence and 1 otherwise.
 * Each column of the is matrix represents the dependence of one communication pair with the other communication pairs.
 *
 * 2. Reduce the adjacent matrix by combining columns iteratively if these columns are independent.
 * the  */
void
Self_energy::get_communication_order_parallel(
  std::vector<communication_pair>*& input_comm_table,
  std::vector < std::set<communication_pair> >*& comm_table_order,
  std::vector < std::map<int, std::set<std::vector<NemoMeshPoint> > > >*& rank_local_momentum_map,
  bool parallel)
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",
                                  get_name());
  std::string tic_toc_prefix = "Self_energy(\"" + temp_tic_toc_name
                               + "\")::get_communication_order_parallel ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\"" + this->get_name()
                       + "\")::get_communication_order_parallel() ";



  comm_table_order->clear();
  rank_local_momentum_map->clear();
  //The total number of available ranks
  int rank_size;
  MPI_Comm_size(holder.one_partition_total_communicator, &rank_size);

  //get the rank of this process
  int my_local_rank = 0;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);

  //map between the momentum point and the rank that hold it.
  std::map<std::vector<NemoMeshPoint>, int> map_momentum_rank(global_job_list);

  //number of available communication pairs or momentum points
  int momentum_size = input_comm_table->size();
  //Hold the adj matrix element part that is calculated by this rank.
  std::vector<int> output;
  //hold the number of element that will be calculated
  int myAdjSizeSecond = 0;
  //The total size of evaluated elements of the adjacent matrix.
  int tot_size = 0;

  // hold number of element of Adj matrix that will be sent by each rank
  int myAdjSizeFirst = 0;

  //If the tot_size the reminder will be assigned to the ranks from 0 to reminder -1
  int reminder_part = 0;

  //The number of elements calculated by the ranks before me.
  int sizeBefore = 0;
  //start indices of the adj matrix part that will be calculated by this rank.
  int jStart = 0;
  int iStart = 0;
  if (parallel == true)
  {

    //Step 1: Compute the adjacent matrix :

    //Step 1.1: Compute the adjacent matrix partition the element of the matrix over available ranks.
    //The matrix size is (momentum_size X momentum_size), but it is symmetric so only half of the element are needed.
    //The element will be calculated are the upper diagonal portion of the matrix.
    //At first column only one element, second column 2 elements, third column is 3 elements.
    //So the total size is summation of the series (1 2 3 ....momentum_size)
    //which is equal to momentum_size*(momentum_size+1)/2
    //The total size of evaluated elements of the adjacent matrix.
    tot_size = (std::pow(momentum_size, 2) + momentum_size) / 2;

    // hold number of element of Adj matrix that will be sent by each rank
    myAdjSizeFirst = tot_size / rank_size;

    //If the tot_size the reminder will be assigned to the ranks from 0 to reminder -1
    reminder_part = (tot_size) % rank_size;

    //Calculate the number of element of adj matrix calculated at each rank.
    //Each rank where 0 < rank <  reminder_part have one extra element .
    //But the ranks have to send equal size of data to each other
    // For ranks >//The number of elements calculated by the ranks before me. = reminder_part they will calculate myAdjSizeSecond = myAdjSizeFirst-1 but they will send  myAdjSizeFirst
    // putting -1 at  the last element.
    if (reminder_part > 0)
      myAdjSizeFirst++;
    //hold the number of element that will be calculated
    myAdjSizeSecond = myAdjSizeFirst;
    //Step 1.2 : calculating the start indices i,j for the part of the matrix that will be calulated by this rank.
    if ((my_local_rank >= reminder_part) && (reminder_part > 0))
    {
      //Correct the number of elements that will be calculated by the ranks < this process rank in case rank > =reminder_part
      myAdjSizeSecond--;
      //The number of elements calculated by the ranks before me.
      sizeBefore = (reminder_part) * myAdjSizeFirst
                   + (my_local_rank - reminder_part) * (myAdjSizeSecond);

    }
    else
    {
      //The number of elements calculated by the ranks before me.
      //Correct the number of elements that will be calculated by the ranks < this process rank in case rank < reminder_part
      sizeBefore = (my_local_rank) * myAdjSizeFirst;

    }

    // The start index in the adj matrix for the local process
    // The equation is size of Adj matrix calculated by ranks before this ranks j(j+1)/2+i
    // sizeBefore= j(j+1)/2+i............  (Eq 1 ), where j and i are the end indices for the
    // previous rank or start indices for this rank minus one.
    // Solve equation 1 to get i and j start
    jStart = std::floor((std::sqrt(1 + 4 * 2 * sizeBefore) - 1) / 2);
    iStart = sizeBefore - (std::pow(jStart, 2) + jStart) / 2;
    //Increment i,j by one to point to the start index.
    jStart++;
    iStart++;
  }
  else
  {
    //The total size of evaluated elements of the adjacent matrix.
    tot_size = (std::pow(momentum_size, 2) + momentum_size) / 2;
    //hold the number of element that will be calculated
    myAdjSizeSecond = tot_size;
    //start indices of the adj matrix part that will be calculated by this rank.
    jStart = 1;
    iStart = 1;
  }
  //Step 1.3: iterate over the unordered input communication table to calculate the dependence.
  int j = jStart;
  int i = iStart;
  bool overlap = false;
  std::vector<communication_pair>::iterator I_set_it = input_comm_table->begin();
  std::vector<communication_pair>::iterator J_set_it = input_comm_table->begin();
  //advance the iterators of the input communication table to point to elements jStart,iStart
  for (int m = 1; m < iStart; m++)
    I_set_it++;
  for (int m = 1; m < jStart; m++)
    J_set_it++;
  for (int k = 0; k < myAdjSizeSecond; k++)
  {

    //communication pair I (First :Vector of momentum , Second : set of vector of momentum)
    std::map<std::vector<NemoMeshPoint>, int>::iterator I_rank_it =
      map_momentum_rank.find((*I_set_it).first);
    //communication pair J (First :Vector of momentum , Second : set of vector of momentum)
    std::map<std::vector<NemoMeshPoint>, int>::iterator J_rank_it =
      map_momentum_rank.find((*J_set_it).first);
    overlap = false;
    // if any 2 elements of the two communication pair have the same rank then these pairs have dependence.
    // and can't calculated in the same iteration.
    // Compare first element of J with first element of I
    if (I_rank_it->second == J_rank_it->second)
    {
      overlap = true;

    }
    //Compare first element of communication pair I with each element of communication pair J second.
    std::set<std::vector<NemoMeshPoint> >::iterator J_set_of_points =
      J_set_it->second.begin();

    for (; ((J_set_of_points != J_set_it->second.end()) && (overlap == false));
         J_set_of_points++)
    {
      J_rank_it = map_momentum_rank.find((*J_set_of_points));
      if (I_rank_it->second == J_rank_it->second)
      {
        overlap = true;

      }

    }
    //Compare first element of communication pair J with each element of communication pair I second.
    J_rank_it = map_momentum_rank.find((*J_set_it).first);
    std::set<std::vector<NemoMeshPoint> >::iterator I_set_of_points =
      I_set_it->second.begin();

    for (; ((I_set_of_points != I_set_it->second.end()) && (overlap == false));
         I_set_of_points++)
    {
      I_rank_it = map_momentum_rank.find((*I_set_of_points));
      if (I_rank_it->second == J_rank_it->second)
      {
        overlap = true;

      }

    }

    //Compare each element of second element of communication pair I with each element of communication pair J second.
    I_set_of_points = I_set_it->second.begin();

    for (; ((I_set_of_points != I_set_it->second.end()) && overlap == false);
         I_set_of_points++)
    {

      I_rank_it = map_momentum_rank.find((*I_set_of_points));
      J_set_of_points = J_set_it->second.begin();
      for (;
           ((J_set_of_points != J_set_it->second.end()) && (overlap == false));
           J_set_of_points++)
      {
        J_rank_it = map_momentum_rank.find((*J_set_of_points));
        if (I_rank_it->second == J_rank_it->second)
        {
          overlap = true;

        }

      }

    }
    //push back the results of the comparison in the output
    //if the two communication pairs are dependent put there i,j indices in the output
    //if the communication pairs are not dependent skip them we will send only the element
    //of the Adj matrix that equal 1. Other element that equal zero will be sent to reduce the sent data size.
    if (overlap == true)
    {
      output.push_back(j);
      output.push_back(i);
    }
    //The upper diagonal is only calculated only for the symmetry.
    if (i >= j)
    {

      //got to the start of the row in the next column
      i = 1;
      I_set_it = input_comm_table->begin();
      //increment to the next column
      J_set_it++;
      j++;

    }
    else
    {
      //increment to go to the next row.
      I_set_it++;
      i++;
    }
  }
  //The size of the total sent data by each rank.
  int input_size;
  //this vector will hold the elements come from other ranks.
  std::vector<int>* input;
  std::vector<int> temp;
  input = &temp;

  if (parallel == true)
  {

    int ouptut_size = output.size();
    //to get the maximum size sent by each rank.
    MPI_Allreduce(&ouptut_size, &input_size, 1, MPI_INT, MPI_MAX,
                  holder.one_partition_total_communicator);

    // if the sent size of this rank is less than the maximum size sent by all ranks fill the
    // reminder part by -1 to reach the maximum size.
    output.resize(input_size, -1);
    //this vector will hold the elements come from other ranks.
    //resize to the total size come from other ranks
    input->resize(input_size * rank_size);
    // Share with all ranks the calculated elements of the Adj matrix
    MPI_Allgather(&output[0], input_size, MPI_INT, &((*input)[0]), input_size,
                  MPI_INT, holder.one_partition_total_communicator);
  }
  else
  {
    //this vector will hold the elements to do the reduce operation.
    input = &output;

  }

  //Calculate needed ranks per communication pair:
  vector<int> needed_ranks;
  //iterate over each communication pair in the ordered communication table to count the number of needed ranks per communication pairs.
  std::set<int> needed_ranks_per_comm_pair;
  needed_ranks.resize(momentum_size);
  i = 0;
  for (I_set_it = input_comm_table->begin(); I_set_it != input_comm_table->end();
       I_set_it++)
  {
    //search for the rank of that mesh point vector
    std::map<std::vector<NemoMeshPoint>, int>::iterator I_rank_it =
      map_momentum_rank.find((*I_set_it).first);
    //insert the rank in the list. List keep only unique set of ranks if the rank inserted
    //before it will not be inserted again.
    needed_ranks_per_comm_pair.insert(I_rank_it->second);
    std::set<std::vector<NemoMeshPoint> >::iterator I_set_of_points =
      I_set_it->second.begin();
    // for each element in the communication pair .
    // find the rank of that element and insert it in the set.
    for (; I_set_of_points != I_set_it->second.end(); I_set_of_points++)
    {
      I_rank_it = map_momentum_rank.find((*I_set_of_points));
      needed_ranks_per_comm_pair.insert(I_rank_it->second);

    }
    //Size of the set is equal to the number of ranks needed.
    needed_ranks[i] = (needed_ranks_per_comm_pair.size());
    i++;
    needed_ranks_per_comm_pair.clear();
  }

  //The debugging code
  //debug the received data
  if (debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    ofstream outfile;
    std::string filename;
    if (parallel == true)
    {
      filename = get_name() + "_output_needed_ranks_" + convert_to_string.str()
                 + ".dat";
      outfile.open(filename.c_str());

      std::stringstream temp_stream;
      for (unsigned int i = 0; i < needed_ranks.size(); i++)
      {

        temp_stream << (needed_ranks)[i] << "\t";

      }
      outfile << temp_stream.str() << "\n";

      outfile.close();
    }
  }

  // construct adj matrix .
  // my_adj2D hold the adj matrix
  std::vector < std::vector<bool>
  > my_adj2D(momentum_size, std::vector<bool>(momentum_size, false));

  // iterate on the input received and copy the elements to the my_adj2D
  for (unsigned int i = 0; i < input->size();)
  {
    // if element is -1 then it is empty and skipped.
    if ((*input)[i] != -1)
    {
      my_adj2D[(*input)[i] - 1][(*input)[i + 1] - 1] = true;
      i += 2;
    }
    else
    {
      i++;
    }
  }

  //This is the indices for the adjacent matrix rows.
  //In case of combining two columns the second column will be deleted.
  //The second row from symmetre have to be deleted as well but this will cost a lot of memory operation.
  //So updating the matrices to delete the index of this row instead of deleting the row.
  std::set<int> rowIndex;

  for (int j = 0; j < momentum_size; j++)
  {
    rowIndex.insert(j);

  }

  //Step 2: start reduce operation of the matrix

  //temp iterator used to make the or operation between the columns if they will be combined
  std::set<int>::iterator rowIndex_it3;

  //iterator to hold the first column  to be combined with rest of columns.
  std::set<int>::iterator colIndex_it;
  //Iterate over the second column that will be combined with  colIndex_it.
  std::set<int>::iterator temp_colIndex_it;

  //iterate over the columns by column.
  //if two columns are independent combine them.
  for (colIndex_it = rowIndex.begin(); colIndex_it != rowIndex.end();
       colIndex_it++)
  {
    //map[key=spatially global MPI-rank, value=set of momentum to be integrated on MPI-rank(key) - need to be within one communication pair]
    std::map<int, std::set<std::vector<NemoMeshPoint> > > ordered_local_second_momenta_for_this_iteration;
    //hold the independent communication pairs for this iteration.
    std::set<communication_pair> ordered_communication_pairs_of_this_iteration;


    //insert the first communication pair into the set of communication pair for this iteration
    ordered_communication_pairs_of_this_iteration.insert(
      (*input_comm_table)[(*colIndex_it)]);
    //update the job list for every involved MPI-rank for this iteration
    //(i.e. ordered_local_second_momenta_for_this_iteration)
    std::set<std::vector<NemoMeshPoint> >::const_iterator temp_set_cit =
      ((*input_comm_table)[(*colIndex_it)]).second.begin();

    //inserting the mesh points in the  ordered_local_second_momenta_for_this_iteration
    for (; temp_set_cit != ((*input_comm_table)[(*colIndex_it)]).second.end();
         temp_set_cit++)
    {
      std::map<std::vector<NemoMeshPoint>, int>::iterator rank_it =
        map_momentum_rank.find(*temp_set_cit);
      std::map<int, std::set<std::vector<NemoMeshPoint> > >::iterator temp_it =
        ordered_local_second_momenta_for_this_iteration.find(rank_it->second);
      if (temp_it != ordered_local_second_momenta_for_this_iteration.end())
        temp_it->second.insert(*temp_set_cit);
      else
      {
        std::set < std::vector<NemoMeshPoint> > temp_vector_set;
        temp_vector_set.insert(*temp_set_cit);
        ordered_local_second_momenta_for_this_iteration[rank_it->second] =
          temp_vector_set;
      }
    }

    //iterate over the Adj matrix column one by one to see if they can be combined with the column colIndex_it
    for (temp_colIndex_it = colIndex_it, temp_colIndex_it++;
         temp_colIndex_it != rowIndex.end();)
    {
      //check if the 2 columns (two communication pair) are independent.
      if ((my_adj2D[*temp_colIndex_it][*colIndex_it] == false)
          && ((needed_ranks[*colIndex_it] + needed_ranks[*temp_colIndex_it])
              <= rank_size))
      {
        //combine the 2 columns in the first columns colIndex_it
        //update the wanted ranks of this iteration by adding the ranks for columns temp_colIndex_it
        //The two set of ranks are exclusive as the two columns have no dependency.
        //The total sum is equivalent to the total number of ranks as there is no rank shared between them
        needed_ranks[*colIndex_it] += needed_ranks[*temp_colIndex_it];

        ordered_communication_pairs_of_this_iteration.insert(
          (*input_comm_table)[(*temp_colIndex_it)]);
        //update the job list for every involved MPI-rank for this iteration (i.e. ordered_local_second_momenta_for_this_iteration)
        temp_set_cit = ((*input_comm_table)[(*temp_colIndex_it)]).second.begin();

        //update the job list for every involved MPI-rank for this iteration
        //(i.e. ordered_local_second_momenta_for_this_iteration)
        for (; temp_set_cit != ((*input_comm_table)[(*temp_colIndex_it)]).second.end();
             temp_set_cit++)
        {
          std::map<std::vector<NemoMeshPoint>, int>::iterator rank_it =
            map_momentum_rank.find(*temp_set_cit);
          std::map<int, std::set<std::vector<NemoMeshPoint> > >::iterator temp_it =
            ordered_local_second_momenta_for_this_iteration.find(
              rank_it->second);
          if (temp_it != ordered_local_second_momenta_for_this_iteration.end())
            temp_it->second.insert(*temp_set_cit);
          else
          {
            std::set < std::vector<NemoMeshPoint> > temp_vector_set;
            temp_vector_set.insert(*temp_set_cit);
            ordered_local_second_momenta_for_this_iteration[rank_it->second] =
              temp_vector_set;
          }
        }
        //If the ranks reached maximum number of ranks then colIndex_it will not be used again no need for
        //or operation between the two columns
        if (needed_ranks[*colIndex_it] <= rank_size)
        {
          // iterate over the rows of the two combined columns to add the dependence of column2 to the dependence
          //of column 1 by or operation.
          std::set<int>::iterator rowIndex_it_temp;
          for (rowIndex_it_temp = temp_colIndex_it;
               rowIndex_it_temp != rowIndex.end(); rowIndex_it_temp++)
          {

            if ((my_adj2D[*rowIndex_it_temp][*colIndex_it]) == false)
            {
              ((my_adj2D[*rowIndex_it_temp][*colIndex_it] =
                  my_adj2D[*rowIndex_it_temp][*temp_colIndex_it]));
            }

          }
        }
        //erase column 2 by erasing its index
        rowIndex_it3 = temp_colIndex_it;
        //advance to the next column
        temp_colIndex_it++;
        //erase the index of the column from the index set
        rowIndex.erase(rowIndex_it3);

        //if ranks needed for this iteration reached the maximum rank
        //start new iteration
        if (needed_ranks[*colIndex_it] >= rank_size)
        {
          break;
        }
      }
      else
      {
        //if both columns are not independent advance to the next column
        temp_colIndex_it++;
      }

    }
    //insert this iteration in the output communication table and job list
    comm_table_order->push_back(ordered_communication_pairs_of_this_iteration);
    rank_local_momentum_map->push_back(
      ordered_local_second_momenta_for_this_iteration);

  }


  //The debugging code
  //debug the received data
  if (debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    ofstream outfile;
    std::string filename;
    if (parallel == true)
    {
      filename = get_name() + "_output_indices_" + convert_to_string.str()
                 + ".dat";
      outfile.open(filename.c_str());
      outfile << "the indices : " << "\n";
      outfile << "my_local_rank : " << my_local_rank << "\n";
      outfile << "parallel flag : " << parallel << "\n";
      outfile << "sizeBefore : " << sizeBefore << "\n";
      outfile << "rank_size : " << rank_size << "\n";
      outfile << "tot_size : " << tot_size << "\n";
      outfile << "reminder_part : " << reminder_part << "\n";
      outfile << "myAdjSizeFirst : " << myAdjSizeFirst << "\n";
      outfile << "myAdjSizeSecond: " << myAdjSizeSecond << "\n";
      outfile << "momentum_size: " << momentum_size << "\n";

      for (unsigned int i = 0; i < input->size();)
      {
        if ((*input)[i] != -1)
        {

          std::stringstream temp_stream;

          temp_stream << (*input)[i] << "\t";
          i++;
          while ((*input)[i] != -1)
            temp_stream << (*input)[i++] << "\t";

          outfile << temp_stream.str() << "\n";
        }
        else
        {

          i++;
        }

      }
      outfile.close();
    }

    //debug for printing the adj Matrix

    int print_size = momentum_size;

    filename = get_name() + "_output_adj"
               "_matrix_" + convert_to_string.str() + ".dat";
    outfile.open(filename.c_str());
    outfile << "communication Adj Matrix parrallel " << "\n";

    for (int j = 0; j < print_size; j++)
    {
      std::stringstream temp_stream;
      for (int i = 0; i <= j; i++)
      {
        temp_stream << my_adj2D[j][i] << "\t";
      }

      outfile << temp_stream.str() << "\n";

    }

    outfile.close();

    //debugging: global_job_list-output

    filename = get_name() + "_output_comm_order_" + convert_to_string.str()
               + ".dat";
    outfile.open(filename.c_str());
    std::map<std::vector<NemoMeshPoint>, int>::iterator I_rank_it;

    for (unsigned int i = 0; i < comm_table_order->size(); i++)
    {
      std::stringstream temp_stream;
      temp_stream << i;
      outfile << "communication iteration " << temp_stream.str() << "\n";
      std::set<communication_pair>::const_iterator temp_cit =
        (*comm_table_order)[i].begin();
      for (; temp_cit != (*comm_table_order)[i].end(); temp_cit++)
      {

        std::stringstream temp_stream;

        I_rank_it = map_momentum_rank.find(temp_cit->first);
        temp_stream << I_rank_it->second << "\t--->\t";

        std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2 =
          temp_cit->second.begin();
        while (c_it2 != temp_cit->second.end())
        {

          I_rank_it = map_momentum_rank.find(*c_it2);
          temp_stream << I_rank_it->second << "\t--->\t";
          c_it2++;
          if (c_it2 != temp_cit->second.end())
          {
            temp_stream << ", ";
          }
        }
        outfile << temp_stream.str() << "\n";
      }
    }
    outfile.close();

    //std::vector<std::map<int,std::set<std::vector<NemoMeshPoint> > > >*& rank_local_momentum_map;
    filename = get_name() + "rank_local_momentum_map" + convert_to_string.str()
               + ".dat";
    outfile.open(filename.c_str());
    for (unsigned int i = 0; i < rank_local_momentum_map->size(); i++)
    {
      std::stringstream temp_stream;
      temp_stream << i;
      outfile << "local momenta during each communication iteration "
              << temp_stream.str() << "\n";
      std::map<int, std::set<std::vector<NemoMeshPoint> > >::const_iterator temp_cit =
        (*rank_local_momentum_map)[i].begin();
      for (; temp_cit != (*rank_local_momentum_map)[i].end(); temp_cit++)
      {
        std::stringstream temp_stream2;
        temp_stream2 << temp_cit->first;
        std::string data_string = "MPI process: " + temp_stream2.str();
        data_string += "\tintegrates locally:\t";
        std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2 =
          temp_cit->second.begin();
        while (c_it2 != temp_cit->second.end())
        {
          std::string convert2;
          const std::vector<NemoMeshPoint>* temp_pointer2 = &(*c_it2);
          translate_momentum_vector(temp_pointer2, convert2);
          data_string += convert2;
          c_it2++;
          if (c_it2 != temp_cit->second.end())
          {
            data_string += ", ";
          }
        }
        outfile << data_string << "\n";
      }
    }
    outfile.close();
  }    //end of the debug

  NemoUtils::toc(tic_toc_prefix);
}

void
Self_energy::communication_rank_sort(
  std::set<communication_pair>*& input_comm_table,
  std::vector<communication_pair>*& output_comm_table, bool maximum_first)
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",
                                  get_name());
  std::string tic_toc_prefix = "Self_energy(\"" + temp_tic_toc_name
                               + "\")::communication_rank_sort ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\"" + this->get_name()
                       + "\")::communication_rank_sort()";

  //get the rank of this process
  int my_local_rank = 0;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);
  //get the total number of ranks.
  int rank_size;
  MPI_Comm_size(holder.one_partition_total_communicator, &rank_size);

  int tot_size = input_comm_table->size();

  // hold number of element of Adj matrix that will be sent by each rank
  int myAdjSizeFirst = tot_size / rank_size;

  // If the tot_size the reminder will be assigned to the ranks from 0 to reminder_part -1
  int reminder_part = (tot_size) % rank_size;

  // Calculate the number of element of adj matrix calculated at each rank.
  // Each rank where 0 < rank <  reminder_part have one extra element .
  // But the ranks have to send equal size of data to each other
  // For ranks > = reminder_part they will calculated myAdjSizeSecond = myAdjSizeFirst-1 but they will send  myAdjSizeFirst
  // putting -1 at  the last element.
  if (reminder_part > 0)
    myAdjSizeFirst++;
  // Hold the output
  std::vector<int> needed_ranks;
  // the start and the end element of the momentum point that will be calculated by this rank
  int jStart;
  int jEnd;
  // case reminder >0 and this rank will not do any part of the reminder
  if ((reminder_part > 0) && (my_local_rank >= reminder_part))
  {

    jStart = reminder_part * myAdjSizeFirst
             + (my_local_rank - reminder_part) * (myAdjSizeFirst - 1);
    jEnd = reminder_part * myAdjSizeFirst
           + (my_local_rank - reminder_part + 1) * (myAdjSizeFirst - 1) - 1;
    //Correct the number of elements that will be calculated by the ranks < this process rank in case rank > =reminder_part

    needed_ranks.resize(myAdjSizeFirst);
    needed_ranks[myAdjSizeFirst - 1] = -1;

  }
  else
  {
    //case of having no reminder or the rank < reminder_part so it will make an extra element
    jStart = my_local_rank * myAdjSizeFirst;
    jEnd = (my_local_rank + 1) * myAdjSizeFirst - 1;
    //Correct the number of elements that will be calculated by the ranks < this process rank in case rank > =reminder_part
    needed_ranks.resize(myAdjSizeFirst);

  }

  //iterate over each communication pair in the ordered communication table to count the number of needed ranks per communication pairs.

  std::set<communication_pair>::iterator I_set_it = input_comm_table->begin();
  std::advance(I_set_it, jStart);
  //map between the momentum point and the rank that hold it.
  std::map<std::vector<NemoMeshPoint>, int> map_momentum_rank(global_job_list);
  std::set<int> needed_ranks_per_comm_pair;
  int i = 0;
  for (; ((I_set_it != input_comm_table->end()) && (jStart + i <= jEnd));
       I_set_it++)
  {
    //search for the rank of that mesh point vector
    std::map<std::vector<NemoMeshPoint>, int>::iterator I_rank_it =
      map_momentum_rank.find((*I_set_it).first);
    //insert the rank in the list. List keep only unique set of ranks if the rank inserted
    //before it will not be inserted again.
    needed_ranks_per_comm_pair.insert(I_rank_it->second);
    std::set<std::vector<NemoMeshPoint> >::iterator I_set_of_points =
      I_set_it->second.begin();
    // for each element in the communication pair .
    // find the rank of that element and insert it in the set.
    for (; I_set_of_points != I_set_it->second.end(); I_set_of_points++)
    {
      I_rank_it = map_momentum_rank.find((*I_set_of_points));
      needed_ranks_per_comm_pair.insert(I_rank_it->second);

    }
    //Size of the set is equal to the number of ranks needed.
    needed_ranks[i] = (needed_ranks_per_comm_pair.size());
    i++;
    needed_ranks_per_comm_pair.clear();
  }

  //this vector will hold the elements come from other ranks.
  std::vector<int> input;
  input.resize(myAdjSizeFirst * rank_size);
  // Share with all ranks the calculated elements
  MPI_Allgather(&needed_ranks[0], myAdjSizeFirst, MPI_INT, &input[0],
                myAdjSizeFirst, MPI_INT, holder.one_partition_total_communicator);
  //this vector will hold the elements come from other ranks after remove delimiter.
  std::vector<int> input_without_delimiter;
  input_without_delimiter.resize(input.size() - reminder_part);
  int k = 0;
  for (i = 0; i < (int) input.size();)
  {
    if (input[i] != -1)
    {
      input_without_delimiter[k] = input[i];
      i++;
      k++;
    }
    else
    {
      i++;
    }
  }
  std::vector<int> ranks;

  ranks.resize(myAdjSizeFirst, 0);
  if ((reminder_part > 0) && (my_local_rank >= reminder_part))
  {
    ranks[myAdjSizeFirst - 1] = -1;
  }

  i = 0;
  for (int j = jStart; j <= jEnd; j++, i++)
  {
    for (k = 0; k < (int) input_without_delimiter.size(); k++)
    {
      if(maximum_first == false)
      {
        if ((input_without_delimiter[j] > input_without_delimiter[k])
            || ((input_without_delimiter[j] == input_without_delimiter[k])
                && (j > k)))
        {

          ranks[i]++;

        }
      }
      else
      {
        if ((input_without_delimiter[j] < input_without_delimiter[k])
            || ((input_without_delimiter[j] == input_without_delimiter[k])
                && (j > k)))
        {

          ranks[i]++;

        }
      }

    }
  }
  //this vector will hold the elements come from other ranks.
  std::vector<int> input_ranks;
  input_ranks.resize(myAdjSizeFirst * rank_size);
  // Share with all ranks the calculated elements
  MPI_Allgather(&ranks[0], myAdjSizeFirst, MPI_INT, &input_ranks[0],
                myAdjSizeFirst, MPI_INT, holder.one_partition_total_communicator);
  //this vector will hold the elements come from other ranks after remove delimiter.
  std::vector<int> input_ranks_without_delimiter;
  //this vector will hold the elements come from other ranks after remove delimiter.

  input_ranks_without_delimiter.resize(input_ranks.size() - reminder_part);
  k = 0;
  for (i = 0; i < (int) input.size();)
  {
    if (input[i] != -1)
    {
      input_ranks_without_delimiter[k] = input_ranks[i];
      i++;
      k++;
    }
    else
    {
      i++;
    }
  }
  I_set_it = input_comm_table->begin();

  output_comm_table->resize(input_comm_table->size());
  for (i = 0; i < (int) input_comm_table->size(); i++)
  {
    (*output_comm_table)[input_ranks_without_delimiter[i]] = (*I_set_it);
    I_set_it++;
  }
  if(debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::string filename = get_name() + "_output_comm_table_after_rank_sort_"
                           + convert_to_string.str() + ".dat";
    ofstream outfile;
    outfile.open(filename.c_str());

    for ( int k = 0; k < (int) output_comm_table->size(); k++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer =
        &((*output_comm_table)[k].first);
      translate_momentum_vector(temp_pointer, data_string);
      data_string += "\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2 =
        (*output_comm_table)[k].second.begin();
      while (c_it2 != (*output_comm_table)[k].second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2 = &(*c_it2);
        translate_momentum_vector(temp_pointer2, convert2);
        data_string += convert2;
        c_it2++;
        if (c_it2 != (*output_comm_table)[k].second.end())
        {
          data_string += ", ";
        }
      }
      outfile << data_string << "\n";
    }
    outfile << "need ranks per communication pairs" << "\n";
    for (i = 0; i < (int) input_without_delimiter.size(); i++)
    {
      outfile << input_without_delimiter[i] << "\t";
    }
    outfile << "\n";
    outfile << "new index " << "\n";
    for (i = 0; i < (int) input_ranks_without_delimiter.size(); i++)
    {
      outfile << input_ranks_without_delimiter[i] << "\t";
    }
    outfile << "\n";
    outfile.close();



  }
  NemoUtils::toc(tic_toc_prefix);
}


void
Self_energy::get_deformation_communication_parallel(
  const double energy_interval, const double momentum_interval,
  std::set<communication_pair>*& output_comm_table)
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",
                                  get_name());
  std::string tic_toc_prefix = "Self_energy(\"" + temp_tic_toc_name
                               + "\")::get_deformation_communication_parallel ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\"" + this->get_name()
                       + "\")::get_deformation_communication_parallel()";
  output_comm_table->clear();

  //get the rank of this process
  int my_local_rank = 0;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);
  //get the total number of ranks.
  int rank_size;
  MPI_Comm_size(holder.one_partition_total_communicator, &rank_size);

  int tot_size = pointer_to_all_momenta->size();

  // hold number of element of Adj matrix that will be sent by each rank
  int myAdjSizeFirst = tot_size / rank_size;

  // If the tot_size the reminder will be assigned to the ranks from 0 to reminder_part -1
  int reminder_part = (tot_size) % rank_size;

  // Calculate the number of element of adj matrix calculated at each rank.
  // Each rank where 0 < rank <  reminder_part have one extra element .
  // But the ranks have to send equal size of data to each other
  // For ranks > = reminder_part they will calculated myAdjSizeSecond = myAdjSizeFirst-1 but they will send  myAdjSizeFirst
  // putting -1 at  the last element.
  if (reminder_part > 0)
    myAdjSizeFirst++;
  // Hold the output
  std::vector<int> output;
  // the start and the end element of the momentum point that will be calculated by this rank
  int jStart;
  int jEnd;
  // case reminder >0 and this rank will not do any part of the reminder
  if ((reminder_part > 0) && (my_local_rank >= reminder_part))
  {

    jStart = reminder_part * myAdjSizeFirst
             + (my_local_rank - reminder_part) * (myAdjSizeFirst - 1);
    jEnd = reminder_part * myAdjSizeFirst
           + (my_local_rank - reminder_part + 1) * (myAdjSizeFirst - 1) - 1;

  }
  else
  {
    //case of having no reminder or the rank < reminder_part so it will make an extra element
    jStart = my_local_rank * myAdjSizeFirst;
    jEnd = (my_local_rank + 1) * myAdjSizeFirst - 1;

  }

  //4. delete t
  const Propagator* temp_propagator = writeable_Propagator; //s.begin()->second;
  //1. loop over all existing momentum points for this rank
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it =
    pointer_to_all_momenta->begin();
  //advance to the start element of this rank
  int indexJ = 0;
  for (; indexJ < jStart; indexJ++)
    c_it++;
  //indexI used to  iterate over momentum points form the start to the end.
  int indexI;
  for (; ((c_it != pointer_to_all_momenta->end()) && (indexJ <= jEnd));
       c_it++, indexJ++)
  {
    //add the first elements of the communication pair to the output.
    //communication pair is <first vector<NemoMeshPoint> , second set of vectors <NemoMeshPoint> >
    output.push_back(indexJ);
    //2. get the energy of the respective energy point and the momentum
    const double energy1 = PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,*c_it, temp_propagator);
    std::vector<double> momentum1 = std::vector<double>(3, 0.0);
    if ((*c_it).size() > 1)
      momentum1 = read_kvector_from_momentum(*c_it, temp_propagator);
    std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2 =
      pointer_to_all_momenta->begin();
    indexI = 0;
    //3. loop over all existing momentum points
    for (; c_it2 != pointer_to_all_momenta->end(); c_it2++, indexI++)
    {
      //4. get the energy of the respective energy point and the momentum
      const double energy2 = PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,*c_it2, temp_propagator);
      std::vector<double> momentum2 = std::vector<double>(3, 0.0);
      if ((*c_it2).size() > 1)
        momentum2 = read_kvector_from_momentum(*c_it2, temp_propagator);
      std::vector<double> momentum_change(momentum2.size(), 0.0);
      for (unsigned int i = 0; i < momentum_change.size(); i++)
        momentum_change[i] = momentum1[i] - momentum2[i];
      double abs_delta_momentum = NemoMath::vector_norm(momentum_change);
      //5. if the difference of the two energies is smaller than the energy_interval
      //and if the absolute value of the momentum change is smaller than momentum_interval - add the second point to temp_comm_set (first point will be first element of the input_comm_table)
      const double energy_difference = std::abs(energy1 - energy2);
      if (energy_difference <= energy_interval
          && abs_delta_momentum <= momentum_interval)
      {
        //add the momentum point to the second elements of the communication pair to the output.
        //communication pair is <first vector<NemoMeshPoint> , second set of vectors <NemoMeshPoint> >
        output.push_back(indexI);
      }
    }
    //delimiter that identify the start of the new communication pair.
    output.push_back(-1);
  }

  //the size of the maximum size of data from all ranks
  int input_size;

  //the size of the data at this ranks.
  int ouptut_size = output.size();

  //Get the maximum size of data at all ranks.
  MPI_Allreduce(&ouptut_size, &input_size, 1, MPI_INT, MPI_MAX,
                holder.one_partition_total_communicator);

  //if the size of data will be send by this process less than the maximum size fill the reminder elements by -1
  output.resize(input_size, -1);
  //this vector will hold the elements come from other ranks.
  std::vector<int> input;
  input.resize(input_size * rank_size);
  // Share with all ranks the calculated elements
  MPI_Allgather(&output[0], input_size, MPI_INT, &input[0], input_size, MPI_INT,
                holder.one_partition_total_communicator);

  //iteratof over the momentum points that hold the first element of the communication pair.
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it_J =
    pointer_to_all_momenta->begin();
  //iteratof over the momentum points that hold the elements of the second part of the communication pair.
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it_I;
  //offset to increment the c_it_J
  int lastIndexJ;
  lastIndexJ = 0;
  //offset to increment the c_it_J
  int lastIndexI;

  for ( int i = 0; i < (int) input.size();)
  {
    //if input != -1 add this element to the communication table
    if (input[i] != -1)
    {
      // advance to the place of that momentum point in the momentum set.
      std::advance(c_it_J, input[i] - lastIndexJ);
      // update the offset for next iteration
      lastIndexJ = input[i];
      //hold the momentum points for the second element of the communication pair.
      std::set < std::vector<NemoMeshPoint> > temp_comm_set;
      c_it_I = pointer_to_all_momenta->begin();
      lastIndexI = 0;
      i++;
      //if input[i]==-1 this end of the communication pair and new communication pair have to be started.
      while (input[i] != -1)
      {
        //advance the iterator to the element at input[i]
        std::advance(c_it_I, input[i] - lastIndexI);
        //insert this point to the communication pair.
        temp_comm_set.insert(*c_it_I);
        //update the offset.
        lastIndexI = input[i];
        //go to the next element of the input
        i++;

      }
      //if the size of temp_comm_set>0, i.e. this momentum is interacting with some other points, add this communication pair to input_comm_table
      if (temp_comm_set.size() > 0)
      {
        communication_pair temp_pair(*c_it_J, temp_comm_set);
        output_comm_table->insert(temp_pair);
      }
    }
    else
    {
      //case input[i]==-1 increment to the next input element
      i++;

    }
  }
  //debugging: global_job_list-output
  if (debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::set<communication_pair>::const_iterator temp_cit =
      output_comm_table->begin();
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::string filename = get_name() + "_output_comm_table_"
                           + convert_to_string.str() + ".dat";
    ofstream outfile;
    outfile.open(filename.c_str());
    for (; temp_cit != output_comm_table->end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer, data_string);
      data_string += "\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2 =
        temp_cit->second.begin();
      while (c_it2 != temp_cit->second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2 = &(*c_it2);
        translate_momentum_vector(temp_pointer2, convert2);
        data_string += convert2;
        c_it2++;
        if (c_it2 != temp_cit->second.end())
        {
          data_string += ", ";
        }
      }
      outfile << data_string << "\n";
    }
    outfile.close();
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::get_communication_order(const std::set<communication_pair>& input_comm_table,
    std::vector<std::set<communication_pair> >*& output_comm_order,
    std::vector<std::map<int,std::set<std::vector<NemoMeshPoint> > > >*& output_local_point_order) const
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::get_communication_order ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_communication_order() ";
  output_comm_order->clear();
  output_local_point_order->clear();
  //pair between "initial" (momentum, energy) and set of "final" (momentum', energy') that the initial couple to
  //typedef std::pair<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > > communication_pair;

  //1. copy of the input_comm_table for those entries that have not been put to output_comm_order, yet
  std::set<communication_pair> total_set_of_unordered_comm_pairs(input_comm_table);


  //NOTE: fill output_local_point_order with the jobs done on one CPU that need to be integrated BEFORE sending...
  //2. loop until total_set_of_unordered_comm_pairs is empty
  while(total_set_of_unordered_comm_pairs.size()>0)
  {
    //msg<<prefix<<"number of communication pairs to distribute is: "<<total_set_of_unordered_comm_pairs.size()<<"\n";
    //communication iteration index =  last index/current size of output_comm_order
    //set of comm_pairs available for this communication iteration
    std::set<communication_pair> set_of_unordered_comm_pairs_for_this_iteration(total_set_of_unordered_comm_pairs);

    //set of communication_pair that can communicate during this communication iteration
    std::set<communication_pair> ordered_communication_pairs_of_this_iteration;

    //map[key=spatially global MPI-rank, value=set of momenta to be integrated on MPI-rank(key) - need to be within one communicationpair]
    std::map<int,std::set<std::vector<NemoMeshPoint> > > ordered_local_second_momenta_for_this_iteration;

    //set of all MPI processes available for this MPI communication iteration
    std::set<int> list_of_available_MPI_ranks;
    std::map<std::vector<NemoMeshPoint>, int>::const_iterator temp_c_it = global_job_list.begin();
    for(; temp_c_it!=global_job_list.end(); temp_c_it++)
      list_of_available_MPI_ranks.insert(temp_c_it->second);
    std::map<std::vector<NemoMeshPoint>, int> MPI_ranks_for_this_iteration(global_job_list);

    NEMO_ASSERT(MPI_ranks_for_this_iteration.size()>0,prefix+"empty MPI_ranks_for_this_iteration\n");
    //3. loop until set_of_unordered_comm_pairs_for_this_iteration is empty or no more MPI-ranks are available
    while(set_of_unordered_comm_pairs_for_this_iteration.size()>0 && MPI_ranks_for_this_iteration.size()>0 && list_of_available_MPI_ranks.size()>0)
    {
      //msg<<prefix<<"number of local pairs to distribute is: "<<set_of_unordered_comm_pairs_for_this_iteration.size()<<"\n";
      bool available=true;
      //3. put first element of current set_of_unordered_comm_pairs_for_this_iteration into ordered_communication_pairs_of_this_iteration if available...
      std::set<communication_pair>::iterator unordered_set_it=set_of_unordered_comm_pairs_for_this_iteration.begin();
      find_biggest_communication_pair(&set_of_unordered_comm_pairs_for_this_iteration,unordered_set_it);

      //for 4.4 store the communication partners of the element (*unordered_set_it) in added_point and added_set_of_points
      const std::vector<NemoMeshPoint> added_point = (*unordered_set_it).first;
      const std::set<std::vector<NemoMeshPoint> > added_set_of_points = (*unordered_set_it).second;

      //3.1 check if the current element of set_of_unordered_comm_pairs_for_this_iteration ((*unordered_set_it).first and all in (*unordered_set_it).second) is within the available MPI_ranks_for_this_iteration
      std::map<std::vector<NemoMeshPoint>, int>::iterator available_it = MPI_ranks_for_this_iteration.find((
            *unordered_set_it).first);
      if(available_it==MPI_ranks_for_this_iteration.end()) available=false;
      else
      {
        std::set<int>::const_iterator set_c_it=list_of_available_MPI_ranks.find(available_it->second);
        if(set_c_it==list_of_available_MPI_ranks.end())
          available=false;
      }
      //if still available, continue to check the availability of the MPI-processes of (*unordered_set_it)->second
      if(available)
      {
        std::set<std::vector<NemoMeshPoint> >::const_iterator temp_set_cit=(*unordered_set_it).second.begin();
        for(; temp_set_cit!=(*unordered_set_it).second.end() && available; temp_set_cit++)
        {
          const std::vector<NemoMeshPoint>& current_point = *temp_set_cit;
          available_it = MPI_ranks_for_this_iteration.find(current_point);
          if(available_it==MPI_ranks_for_this_iteration.end()) available=false;
          else
          {
            std::set<int>::const_iterator set_c_it=list_of_available_MPI_ranks.find(available_it->second);
            if(set_c_it==list_of_available_MPI_ranks.end())
              available=false;
          }
        }
      }
      if(available)
      {
        //3.2 add the current communication_pair to ordered_communication_pairs_of_this_iteration
        ordered_communication_pairs_of_this_iteration.insert(*unordered_set_it);
        //3.3 update the job list for every involved MPI-rank for this iteration (i.e. ordered_local_second_momenta_for_this_iteration)
        std::set<std::vector<NemoMeshPoint> >::const_iterator temp_set_cit=(*unordered_set_it).second.begin();
        for(; temp_set_cit!=(*unordered_set_it).second.end(); temp_set_cit++)
        {
          std::map<std::vector<NemoMeshPoint>, int>::iterator rank_it = MPI_ranks_for_this_iteration.find(*temp_set_cit);
          std::map<int,std::set<std::vector<NemoMeshPoint> > >::iterator temp_it=ordered_local_second_momenta_for_this_iteration.find(
                rank_it->second);
          if(temp_it!=ordered_local_second_momenta_for_this_iteration.end())
            temp_it->second.insert(*temp_set_cit);
          else
          {
            std::set<std::vector<NemoMeshPoint> > temp_vector_set;
            temp_vector_set.insert(*temp_set_cit);
            ordered_local_second_momenta_for_this_iteration[rank_it->second]=temp_vector_set;
          }
        }


        //4. delete this communication pair from the to-order list as well as all involved MPI-ranks from the availability list of this iteration
        //(i.e. from total_set_of_unordered_comm_pairs and from MPI_ranks_for_this_iteration)
        //4.1 delete this element (*unordered_set_it) from total_set_of_unordered_comm_pairs
        std::set<communication_pair>::iterator temp_total_it=total_set_of_unordered_comm_pairs.find(*unordered_set_it);
        NEMO_ASSERT(temp_total_it!=total_set_of_unordered_comm_pairs.end(),prefix+"inconsistent find result of total_set_of_unordered_comm_pairs\n");
        total_set_of_unordered_comm_pairs.erase(temp_total_it);
        //4.2 delete all involved MPI-processes from the availability list
        available_it = MPI_ranks_for_this_iteration.find((*unordered_set_it).first);
        NEMO_ASSERT(available_it!= MPI_ranks_for_this_iteration.end(),prefix+"inconsistent find result of MPI_ranks_for_this_iteration (first)\n");
        std::set<int>::iterator set_it=list_of_available_MPI_ranks.find(available_it->second);
        if(set_it!=list_of_available_MPI_ranks.end())
          list_of_available_MPI_ranks.erase(set_it);
        MPI_ranks_for_this_iteration.erase(available_it);
        temp_set_cit=(*unordered_set_it).second.begin();
        for(; temp_set_cit!=(*unordered_set_it).second.end(); temp_set_cit++)
        {
          available_it = MPI_ranks_for_this_iteration.find(*temp_set_cit);
          //NEMO_ASSERT(available_it!= MPI_ranks_for_this_iteration.end(),prefix+"inconsistent find result of MPI_ranks_for_this_iteration (second)\n");
          if(available_it!=MPI_ranks_for_this_iteration.end())
            MPI_ranks_for_this_iteration.erase(available_it);
          set_it=list_of_available_MPI_ranks.find(available_it->second);
          if(set_it!=list_of_available_MPI_ranks.end())
            list_of_available_MPI_ranks.erase(set_it);
        }
      }
      //4.3 if (*unordered_set_it) was available or if not, it should not be available any longer in any case...
      set_of_unordered_comm_pairs_for_this_iteration.erase(*unordered_set_it);

      //4.4 delete the communication partners of the element (*unordered_set_it) from the set_of_unordered_comm_pairs_for_this_iteration
      unordered_set_it=set_of_unordered_comm_pairs_for_this_iteration.begin();
      for(; unordered_set_it!=set_of_unordered_comm_pairs_for_this_iteration.end()&&set_of_unordered_comm_pairs_for_this_iteration.size()>0;)
      {
        //if there is overlap of the current point (first and second of *unordered_set_it) with either the added_point or added_set_of_points,
        //delete current point from the set_of_unordered_comm_pairs_for_this_iteration
        bool found = false;
        //4.4.1 is the first of current point the added_point?
        //if((*unordered_set_it).first==added_point) found=true;
        if(added_point == (*unordered_set_it).first) found=true;

        //4.4.2 is the added_point within second of current point?
        if(!found)
        {
          std::set<std::vector<NemoMeshPoint> >::iterator second_iterator=(*unordered_set_it).second.find(added_point);
          if(second_iterator!=(*unordered_set_it).second.end()) found = true;
        }
        //4.4.3 is the first of current point within added_set_of_points?
        if(!found)
        {
          std::set<std::vector<NemoMeshPoint> >::iterator second_iterator=added_set_of_points.find((*unordered_set_it).first);
          if(second_iterator!=added_set_of_points.end()) found = true;
        }
        //4.4.4 is any element of second of current point within added_set_of_points?
        if(!found)
        {
          std::set<std::vector<NemoMeshPoint> >::iterator second_iterator=(*unordered_set_it).second.begin();
          for(; second_iterator!=(*unordered_set_it).second.end() && !found; second_iterator++)
          {
            std::set<std::vector<NemoMeshPoint> >::iterator third_iterator=added_set_of_points.find((*second_iterator));
            if(third_iterator!=added_set_of_points.end()) found = true;
          }
        }
        if(found)
        {
          //msg<<prefix<<"found entry in set_of_unordered_comm_pairs_for_this_iteration to delete\n";
          set_of_unordered_comm_pairs_for_this_iteration.erase(unordered_set_it++);
        }
        else
          unordered_set_it++;
      }
      //5. reset the iterator to the beginning of the set_of_unordered_comm_pairs_for_this_iteration and repeat until set_of_unordered_comm_pairs_for_this_iteration is empty
      unordered_set_it=set_of_unordered_comm_pairs_for_this_iteration.begin();
    }
    //6. store the communication_pairs_of_this_iterations in output_comm_order and also store the jobs per MPI-process in output_local_point_order
    output_comm_order->push_back(ordered_communication_pairs_of_this_iteration);
    output_local_point_order->push_back(ordered_local_second_momenta_for_this_iteration);
    //7. repeat 2.-6. until the working_copy is empty (i.e. all elements are ordered)
  }

  //debugging: global_job_list-output
  if(debug_output_job_list&&!no_file_output)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::string filename=get_name()+"_output_comm_order_"+convert_to_string.str()+".dat";
    ofstream outfile;
    outfile.open(filename.c_str());
    for(unsigned int i=0; i<output_comm_order->size(); i++)
    {
      std::stringstream temp_stream;
      temp_stream << i;
      outfile<<"communication iteration "<<temp_stream.str()<<"\n";
      std::set<communication_pair>::const_iterator temp_cit=(*output_comm_order)[i].begin();
      for(; temp_cit!=(*output_comm_order)[i].end(); temp_cit++)
      {
        std::string data_string;
        const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
        translate_momentum_vector(temp_pointer,data_string);
        data_string +="\t--->\t";
        std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
        while(c_it2!=temp_cit->second.end())
        {
          std::string convert2;
          const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
          translate_momentum_vector(temp_pointer2,convert2);
          data_string +=convert2;
          c_it2++;
          if(c_it2!=temp_cit->second.end())
          {
            data_string +=", ";
          }
        }
        outfile<<data_string<<"\n";
      }
    }
    outfile.close();

    //std::vector<std::map<int,std::set<std::vector<NemoMeshPoint> > > >*& output_local_point_order;
    filename=get_name()+"output_local_point_order"+convert_to_string.str()+".dat";
    outfile.open(filename.c_str());
    for(unsigned int i=0; i<output_local_point_order->size(); i++)
    {
      std::stringstream temp_stream;
      temp_stream << i;
      outfile<<"local momenta during each communication iteration "<<temp_stream.str()<<"\n";
      std::map<int,std::set<std::vector<NemoMeshPoint> > >::const_iterator temp_cit=(*output_local_point_order)[i].begin();
      for(; temp_cit!=(*output_local_point_order)[i].end(); temp_cit++)
      {
        std::stringstream temp_stream2;
        temp_stream2<< temp_cit->first;
        std::string data_string="MPI process: "+temp_stream2.str();
        data_string +="\tintegrates locally:\t";
        std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
        while(c_it2!=temp_cit->second.end())
        {
          std::string convert2;
          const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
          translate_momentum_vector(temp_pointer2,convert2);
          data_string +=convert2;
          c_it2++;
          if(c_it2!=temp_cit->second.end())
          {
            data_string +=", ";
          }
        }
        outfile<<data_string<<"\n";
      }
    }
    outfile.close();
  }
  NemoUtils::toc(tic_toc_prefix);
}
void Self_energy::get_communication_continual_scattering(const double energy_interval, const double momentum_interval,
    std::set<communication_pair>*& output_comm_table) const
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::get_communication_continual_scattering ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_communication_continual_scattering()";
  output_comm_table->clear();
  //5.0 get the spatially-global rank of this MPI-process
  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);

  //const Propagator* temp_propagator=writeable_Propagators.begin()->second;
  //1. loop over all existing momentum points
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
  std::set<vector<NemoMeshPoint> > local_mom;
  std::string local_table_str = options.get_option("local_comm_table",std::string(""));
  bool local_table ;
  if(local_table_str=="true")
  {
    local_table=true;
  }
  else
  {
    local_table=false;
  }
  std::vector <double> interval(2,0.0);
  if(local_table==true)
  {
    for(; c_it!=pointer_to_all_momenta->end(); c_it++)
    {
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it));
      if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
      {
        local_mom.insert(job_cit->first);
      }
    }
    for(unsigned int i =0; i<Mesh_tree_names.size(); i++)
    {
      if(Mesh_tree_names[i]=="energy")
      {
        interval[i]=energy_interval;

      }
      else
      {
        interval[i]=momentum_interval;
      }
    }
  }
  c_it=pointer_to_all_momenta->begin();
  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {

    bool coupling=false;
    std::set<std::vector<NemoMeshPoint> >::const_iterator local_it=local_mom.begin();
    if(local_table==true)
    {
      for(; local_it!=local_mom.end(); local_it++)
      {
        if(check_coupling(&(*local_it),&(* c_it), &interval))
        {
          coupling=true;
          break;
        }
      }
    }
    else
    {
      coupling=true;
    }
    if(coupling==true)
    {
      std::set<std::vector<NemoMeshPoint> > temp_comm_set;

      std::map<std::string,double> conditions;
      //find the energy mesh name
      std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
      std::string energy_name=std::string("");
      for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
        if(momentum_name_it->second==NemoPhys::Energy)
          energy_name=momentum_name_it->first;
      std::string condition_name=energy_name+"_interval";
      conditions[condition_name]=energy_interval;
      //find the momentum mesh name - if existing
      std::string k_vector_name=std::string("");
      momentum_name_it=momentum_mesh_types.begin();
      for (; momentum_name_it!=momentum_mesh_types.end()&&k_vector_name==std::string(""); ++momentum_name_it)
        if(momentum_name_it->second==NemoPhys::Momentum_2D||momentum_name_it->second==NemoPhys::Momentum_1D||momentum_name_it->second==NemoPhys::Momentum_3D)
          k_vector_name=momentum_name_it->first;

      condition_name=k_vector_name+"_interval";
      conditions[condition_name]=momentum_interval;
      find_specific_momenta(conditions, *c_it, temp_comm_set);

      //2. if the size of temp_comm_set>0, i.e. this momentum is interacting with some other points, add this communication pair to input_comm_table
      if(temp_comm_set.size()>0)
      {
        //save the ranks in the set to be used by MPI_NEMO_Reduce()
        communication_pair temp_pair(*c_it,temp_comm_set);
        output_comm_table->insert(temp_pair);
      }
    }
  }

  //debugging: global_job_list-output
  if(debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    std::set<communication_pair>::const_iterator temp_cit=output_comm_table->begin();
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::string filename=get_name()+"_output_comm_table_"+convert_to_string.str()+".dat";
    ofstream outfile;
    outfile.open(filename.c_str());
    for(; temp_cit!=output_comm_table->end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer,data_string);
      data_string +="\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
      while(c_it2!=temp_cit->second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
        translate_momentum_vector(temp_pointer2,convert2);
        data_string +=convert2;
        c_it2++;
        if(c_it2!=temp_cit->second.end())
        {
          data_string +=", ";
        }
      }
      outfile<<data_string<<"\n";
    }
    outfile.close();
  }
  NemoUtils::toc(tic_toc_prefix);
}

bool Self_energy::get_communication_continual_scattering_entry(const double energy_interval, const double momentum_interval,
    std::vector< communication_pair>*& output_comm_table,std::vector<std::set< std::vector< NemoMeshPoint> > >*& local_point, unsigned int& start_index) const
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::get_communication_continual_scattering_entry ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_communication_continual_scattering_entry()";
  output_comm_table->clear();
  local_point->clear();
  //5.0 get the spatially-global rank of this MPI-process
  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);

  //const Propagator* temp_propagator=writeable_Propagators.begin()->second;
  //1. loop over all existing momentum points
  std::set<std::vector<NemoMeshPoint> >::iterator c_it=pointer_to_all_momenta->begin();
  std::set<vector<NemoMeshPoint> > local_mom;

  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it));
    if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
    {
      local_mom.insert(job_cit->first);
    }
  }
  std::vector <double> interval(Mesh_tree_names.size(),0.0);
  for(unsigned int i =0; i<Mesh_tree_names.size(); i++)
  {
    if(Mesh_tree_names[i]=="energy")
    {
      interval[i]=energy_interval;

    }
    else
    {
      interval[i]=momentum_interval;
    }
  }

  c_it=pointer_to_all_momenta->begin();
  unsigned int index=0;
  for(; c_it!=pointer_to_all_momenta->end() && index<start_index; c_it++)
  {
    index++;
  }

  if(c_it==pointer_to_all_momenta->end())
  {
    return false;
  }

  bool coupling=false;
  for(; c_it!=pointer_to_all_momenta->end() && coupling==false;  c_it++,start_index++)
  {


    std::set<std::vector<NemoMeshPoint> >::const_iterator local_it=local_mom.begin();

    for(; local_it!=local_mom.end(); local_it++)
    {
      if(check_coupling(&(*local_it),&(* c_it), &interval))
      {
        coupling=true;
        break;
      }
    }

    if(coupling==true)
    {
      std::set<std::vector<NemoMeshPoint> > temp_comm_set;

      std::map<std::string,double> conditions;
      //find the energy mesh name
      std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
      std::string energy_name=std::string("");
      for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
        if(momentum_name_it->second==NemoPhys::Energy)
          energy_name=momentum_name_it->first;
      std::string condition_name=energy_name+"_interval";
      conditions[condition_name]=energy_interval;
      //find the momentum mesh name - if existing
      std::string k_vector_name=std::string("");
      momentum_name_it=momentum_mesh_types.begin();
      for (; momentum_name_it!=momentum_mesh_types.end()&&k_vector_name==std::string(""); ++momentum_name_it)
        if(momentum_name_it->second==NemoPhys::Momentum_2D||momentum_name_it->second==NemoPhys::Momentum_1D||momentum_name_it->second==NemoPhys::Momentum_3D)
          k_vector_name=momentum_name_it->first;

      condition_name=k_vector_name+"_interval";
      conditions[condition_name]=momentum_interval;

      find_specific_momenta(conditions, *c_it, temp_comm_set);

      //2. if the size of temp_comm_set>0, i.e. this momentum is interacting with some other points, add this communication pair to input_comm_table
      if(temp_comm_set.size()>0)
      {
        //save the ranks in the set to be used by MPI_NEMO_Reduce()
        communication_pair temp_pair((*c_it),temp_comm_set);

        output_comm_table->push_back(temp_pair);
        std::set<std::vector<NemoMeshPoint> > temp_local;
        std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it));
        if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
        {
          temp_local.insert(job_cit->first);
        }
        std::set<std::vector<NemoMeshPoint> >::iterator c_it_temp= temp_comm_set.begin();

        for(; c_it_temp!=temp_comm_set.end(); c_it_temp++)
        {
          std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it_temp));
          if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
          {
            temp_local.insert(job_cit->first);
          }
        }
        local_point->push_back(temp_local);
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix);
  return coupling;
}

bool Self_energy::check_coupling(const std::vector<NemoMeshPoint>* a ,const std::vector<NemoMeshPoint>* b ,std::vector<double >* interval) const
{
  for(unsigned int i=0; i<a->size(); i++)
  {
    std::vector<double> a_coords=(*a)[i].get_coords();
    std::vector<double> b_coords=(*b)[i].get_coords();
    double sqrt_distance=0.0;
    double distance=0.0;
    for(unsigned int ii=0; ii<a_coords.size(); ii++)
    {
      distance+=(a_coords[ii]-b_coords[ii])*(a_coords[ii]-b_coords[ii]);
    }
    sqrt_distance=std::sqrt(distance);
    //add those points to relevant_child_temp_points that are within the interval
    if(sqrt_distance>(*interval)[i])
    {
      return false;
    }
  }
  return true;
}
void Self_energy::get_communication_discrete_scattering(const double phonon_energy, const double momentum_interval,
    std::set<communication_pair>*& output_comm_table_emission,std::set<communication_pair>*& output_comm_table_absorption)
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::get_optical_deformation_communication ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_optical_deformation_communication()";
  output_comm_table_emission->clear();
  output_comm_table_absorption->clear();
  const Propagator* temp_propagator=writeable_Propagator; //s.begin()->second;

  //0. precalculation: prepare an inverse communication table for the absorption
  std::map<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > > temp_inverse_comm_table;
  std::map<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > > temp_inverse_comm_table2;

  std::map<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > > temp_emission_comm_table;
  std::map<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > > temp_absorption_comm_table;
  std::string energy_name=std::string("");
//   bool k_momenta_exists = false; //TODO: Use this variable

  //1. loop over all existing momentum points
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    std::set<std::vector<NemoMeshPoint> > temp_comm_set;
    //2. get the energy of the respective energy point and the momentum
    const double energy1=PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,*c_it,temp_propagator);
    std::map<std::string,double> conditions;
    //find the energy mesh name and create the proper condition for it
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
    //std::string energy_name=std::string("");
    for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Energy)
        energy_name=momentum_name_it->first;
    std::string condition_name=energy_name+"_value";
    conditions[condition_name]=energy1-phonon_energy;
    //find the momentum mesh name - if existing and create the proper condition for it
    std::string k_vector_name=std::string("");
    momentum_name_it=momentum_mesh_types.begin();
    for (; momentum_name_it!=momentum_mesh_types.end()&&k_vector_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Momentum_2D||momentum_name_it->second==NemoPhys::Momentum_1D||momentum_name_it->second==NemoPhys::Momentum_3D)
        k_vector_name=momentum_name_it->first;
    if(k_vector_name!=std::string(""))
    {
//      k_momenta_exists = true;  //TODO: Use this variable
      condition_name=k_vector_name+"_interval";
      conditions[condition_name]=momentum_interval;
    }
    //this is for emission - coupling *c_it down to a lower energy
    find_specific_momenta(conditions, *c_it, temp_comm_set);
    std::set<vector<NemoMeshPoint> >::iterator temp_it = temp_comm_set.begin();
    //double energy_1=PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,*c_it,temp_propagator);
    for(; temp_it != temp_comm_set.end();  )
    {
      const double energy2=PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,*temp_it,temp_propagator);
      if(energy2>energy1)
      {
        temp_comm_set.erase(temp_it);
        temp_it++;

      }
      else if(temp_comm_set.size()>1&&*temp_it==*c_it)
      {
        temp_comm_set.erase(temp_it);
        temp_it++;
      }
      else
        temp_it++;
    }

    if(temp_comm_set.empty())
    {
      temp_comm_set.insert(*c_it);
    }

    //3. if the size of temp_comm_set>0, i.e. this momentum is interacting with some other points, add this communication pair to output_comm_table(s)
    //NOTE: temp_comm_set is for emission only! Absorption is done by reverting temp_comm_set
    if(temp_comm_set.size()>0)
    {
      temp_emission_comm_table[*c_it] = temp_comm_set;

      //4. create the inverse of the current entries in output_comm_table_emission
      std::set<std::vector<NemoMeshPoint> >::const_iterator forward_set_it=temp_comm_set.begin();
      for(; forward_set_it!=temp_comm_set.end(); ++forward_set_it)
      {
        std::map<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > >::iterator inverse_it=temp_inverse_comm_table.find(*forward_set_it);
        if(inverse_it==temp_inverse_comm_table.end())
        {
          std::set<std::vector<NemoMeshPoint> > temp_set;
          temp_set.insert(*c_it);
          temp_inverse_comm_table[*forward_set_it]=temp_set;
        }
        else
          inverse_it->second.insert(*c_it);
      }
    }
    //get absorption
    {
      std::set<std::vector<NemoMeshPoint> > temp_comm_inverse_set;
      std::map<std::string,double> conditions2;
      condition_name=energy_name+"_value";
      conditions2[condition_name]=energy1+phonon_energy;
      if(k_vector_name!=std::string(""))
      {
        condition_name=k_vector_name+"_interval";
        conditions2[condition_name]=momentum_interval;
      }
      //this is for absorption
      find_specific_momenta(conditions2, *c_it, temp_comm_inverse_set);
      std::set<vector<NemoMeshPoint> >::iterator temp_it = temp_comm_inverse_set.begin();

      for(; temp_it != temp_comm_inverse_set.end(); )
      {
        const double energy2=PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,*temp_it,temp_propagator);
        if(energy2<energy1)
        {
          temp_comm_inverse_set.erase(temp_it);
          temp_it++;
        }
        else if(temp_comm_inverse_set.size()>1&&*temp_it==*c_it)
        {
          temp_comm_inverse_set.erase(temp_it);
          temp_it++;
        }
        else
          temp_it++;
      }
      if(temp_comm_inverse_set.empty())
      {
        temp_comm_inverse_set.insert(*c_it);
      }
      if(temp_comm_inverse_set.size()>0)
      {
        //communication_pair temp_pair(*c_it,temp_comm_inverse_set);
        //output_comm_table_absorption->insert(temp_pair);

        temp_absorption_comm_table[*c_it] = temp_comm_inverse_set;

        std::set<std::vector<NemoMeshPoint> >::const_iterator forward_set_it=temp_comm_inverse_set.begin();
        for(; forward_set_it!=temp_comm_inverse_set.end(); ++forward_set_it)
        {
          std::map<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > >::iterator inverse_it=temp_inverse_comm_table2.find(*forward_set_it);
          if(inverse_it==temp_inverse_comm_table2.end())
          {
            std::set<std::vector<NemoMeshPoint> > temp_set;
            temp_set.insert(*c_it);
            temp_inverse_comm_table2[*forward_set_it]=temp_set;
          }
          else
            inverse_it->second.insert(*c_it);
        }
      }
    }

  }
  //5. add the (now finished) inverse_comm_table to the output_comm_table_absorption
  std::map<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > >::iterator c_inverse_it=temp_inverse_comm_table.begin();
  for(; c_inverse_it!=temp_inverse_comm_table.end(); ++c_inverse_it)
  {
    //search in output_comm_table_absorption for
    std::map<vector<NemoMeshPoint>,std::set<vector<NemoMeshPoint> > >::iterator absorp_it = temp_absorption_comm_table.find(c_inverse_it->first);
    if(absorp_it==temp_absorption_comm_table.end())
      temp_absorption_comm_table[c_inverse_it->first] = c_inverse_it->second;
    else
    {
      std::set<vector<NemoMeshPoint> >::iterator set_it = c_inverse_it->second.begin();
      for(; set_it != c_inverse_it->second.end(); ++set_it)
        absorp_it->second.insert(*set_it);
      //absorp_it->second.insert(absorp_it->second.end(),c_inverse_it->second.begin(),c_inverse_it->second.end());
    }
    //5.1 translate the respective map entry into a communication_pair
    //communication_pair temp_pair(c_inverse_it->first,c_inverse_it->second);
    //5.2 add the result of 5.1 to the output_comm_table_absorption
    //output_comm_table_absorption->insert(temp_pair);


  }
  c_inverse_it=temp_inverse_comm_table2.begin();

  for(; c_inverse_it!=temp_inverse_comm_table2.end(); ++c_inverse_it)
  {
    //5.1 translate the respective map entry into a communication_pair
    //communication_pair temp_pair(c_inverse_it->first,c_inverse_it->second);
    //5.2 add the result of 5.1 to the output_comm_table_absorption
    //output_comm_table_emission->insert(temp_pair);
    std::map<vector<NemoMeshPoint>,std::set<vector<NemoMeshPoint> > >::iterator emiss_it = temp_emission_comm_table.find(c_inverse_it->first);
    if(emiss_it==temp_emission_comm_table.end())
      temp_emission_comm_table[c_inverse_it->first] = c_inverse_it->second;
    else
    {
      std::set<vector<NemoMeshPoint> >::iterator set_it = c_inverse_it->second.begin();
      for(; set_it != c_inverse_it->second.end(); ++set_it)
        emiss_it->second.insert(*set_it);
    }
  }

  //loop through absorption and emission and create the communication pairs
  std::map<vector<NemoMeshPoint>, std::set<vector<NemoMeshPoint> > >::iterator it_comm_pair = temp_emission_comm_table.begin();
  for(; it_comm_pair != temp_emission_comm_table.end(); ++it_comm_pair)
  {
    communication_pair temp_pair(it_comm_pair->first,it_comm_pair->second);
    output_comm_table_emission->insert(temp_pair);

    std::set<vector<NemoMeshPoint> >::iterator set_it = it_comm_pair->second.begin();

    const std::vector<NemoMeshPoint> temp_vector_pointer=it_comm_pair->first;
    std::set<unsigned int> Hamilton_momentum_indices;
    std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
    PropagationUtilities::find_Hamiltonian_momenta(this,writeable_Propagator,pointer_to_Hamilton_momentum_indices);
    NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
    if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=temp_vector_pointer[*(pointer_to_Hamilton_momentum_indices->begin())];

    //if there is Hamilton momenta need to store norm. factors with respect to the Hamilton momenta
    if(pointer_to_Hamilton_momentum_indices!=NULL)
    {
      for(; set_it != it_comm_pair->second.end(); ++set_it)
      {
        NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
        const std::vector<NemoMeshPoint> temp_vector_pointer2 = *set_it;
        if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=temp_vector_pointer2[*(pointer_to_Hamilton_momentum_indices->begin())];

        {
          double emiss_factor = find_integration_weight(energy_name, *set_it, temp_propagator);
          std::map<vector<NemoMeshPoint>,std::map<NemoMeshPoint,double> >::iterator e_it = emission_norm_factor.find(it_comm_pair->first);
          if(e_it!=emission_norm_factor.end())
          {
            std::map<NemoMeshPoint,double>::iterator map_it = e_it->second.find(temp_NemoMeshPoint2);
            if(map_it!=e_it->second.end())
              map_it->second += emiss_factor;
            else
              (e_it->second)[temp_NemoMeshPoint2] = emiss_factor;
          }
          else
          {
            std::map<NemoMeshPoint,double> temp_map;
            temp_map[temp_NemoMeshPoint2] = emiss_factor;
            emission_norm_factor[it_comm_pair->first] = temp_map;
          }
        }
      }

    }
    else
    {
      double emiss_factor = 0.0;
      for(; set_it != it_comm_pair->second.end(); ++set_it)
        emiss_factor += find_integration_weight(energy_name, *set_it, temp_propagator);

      std::map<NemoMeshPoint,double> temp_map;
      temp_map[temp_NemoMeshPoint] = emiss_factor;
      emission_norm_factor[it_comm_pair->first] = temp_map;
    }

  }
  it_comm_pair = temp_absorption_comm_table.begin();
  for(; it_comm_pair != temp_absorption_comm_table.end(); ++it_comm_pair)
  {
    communication_pair temp_pair(it_comm_pair->first,it_comm_pair->second);
    output_comm_table_absorption->insert(temp_pair);

    std::set<vector<NemoMeshPoint> >::iterator set_it = it_comm_pair->second.begin();
    //for(; set_it != it_comm_pair->second.end(); ++set_it)
    //  emiss_factor += find_integration_weight(energy_name, *set_it, temp_propagator);

    const std::vector<NemoMeshPoint> temp_vector_pointer=it_comm_pair->first;
    std::set<unsigned int> Hamilton_momentum_indices;
    std::set<unsigned int>* pointer_to_Hamilton_momentum_indices=&Hamilton_momentum_indices;
    PropagationUtilities::find_Hamiltonian_momenta(this,writeable_Propagator,pointer_to_Hamilton_momentum_indices);
    NemoMeshPoint temp_NemoMeshPoint(0,std::vector<double>(3,0.0));
    if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint=temp_vector_pointer[*(pointer_to_Hamilton_momentum_indices->begin())];

    //if there is Hamilton momenta need to store norm. factors with respect to the Hamilton momenta
    if(pointer_to_Hamilton_momentum_indices!=NULL)
    {
      for(; set_it != it_comm_pair->second.end(); ++set_it)
      {
        NemoMeshPoint temp_NemoMeshPoint2(0,std::vector<double>(3,0.0));
        const std::vector<NemoMeshPoint> temp_vector_pointer2 = *set_it;
        if(pointer_to_Hamilton_momentum_indices!=NULL) temp_NemoMeshPoint2=temp_vector_pointer2[*(pointer_to_Hamilton_momentum_indices->begin())];
        //double temp_energy1=PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,it_comm_pair->first,temp_propagator);
        //double temp_energy2 = PropagationUtilities::PropagationUtilities::read_energy_from_momentum(this,*set_it,temp_propagator);
        {
          double absorp_factor = find_integration_weight(energy_name, *set_it, temp_propagator);
          std::map<vector<NemoMeshPoint>,std::map<NemoMeshPoint,double> >::iterator e_it = absorption_norm_factor.find(it_comm_pair->first);
          if(e_it!=absorption_norm_factor.end())
          {
            std::map<NemoMeshPoint,double>::iterator map_it = e_it->second.find(temp_NemoMeshPoint2);
            if(map_it!=e_it->second.end())
              map_it->second += absorp_factor;
            else
              (e_it->second)[temp_NemoMeshPoint2] = absorp_factor;
          }
          else
          {
            std::map<NemoMeshPoint,double> temp_map;
            temp_map[temp_NemoMeshPoint2] = absorp_factor;
            absorption_norm_factor[it_comm_pair->first] = temp_map;
          }
        }
      }
    }
    else
    {
      double absorp_factor = 0.0;
      for(; set_it != it_comm_pair->second.end(); ++set_it)
        absorp_factor += find_integration_weight(energy_name, *set_it, temp_propagator);

      std::map<NemoMeshPoint,double> temp_map;
      temp_map[temp_NemoMeshPoint] = absorp_factor;
      absorption_norm_factor[it_comm_pair->first] = temp_map;
    }
  }

  //debugging: global_job_list-output of the emission and absorption list for each MPI rank
  if(debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    std::set<communication_pair>::const_iterator temp_cit=output_comm_table_emission->begin();
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::string filename=get_name()+"_output_comm_table_emission"+convert_to_string.str()+".dat";
    std::string filename2="emission_norm_table.dat";
    ofstream outfile;
    ofstream outfile2;
    outfile.open(filename.c_str());
    for(; temp_cit!=output_comm_table_emission->end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer,data_string);
      data_string +="\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
      while(c_it2!=temp_cit->second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
        translate_momentum_vector(temp_pointer2,convert2);
        data_string +=convert2;
        c_it2++;
        if(c_it2!=temp_cit->second.end())
        {
          data_string +=", ";
        }
      }
      outfile<<data_string<<"\n";

    }
    outfile.close();
    if(myrank==0)
    {

      outfile2.open(filename2.c_str());
      std::map<vector<NemoMeshPoint>, std::map<NemoMeshPoint, double > >::iterator temp_it = emission_norm_factor.begin();
      for(; temp_it!=emission_norm_factor.end(); temp_it++)
      {
        std::string data_string;
        const std::vector<NemoMeshPoint>* temp_pointer = &(temp_it->first);
        translate_momentum_vector(temp_pointer,data_string);
        data_string +="\t--->\t";
        std::map<NemoMeshPoint,double >::iterator temp_it2 = temp_it->second.begin();
        for(; temp_it2!=temp_it->second.end(); ++temp_it2++)
        {
          //loop through map for given E,k
          data_string += NemoUtils::nemo_to_string(temp_it2->first[2]) + " " +  NemoUtils::nemo_to_string(temp_it2->second);
          data_string += " ";


        }
        outfile2 << data_string << " \n";

      }
      outfile2.close();
    }


    temp_cit=output_comm_table_absorption->begin();
    std::stringstream convert_to_string2;
    convert_to_string2 << myrank;
    filename=get_name()+"_output_comm_table_absorption"+convert_to_string.str()+".dat";
    outfile.open(filename.c_str());
    filename2="absorption_norm_table.dat";

    for(; temp_cit!=output_comm_table_absorption->end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer,data_string);
      data_string +="\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
      while(c_it2!=temp_cit->second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
        translate_momentum_vector(temp_pointer2,convert2);
        data_string +=convert2;
        c_it2++;
        if(c_it2!=temp_cit->second.end())
        {
          data_string +=", ";
        }
      }
      outfile<<data_string<<"\n";
    }
    outfile.close();

    if(myrank==0)
    {

      outfile2.open(filename2.c_str());
      std::map<vector<NemoMeshPoint>, std::map<NemoMeshPoint, double > >::iterator temp_it = absorption_norm_factor.begin();
      for(; temp_it!=absorption_norm_factor.end(); temp_it++)
      {
        std::string data_string;
        const std::vector<NemoMeshPoint>* temp_pointer = &(temp_it->first);
        translate_momentum_vector(temp_pointer,data_string);
        data_string +="\t--->\t";
        std::map<NemoMeshPoint,double >::iterator temp_it2 = temp_it->second.begin();
        for(; temp_it2!=temp_it->second.end(); ++temp_it2++)
        {
          //loop through map for given E,k
          data_string += NemoUtils::nemo_to_string(temp_it2->first[2]) + " " +  NemoUtils::nemo_to_string(temp_it2->second);
          data_string += " ";


        }
        outfile2 << data_string << " \n";

      }
      outfile2.close();
    }

  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::print_comm_single_entry( std::vector< communication_pair>* output_comm_table, bool flag_clear)
{

  if(debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    std::vector<communication_pair>::const_iterator temp_cit=output_comm_table->begin();
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::string filename=get_name()+"_output_comm_table_"+convert_to_string.str()+".dat";
    ofstream outfile;
    if(flag_clear==true)
    {
      outfile.open(filename.c_str(),ios_base::out);
    }
    else
    {
      outfile.open(filename.c_str(),ios_base::out | ios_base::app);
    }

    for(; temp_cit!=output_comm_table->end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer,data_string);
      data_string +="\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
      while(c_it2!=temp_cit->second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
        translate_momentum_vector(temp_pointer2,convert2);
        data_string +=convert2;
        c_it2++;
        if(c_it2!=temp_cit->second.end())
        {
          data_string +=", ";
        }
      }
      outfile<<data_string<<"\n";
    }
    outfile.close();
  }
}

bool Self_energy::get_communication_discrete_scattering_entry(const double phonon_energy, const double momentum_interval,
    std::vector<communication_pair>*& output_comm_table
    ,std::vector<std::set< std::vector< NemoMeshPoint> > >*& local_point,  unsigned int& start_index)
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::get_communication_discrete_scattering_entry ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_communication_discrete_scattering_entry()";
  output_comm_table->clear();
  local_point->clear();
  const Propagator* temp_propagator=writeable_Propagator; //s.begin()->second;


  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);

  //const Propagator* temp_propagator=writeable_Propagators.begin()->second;
  //1. loop over all existing momentum points
  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();

  c_it=pointer_to_all_momenta->begin();
  int index=0;
  for(; c_it!=pointer_to_all_momenta->end() && index<(int)start_index; c_it++)
  {
    index++;
  }
  if(c_it==pointer_to_all_momenta->end())
  {
    NemoUtils::toc(tic_toc_prefix);
    return false;
  }

  //double energy1=PropagationUtilities::read_energy_from_momentum(this,*c_it,temp_propagator);
  //int energy_index=0;


  //1. loop over all existing momentum points
  bool coupling=false;
  for(; (c_it!=pointer_to_all_momenta->end())&& (!coupling); c_it++,start_index++)
  {

    std::set<std::vector<NemoMeshPoint> > temp_comm_set;

    std::set<std::vector<NemoMeshPoint> > temp_comm_set_absorp;
    //2. get the energy of the respective energy point and the momentum
    const double energy1=PropagationUtilities::read_energy_from_momentum(this,*c_it,temp_propagator);
    std::map<std::string,double> conditions;
    //find the energy mesh name and create the proper condition for it
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
    std::string energy_name=std::string("");
    for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Energy)
        energy_name=momentum_name_it->first;
    std::string condition_name=energy_name+"_value";
    conditions[condition_name]=energy1-phonon_energy;
    //find the momentum mesh name - if existing and create the proper condition for it
    std::string k_vector_name=std::string("");
    momentum_name_it=momentum_mesh_types.begin();
    for (; momentum_name_it!=momentum_mesh_types.end()&&k_vector_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Momentum_2D||momentum_name_it->second==NemoPhys::Momentum_1D||momentum_name_it->second==NemoPhys::Momentum_3D)
        k_vector_name=momentum_name_it->first;
    if(k_vector_name!=std::string(""))
    {
      condition_name=k_vector_name+"_interval";
      conditions[condition_name]=momentum_interval;
    }
    //this is for emission - coupling *c_it down to a lower energy
    find_specific_momenta(conditions, *c_it, temp_comm_set);
    condition_name=energy_name+"_value";
    conditions[condition_name]=phonon_energy;
    find_specific_momenta(conditions, *c_it, temp_comm_set_absorp,true);
    //3. if the size of temp_comm_set>0, i.e. this momentum is interacting with some other points, add this communication pair to output_comm_table(s)
    //check that in emission no coupling with higher energy
    double energy_1=PropagationUtilities::read_energy_from_momentum(this,*c_it,temp_propagator);
    std::set<std::vector<NemoMeshPoint> >::iterator c_it_temp= temp_comm_set.begin();
    for(; c_it_temp!=temp_comm_set.end();)
    {
      double energy2 = PropagationUtilities::read_energy_from_momentum(this,*c_it_temp,temp_propagator);
      if(energy2 > energy_1)
      {
        temp_comm_set.erase(c_it_temp++);
      }
      else
      {
        c_it_temp++;
      }
    }
    c_it_temp= temp_comm_set_absorp.begin();
    for(; c_it_temp!=temp_comm_set_absorp.end();)
    {
      double energy2 = PropagationUtilities::read_energy_from_momentum(this,*c_it_temp,temp_propagator);
      if(energy2 < energy_1)
      {
        temp_comm_set_absorp.erase(c_it_temp++);
      }
      else
      {
        c_it_temp++;
      }
    }
    if(temp_comm_set.size()>0)
    {

      std::set<std::vector<NemoMeshPoint> > temp_local;
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it));
      if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
      {
        temp_local.insert(job_cit->first);
      }
      c_it_temp= temp_comm_set.begin();

      for(; c_it_temp!=temp_comm_set.end(); c_it_temp++)
      {
        std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it_temp));
        if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
        {
          temp_local.insert(job_cit->first);
        }
      }
      if(temp_local.size()>0)
      {
        coupling=true;
        local_point->push_back(temp_local);
        communication_pair temp_pair(*c_it,temp_comm_set);
        output_comm_table->push_back(temp_pair);
      }
    }
    if(temp_comm_set_absorp.size()>0)
    {
      std::set<std::vector<NemoMeshPoint> > temp_local;
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it));
      if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
      {
        temp_local.insert(job_cit->first);
      }
      std::set<std::vector<NemoMeshPoint> >::iterator c_it_temp= temp_comm_set_absorp.begin();

      for(; c_it_temp!=temp_comm_set_absorp.end(); c_it_temp++)
      {
        std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it_temp));
        if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
        {
          temp_local.insert(job_cit->first);
        }
      }
      if(temp_local.size()>0)
      {
        coupling=true;
        local_point->push_back(temp_local);
        communication_pair temp_pair(*c_it,temp_comm_set_absorp);
        output_comm_table->push_back(temp_pair);
      }

    }
  }
  NemoUtils::toc(tic_toc_prefix);
  return coupling;

}



bool Self_energy::get_communication_discrete_scattering_entry(const double phonon_energy, const double momentum_interval,
    std::vector<communication_pair>*& output_comm_table
    ,std::vector<std::set< std::vector< NemoMeshPoint> > >*& local_point)
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::get_communication_discrete_scattering_entry ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_communication_discrete_scattering_entry()";
  output_comm_table->clear();
  local_point->clear();

  // determine the communication table due to the scattering type
  std::set<communication_pair> scattering_communication_table;
  std::set<communication_pair> * scattering_communication_table_ptr = &scattering_communication_table ;
  std::set<communication_pair> scattering_communication_table_absorp;
  std::set<communication_pair> * scattering_communication_table_absorp_ptr = &scattering_communication_table_absorp;
  bool coupling = false;
  get_communication_discrete_scattering(phonon_energy,momentum_interval,
      scattering_communication_table_ptr , scattering_communication_table_absorp_ptr);

  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);
  std::set<std::vector<NemoMeshPoint> > temp_local;
  std::set<communication_pair>::iterator scattering_communication_table_it = scattering_communication_table_ptr->begin();
  for(;scattering_communication_table_it!=scattering_communication_table_ptr->end();scattering_communication_table_it++)
  {
    temp_local.clear();
    std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find(scattering_communication_table_it->first);
    if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
    {
      temp_local.insert(job_cit->first);
    }
    std::set<std::vector<NemoMeshPoint> >::iterator c_it_temp= scattering_communication_table_it->second.begin();
    for(; c_it_temp!=scattering_communication_table_it->second.end(); c_it_temp++)
    {
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it_temp));
      if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
      {
        temp_local.insert(job_cit->first);
      }
    }
    if(temp_local.size()>0)
    {
      coupling=true;
      local_point->push_back(temp_local);
      output_comm_table->push_back(*scattering_communication_table_it);
    }
  }
  scattering_communication_table_it=scattering_communication_table_absorp_ptr->begin();
  for(;scattering_communication_table_it!=scattering_communication_table_absorp_ptr->end();scattering_communication_table_it++)
  {
    temp_local.clear();
    std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find(scattering_communication_table_it->first);
    if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
    {
      temp_local.insert(job_cit->first);
    }
    std::set<std::vector<NemoMeshPoint> >::iterator c_it_temp= scattering_communication_table_it->second.begin();
    for(; c_it_temp!=scattering_communication_table_it->second.end(); c_it_temp++)
    {
      std::map<std::vector<NemoMeshPoint>, int>::const_iterator job_cit=global_job_list.find((*c_it_temp));
      if(job_cit!=global_job_list.end()&& job_cit->second== my_local_rank)
      {
        temp_local.insert(job_cit->first);
      }
    }
    if(temp_local.size()>0)
    {
      coupling=true;
      local_point->push_back(temp_local);
      output_comm_table->push_back(*scattering_communication_table_it);
    }
  }
  NemoUtils::toc(tic_toc_prefix);
  return coupling;
}



void Self_energy::get_communication_readin_discrete_scattering(const double momentum_interval, std::set<communication_pair>*& output_comm_table_emission,
    std::set<communication_pair>*& output_comm_table_absorption)
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::get_communication_readin_discrete_scattering ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::get_communication_readin_discrete_scattering()";
  output_comm_table_emission->clear();
  output_comm_table_absorption->clear();

  //set up a comparison struct -- can move somewhere else if this is useful
  //assumption is range.first < range.second


  //file name
  std::string absorp_filename = options.get_option("absorption_readin_name", std::string("absorption_table.dat"));
  std::string emiss_filename = options.get_option("emission_readin_name", std::string("emission_table.dat"));

  //open file for absorption and put into map.
  ifstream absorp_file;
  absorp_file.open(absorp_filename.c_str(), ifstream::in);
  absorp_file.precision(16);
  //key initial energy, value final energy -- 1 to 1 but the file does not need unique inital values.
  double tolerance = 1E-4;//look +-tolerance
  std::map<NemoMath::double_range, set<double>, NemoMath::CompareDoubleRange > absorption_energy_map;
  double initial_energy = 0.0;
  double final_energy = 0.0;
  while(absorp_file >> initial_energy >> final_energy)
  {
    std::map<NemoMath::double_range, set<double>, NemoMath::CompareDoubleRange>::iterator absorp_it = absorption_energy_map.find(
          NemoMath::make_range(initial_energy-tolerance,initial_energy+tolerance));
    if(absorp_it == absorption_energy_map.end())
    {
      set<double> temp_set;
      temp_set.insert(final_energy);
      absorption_energy_map[NemoMath::make_range(initial_energy-tolerance,initial_energy+tolerance)] = temp_set;
    }
    else
    {
      absorp_it->second.insert(final_energy);
    }
  }
  absorp_file.close();

  //open file for emission and put into map.
  ifstream emiss_file;
  emiss_file.open(emiss_filename.c_str(), ifstream::in);
  emiss_file.precision(16);
  //key initial energy, value final energy -- 1 to 1 but the file does not need unique inital values.

  std::map<NemoMath::double_range, set<double>, NemoMath::CompareDoubleRange > emission_energy_map;
  initial_energy = 0.0;
  final_energy = 0.0;
  while(emiss_file >> initial_energy >> final_energy)
  {
    std::map<NemoMath::double_range, set<double>, NemoMath::CompareDoubleRange >::iterator emiss_it = emission_energy_map.find(
          NemoMath::make_range(initial_energy-tolerance,initial_energy+tolerance));
    if(emiss_it == emission_energy_map.end())
    {
      set<double> temp_set;
      temp_set.insert(final_energy);
      emission_energy_map[NemoMath::make_range(initial_energy-tolerance,initial_energy+tolerance)] = temp_set;
    }
    else
    {
      emiss_it->second.insert(final_energy);
    }
  }
  emiss_file.close();

  //loop through all momenta and check for energy in emission and absorption map

  std::set<std::vector<NemoMeshPoint> >::const_iterator c_it=pointer_to_all_momenta->begin();
  for(; c_it!=pointer_to_all_momenta->end(); c_it++)
  {
    std::set<std::vector<NemoMeshPoint> > absorp_comm_set;
    std::set<std::vector<NemoMeshPoint> > emiss_comm_set;
    //2. get the energy of the respective energy point and the momentum
    const double energy1=PropagationUtilities::read_energy_from_momentum(this,*c_it,writeable_Propagator);
    std::map<std::string,double> conditions;
    //find the energy mesh name and create the proper condition for it
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
    std::string energy_name=std::string("");
    for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Energy)
        energy_name=momentum_name_it->first;
    std::string condition_name=energy_name+"_value";
    //find the momentum mesh name - if existing and create the proper condition for it
    std::string k_vector_name=std::string("");
    momentum_name_it=momentum_mesh_types.begin();
    for (; momentum_name_it!=momentum_mesh_types.end()&&k_vector_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Momentum_2D||momentum_name_it->second==NemoPhys::Momentum_1D||momentum_name_it->second==NemoPhys::Momentum_3D)
        k_vector_name=momentum_name_it->first;
    if(k_vector_name!=std::string(""))
    {
      condition_name=k_vector_name+"_interval";
      conditions[condition_name]=momentum_interval;
    }
    std::string condition_name2=energy_name+"_value";
    //absorption
    set<double> absorp_set = absorption_energy_map[NemoMath::make_range(energy1-tolerance,energy1+tolerance)];
    NEMO_ASSERT(!absorp_set.empty(), prefix + " absorp_set is empty");
    //loop through the absorp_set in order to find the value to insert into the absorp_comm_set
    set<double>::iterator absorp_set_it = absorp_set.begin();
    for(; absorp_set_it!=absorp_set.end(); ++absorp_set_it)
    {
      std::set<std::vector<NemoMeshPoint> > temp_comm_set;
      conditions.erase(condition_name2);
      conditions[condition_name2]=*absorp_set_it;
      //this is for emission - coupling *c_it down to a lower energy
      find_specific_momenta(conditions, *c_it, temp_comm_set);
      //NEMO_ASSERT(temp_comm_set.size()==1,prefix+ ": temp_comm_set came back with more than one value, expected one-to-one");
      absorp_comm_set.insert(temp_comm_set.begin(),temp_comm_set.end());
    }

    if(!absorp_comm_set.empty())
    {
      communication_pair temp_pair(*c_it,absorp_comm_set);
      output_comm_table_absorption->insert(temp_pair);
    }

    //emission
    set<double> emiss_set = emission_energy_map[NemoMath::make_range(energy1-tolerance,energy1+tolerance)];
    NEMO_ASSERT(!emiss_set.empty(), prefix + " emiss_set is empty");
    //loop through the absorp_set in order to find the value to insert into the emiss_comm_set
    set<double>::iterator emiss_set_it = emiss_set.begin();//emiss_energy_it->second.begin();
    for(; emiss_set_it!=emiss_set.end(); ++emiss_set_it)
    {
      std::set<std::vector<NemoMeshPoint> > temp_comm_set;
      conditions.erase(condition_name2);
      conditions[condition_name2]=*emiss_set_it;
      //this is for emission - coupling *c_it down to a lower energy
      find_specific_momenta(conditions, *c_it, temp_comm_set);
      //NEMO_ASSERT(temp_comm_set.size()==1,prefix+ ": temp_comm_set came back with more than one value, expected one-to-one");
      emiss_comm_set.insert(temp_comm_set.begin(),temp_comm_set.end());
    }

    if(!emiss_comm_set.empty())
    {
      communication_pair temp_pair(*c_it,emiss_comm_set);
      output_comm_table_emission->insert(temp_pair);
    }

  }

  //debugging: global_job_list-output of the emission and absorption list for each MPI rank
  if(debug_output_job_list&&!no_file_output)
  {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    std::set<communication_pair>::const_iterator temp_cit=output_comm_table_emission->begin();
    std::stringstream convert_to_string;
    convert_to_string << myrank;
    std::string filename=get_name()+"_output_comm_table_readin_emission"+convert_to_string.str()+".dat";
    ofstream outfile;
    outfile.open(filename.c_str());
    for(; temp_cit!=output_comm_table_emission->end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer,data_string);
      data_string +="\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
      while(c_it2!=temp_cit->second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
        translate_momentum_vector(temp_pointer2,convert2);
        data_string +=convert2;
        c_it2++;
        if(c_it2!=temp_cit->second.end())
        {
          data_string +=", ";
        }
      }
      outfile<<data_string<<"\n";
    }
    outfile.close();

    temp_cit=output_comm_table_absorption->begin();
    std::stringstream convert_to_string2;
    convert_to_string2 << myrank;
    filename=get_name()+"_output_comm_table_readin_absorption"+convert_to_string.str()+".dat";
    outfile.open(filename.c_str());
    for(; temp_cit!=output_comm_table_absorption->end(); temp_cit++)
    {
      std::string data_string;
      const std::vector<NemoMeshPoint>* temp_pointer = &(temp_cit->first);
      translate_momentum_vector(temp_pointer,data_string);
      data_string +="\t--->\t";
      std::set<std::vector<NemoMeshPoint> >::const_iterator c_it2=temp_cit->second.begin();
      while(c_it2!=temp_cit->second.end())
      {
        std::string convert2;
        const std::vector<NemoMeshPoint>* temp_pointer2=&(*c_it2);
        translate_momentum_vector(temp_pointer2,convert2);
        data_string +=convert2;
        c_it2++;
        if(c_it2!=temp_cit->second.end())
        {
          data_string +=", ";
        }
      }
      outfile<<data_string<<"\n";
    }
    outfile.close();
  }

  NemoUtils::toc(tic_toc_prefix);
}
void Self_energy::find_biggest_communication_pair(const std::set<communication_pair>* input_set,
    std::set<communication_pair>::const_iterator iterator_to_biggest_pair) const
{
  std::string temp_tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+temp_tic_toc_name+"\")::find_biggest_communication_pair ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+this->get_name()+"\")::find_biggest_communication_pair() ";
  NEMO_ASSERT(input_set!=NULL,prefix+"received NULL pointer instead of the input_set\n");
  //1. loop over the set entries
  std::set<communication_pair>::const_iterator c_it=input_set->begin();
  iterator_to_biggest_pair = c_it;
  for(; c_it!=input_set->end(); c_it++)
  {
    //2. if current element is bigger then the stored one, take it...
    if((*c_it).second.size()>(*iterator_to_biggest_pair).second.size())
      iterator_to_biggest_pair=c_it;
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::calculate_scattering_rate(const Propagator* retarded_self, const std::vector<NemoMeshPoint>& momentum)
{
	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::calculate_scattering_rate ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(\"" + this->get_name() + "\")::calculate_scattering_rate() ";

	//debug Fabio
	//double e = PropagationUtilities::read_energy_from_momentum(this, momentum, retarded_self);
	//cout << "[ENERGY]   " << e << "\n";

	//estimate the integration step size dz - assuming that all atoms are equally separated...
	//get the unit cell volume and divide by the number of atoms in it
	const Domain* temp_domain = Hamilton_Constructor->get_const_simulation_domain();
	//double volume=temp_domain->return_periodic_space_volume();  //lattice_constant^2
	std::vector <std::vector <double> > trans_vectors = temp_domain->get_translation_vectors();
	//get the lattice constant. Used for kz mesh construction
	double lattice_const = 0.5;
	if (trans_vectors[0][0] == trans_vectors[1][1])
		lattice_const = trans_vectors[0][0];
	else if (trans_vectors[1][1] == trans_vectors[2][2])
		lattice_const = trans_vectors[1][1];
	else if (trans_vectors[2][2] == trans_vectors[0][0])
		lattice_const = trans_vectors[2][2];
	//else 
	//  throw std::runtime_error(prefix+"lattice_const is not defined\n");
	//double average_atom_distance=0.0;//to be defined in 3.1

	//1. get the scattering self-energy matrix according to the retarded_self and the momentum
	Simulation* source = find_source_of_data(retarded_self->get_name());
	PetscMatrixParallelComplex* self_energy_matrix = NULL;
	source->get_data(retarded_self->get_name(), &momentum, self_energy_matrix);
	//2. get the energy and the k-vector from the momentum
	//double energy=PropagationUtilities::read_energy_from_momentum(this,momentum, retarded_self);
	std::vector<double> momentum_point(3, 0.0);
	if (options.get_option("pop_wire_scattering_rate_temp_fix", false))
	{
		//
	}
	else
	{
		momentum_point = read_kvector_from_momentum(momentum, retarded_self);
	}
	//3. mesh the k-spaces that correspond to the propagation coordinate distances
	//3.1 analyse the required k-space dimensionality
	const std::vector<std::vector <double> >&  k_basis_vector = Hamilton_Constructor->get_const_simulation_domain()->get_reciprocal_vectors();

	if (!options.get_option("pop_wire_scattering_rate_temp_fix", false))
	{
		NEMO_ASSERT(k_basis_vector.size()>0 && k_basis_vector.size()<3, prefix + "wrong dimensionality of the k-basis\n");
	}

	std::set<std::vector<double> > all_momentum_vectors;
	//3.1a) if only one momentum direction is missing
	if (k_basis_vector.size() == 2)
	{
		//3.1a).1 get the missing momentum direction
		std::vector<double> new_k_direction(3, 0.0);
		bool rotational_symmetry = options.get_option("scattering_rate_rotational_symmetry", bool(false));
		//int nonzero_j = -1;
		//int nonzero_i = -1;
		//for(unsigned int i = 0; i < k_basis_vector.size(); i++)
		//  for(unsigned int j = 0; j < k_basis_vector[i].size(); j++)
		//  {
		//   if(k_basis_vector[i][j] >0.0)
		//    {
		//      nonzero_i = i;
		//      nonzero_j = j;
		//    }
		//  }

		NemoMath::vector_outer_product(&(k_basis_vector[0][0]), &(k_basis_vector[1][0]), &(new_k_direction[0]));
		if (rotational_symmetry)
		{
			new_k_direction[2] = new_k_direction[0];
			new_k_direction[0] = 0.0;
		}
		//3.1a).2 assemble all_momentum_vectors
		const int number_of_k_points = options.get_option("number_of_k_points_for_scattering_rate", 10);
		for (int i = 0; i<number_of_k_points; i++)
		{
			std::vector<double> temp_vector(new_k_direction);
			//double factor = i/double(number_of_k_points)/NemoMath::vector_norm_3d(&(new_k_direction[0]));
			double kz_momentum_point = i / double(number_of_k_points)*NemoMath::pi / lattice_const;
			for (unsigned int j = 0; j<3; j++)
			{
				// temp_vector[j]=temp_vector[j]*factor+momentum_point[j];
				if (j == 2)
					temp_vector[j] = kz_momentum_point;
				else
					temp_vector[j] = momentum_point[j];
			}
			all_momentum_vectors.insert(temp_vector);
		}
		//average_atom_distance=volume;//divided by the number of atoms in the unit cell
	}
	else
	{
		//3.1b).1 get the two missing momentum directions...
		if (k_basis_vector.size() == 0 && options.get_option("pop_wire_scattering_rate_temp_fix", false))
		{
			//3.1b).2 assemble all_momentum_vectors
			double lattice_const = options.get_option("lattice_const", 1.0); //[nm]
			if (options.get_option("use_kz_mesh", false))
			{
				const int number_of_k_points = options.get_option("number_of_k_points_for_scattering_rate", 10);
				for (int i = 0; i < number_of_k_points; i++)
				{
					double k_max = NemoMath::pi / lattice_const;
					std::vector<double> temp_vector(3, 0.0);
					for (unsigned int j = 0; j < 3; j++)
					{
						if (j == 0)
						{
							temp_vector[j] = (i - 1) * k_max / (number_of_k_points - 1) + momentum_point[j];
						}
						else
						{
							temp_vector[j] = momentum_point[j];
						}
					}
					all_momentum_vectors.insert(temp_vector);
				}
			}
			else
			{
				double energy = PropagationUtilities::read_energy_from_momentum(this, momentum, retarded_self);
				double lattice_const = options.get_option("lattice_const", 1.0); //[nm]
				double lattice_const_in_meter = lattice_const * 1e-9;
				double mass = options.get_option("effective_mass", 0.067);
				double mstar = mass * NemoPhys::electron_mass;
				double temp = 1 - (energy*NemoPhys::elementary_charge*mstar*lattice_const_in_meter*lattice_const_in_meter) / (NemoPhys::hbar*NemoPhys::hbar);
				double kz = (1 / lattice_const)*std::acos(temp);
				std::vector<double> temp_vector(3, 0.0);
				temp_vector[0] = kz;
				all_momentum_vectors.insert(temp_vector);
			}
			//average_atom_distance = lattice_const;//divided by the number of atoms in the unit cell
		}
		else
		{
			//3.1a).2 assemble all_momentum_vectors
			const int number_of_k_points = options.get_option("number_of_k_points_for_scattering_rate", 10);
			if (options.get_option("use_kz_mesh", false))
			{
				for (int i = 0; i<number_of_k_points; i++)
				{
					std::vector<double> temp_vector(3, 0.0);
					double kz_momentum_point = i / double(number_of_k_points)*NemoMath::pi / lattice_const;
					for (unsigned int j = 0; j<3; j++)
					{
						if (j == 0)
							temp_vector[j] = kz_momentum_point;
						else
							temp_vector[j] = momentum_point[j];
					}
					all_momentum_vectors.insert(temp_vector);
				}
			}
			else
			{
				double energy = PropagationUtilities::read_energy_from_momentum(this, momentum, retarded_self);
				std::vector<double> kparallel = PropagationUtilities::read_kvector_from_momentum(this, momentum, retarded_self);
				double lattice_const = options.get_option("lattice_const", 1.0); //[nm]
				//double lattice_const_in_meter = lattice_const * 1e-9;
				double mass = options.get_option("effective_mass", 0.067);
				double mstar = mass * NemoPhys::electron_mass;
				unsigned int confinement_mat_size = options.get_option("confinement_matrix_size", 1);
				double equant = 10 * std::exp(-0.55*confinement_mat_size) + 0.55;
				double kz = std::sqrt(2 * mstar*(energy - equant)*NemoPhys::elementary_charge / (NemoPhys::hbar*NemoPhys::hbar*1e9*1e9) - (kparallel[2] * kparallel[2]));
				std::vector<double> temp_vector(3, 0.0);
				temp_vector[0] = kz;
				all_momentum_vectors.insert(temp_vector);
			}


		}
	}

	//Decide whether to do a mode space transformation or not (This is needed for obtaining scattering rates of UTB/nanowire for non-local scattering)
	bool mode_space_transformation = options.get_option("scattering_rate_mode_space_transformation", false);
	if (mode_space_transformation)
	{
		std::string prefix = "Self_energy(\"" + get_name() + "\")::calculate_scattering_rate ";

		//PetscMatrixParallelComplex* self_energy_matrix_ms = NULL;
		std::vector<unsigned int> mode1, mode2;
		options.get_option("initial_mode", mode1);
		options.get_option("final_mode", mode2);

		//Solve for the cross section modes with auxiliary Schroedinger solver
		std::string schroedi_sim_name = options.get_option("mode_space_schroedinger_solver", std::string(""));
		NEMO_ASSERT(!schroedi_sim_name.empty(), prefix + "specify mode space schroedinger solver\n");
		Schroedinger* ms_schroedi_solver = NULL;
		std::vector<std::vector<std::complex<double> > > ms_eigenfunctions;
		ms_schroedi_solver = dynamic_cast<Schroedinger*>(find_simulation(schroedi_sim_name));
		ms_schroedi_solver->solve();
		ms_schroedi_solver->get_data("eigenfunctions", ms_eigenfunctions);
		unsigned int number_of_eigenvalues = ms_eigenfunctions[0].size();
		unsigned int number_of_vectors_size = ms_eigenfunctions[0].size();

		//Obtain number of slabs in the transport direction
		const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*> (this->get_simulation_domain());
		const AtomicStructure&  atoms = domain->get_atoms();
		ConstActiveAtomIterator it = atoms.active_atoms_begin();
		unsigned int num_atoms = atoms.active_atoms_size();
		const AtomStructNode& nd = it.node();
		const Atom*           atom = nd.atom;
		HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (Hamilton_Constructor->get_material_properties(atom->get_material()));
		unsigned int num_orbitals = tb_ham->get_number_of_orbitals(atom);
		unsigned int mat_size = num_atoms*num_orbitals;
		unsigned int num_slabs = mat_size / number_of_vectors_size;
		unsigned int number_of_rows = num_slabs*number_of_eigenvalues;
		unsigned int number_of_cols = mat_size;

		//Initialize the transformation matrix
		PetscMatrixParallelComplex* trans_matrix = new PetscMatrixParallelComplex(number_of_rows, number_of_cols, get_simulation_domain()->get_communicator());
		trans_matrix->set_num_owned_rows(number_of_rows);
		for (unsigned int i = 0; i<number_of_rows; i++)
			trans_matrix->set_num_nonzeros(i, number_of_cols, 0);
		trans_matrix->allocate_memory();
		trans_matrix->set_to_zero();

		//Setup the transformation matrix
		unsigned int num_modes_per_slab = number_of_eigenvalues;
		cplx temp_val(0.0, 0.0);
		for (unsigned int idx_slab = 0; idx_slab < num_slabs; idx_slab++)
		{
			for (unsigned int i = idx_slab*num_modes_per_slab; i < num_modes_per_slab + idx_slab*num_modes_per_slab; i++)
			{
				for (unsigned int j = 0; j < number_of_vectors_size; j++)
				{
					temp_val = ms_eigenfunctions[i][j];
					if (abs(temp_val) > 1e-8)
						trans_matrix->set(j + num_slabs*number_of_eigenvalues, i + num_slabs*number_of_eigenvalues, temp_val);
				}
			}
		}

		//Perform multiplication
		/* PetscMatrixParallelComplex* temp_matrix = NULL;
		PetscMatrixParallelComplex* cj_temp = NULL;
		PetscMatrixParallelComplex::mult(*Matrix, *transformation_matrix_right, &temp_matrix); //A*V2
		temp_matrix->matrix_convert_dense();
		cj_temp = new PetscMatrixParallelComplex(transformation_matrix_left->get_num_cols(),
		transformation_matrix_left->get_num_rows(),
		get_simulation_domain()->get_communicator());
		transformation_matrix_left->hermitian_transpose_matrix(*cj_temp, MAT_INITIAL_MATRIX); // V1'
		PetscMatrixParallelComplex::mult(*cj_temp, *temp_matrix, &transformed_matrix); //V1'*A*V2*/
	}


	//4. perform the 3D Fourier transformation (note the spatial parallelization) of the spatial coordinates of the self-energy into the respective momenta (while looping over atoms)
	//4.1 get the sparsity pattern of the self-energy matrix
	const std::set<std::pair<int, int> >* pointer_to_set_of_row_col_indices = &set_of_row_col_indices;
	source->get_data(std::string("sparsity_pattern"), pointer_to_set_of_row_col_indices);
	if (pointer_to_set_of_row_col_indices == NULL)
		pointer_to_set_of_row_col_indices = &set_of_row_col_indices;
	NEMO_ASSERT(pointer_to_set_of_row_col_indices != NULL, prefix + "solver \"" + source->get_name() + "\" return NULL for the sparsity pattern\n");

	//for later store all relevant row indices
	std::set<int> set_of_rows;
	for (std::set<std::pair<int, int> >::const_iterator temp_c_it = pointer_to_set_of_row_col_indices->begin(); temp_c_it != pointer_to_set_of_row_col_indices->end(); temp_c_it++)
		set_of_rows.insert((*temp_c_it).first);


	//result container
	//note: storing the results as a function of the first atom is an approximation,
	//      since the center of the propagation is moving when one propagation coord. is fixed and the other is moving
	std::map<std::vector<double>, std::map<unsigned int, double> > result_map; //key=momentum, value=map<key=DOF-number, value=scattering rate>

																			   //4.2 loop over all atoms
	const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*> (this->get_simulation_domain());
	const AtomicStructure&  atoms = domain->get_atoms();
	ConstActiveAtomIterator it = atoms.active_atoms_begin();
	ConstActiveAtomIterator end = atoms.active_atoms_end();
	const DOFmapInterface&           dof_map = Hamilton_Constructor->get_dof_map();
	//HamiltonConstructor* tb_ham = NULL;

	for (; it != end; ++it)
	{
		const AtomStructNode& nd = it.node();
		const Atom*           atom = nd.atom;
		HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (Hamilton_Constructor->get_material_properties(atom->get_material()));
		if (tb_ham->get_number_of_orbitals(atom)>1)
			atomic_output_only = true;
		const unsigned int    atom_id = it.id();

		//position of the first atom
		std::vector<double> position1(3, 0.0);
		for (unsigned int i = 0; i<3; i++)
			position1[i] = nd.position[i];

		//4.2.1 loop over all orbitals
		const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
		map<short, unsigned int>::const_iterator c_it = atom_dofs->begin();
		for (; c_it != atom_dofs->end(); c_it++)
		{
			const unsigned int row_index = c_it->second;

			//debug Fabio
			//cout << "atom id:  " << atom_id << "  [row index]  " << row_index << "\n";

			//4.2.2 check whether this DOF is in the rows of set_of_row_col_indices, if so...
			if (set_of_rows.find(row_index) != set_of_rows.end())
			{
				//4.2.3 loop over all atoms 2
				ConstActiveAtomIterator it2 = atoms.active_atoms_begin();
				for (; it2 != end; ++it2)
				{
					const unsigned int    atom_id2 = it2.id();
					//position of the second atom
					std::vector<double> position2(3, 0.0);
					for (unsigned int i = 0; i<3; i++)
						position2[i] = it2.node().position[i];

					//4.2.4 loop over all orbitals 2
					const map<short, unsigned int>* atom_dofs2 = dof_map.get_atom_dof_map(atom_id2);
					map<short, unsigned int>::const_iterator c_it2 = atom_dofs2->begin();
					for (; c_it2 != atom_dofs2->end(); c_it2++)
					{
						//4.2.5 decide which element to take
						int col_to_take = c_it2->second;
						int sum_row_col = 2 * (int)row_index;
						int row_index2 = (int)c_it2->second;
						int row_to_take = std::abs(sum_row_col - row_index2);

						//4.2.6 decide if need to take it
						bool take_it = false;
						std::pair<int, int> temp_index_pair(row_to_take, col_to_take);
						if (pointer_to_set_of_row_col_indices->find(temp_index_pair) != pointer_to_set_of_row_col_indices->end())
							if (row_to_take + col_to_take == sum_row_col)
								take_it = true;

						if (take_it)
						{
							//4.2.7 calculate the position distance
							std::vector<double> temp_distance(position1);
							for (unsigned int i = 0; i<3; i++)
								temp_distance[i] -= position2[i];

							//4.2.8 loop over all momenta
							std::set<std::vector<double> >::const_iterator momentum_it = all_momentum_vectors.begin();
							for (; momentum_it != all_momentum_vectors.end(); momentum_it++)
							{
								//4.2.9 calculate the position distance, the Fourier factor and store with respect to the half of the position sum
								std::complex<double> exponent = std::complex<double>(0.0, NemoMath::vector_scalar_product_3d(&(temp_distance[0]), &((*momentum_it)[0])));
								std::complex<double> temp_result = std::exp(exponent)*self_energy_matrix->get(row_to_take, col_to_take);
								temp_result *= -2 / NemoPhys::hbar*NemoPhys::elementary_charge;

								//factor of 2 needed for non local where scattering rate is integrated along anti-diagonal
                std::string scattering_type = options.get_option("scattering_type", std::string(""));
								if (scattering_type == "polar_optical_Froehlich")
								{
									temp_result *= 2.0;
								}

								//debug Fabio
								/*std::complex<double> element = self_energy_matrix->get(row_index, c_it2->second);
								cout << "getting element (i,j) from sigmaR  " << "i = " << row_to_take << "   j = " << col_to_take << "||    element is: ";
								cout << element.real() << " + i" << element.imag() << "\n";
								*/

								std::map<std::vector<double>, std::map<unsigned int, double> >::iterator temp_result_it = result_map.find(*momentum_it);
								if (temp_result_it == result_map.end())
								{
									std::map<unsigned int, double> temp_map;
									temp_map[row_index] = temp_result.imag();
									result_map[*momentum_it] = temp_map;
									//double energy = PropagationUtilities::read_energy_from_momentum(this,momentum,retarded_self);
									//std::stringstream temp_stream;
									//temp_stream << "scattering rate :" << temp_result.imag() << " for energy : " << energy << " \n";
									//cerr << "scattering rate :" << temp_result.imag() << " for energy : " << energy << " \n";
									//msg.print_message(NemoUtils::MsgLevel(1),temp_stream.str());
								}
								else
								{
									std::map<unsigned int, double>::iterator temp_it2 = temp_result_it->second.find(row_index);
									if (temp_it2 == temp_result_it->second.end())
									{
										temp_result_it->second[row_index] = temp_result.imag();
									}
									else
										temp_it2->second += temp_result.imag();
								}
							}
						}
					}
				}
			}
		}
	}
	//5. save the result to file using print_atomic_map, filename is according to momentum...
	std::map<std::vector<double>, std::map<unsigned int, double> >::const_iterator print_c_it = result_map.begin();
	for (; print_c_it != result_map.end(); print_c_it++)
	{
		std::string file_name = "";
		double energy = PropagationUtilities::read_energy_from_momentum(this, momentum, retarded_self);
		std::stringstream temp_stream;
		temp_stream << energy << "_";
		translate_momentum_vector(&(print_c_it->first), file_name);
		boost::filesystem::create_directory("./scattering_rate");
		file_name = "./scattering_rate/" + retarded_self->get_name() + temp_stream.str() + file_name + "scatt_rate";
		print_atomic_map(print_c_it->second, file_name);

		//debug by Fabio
		/*std::ofstream f("map.dat");
		std::map<unsigned int, double>::const_iterator fabio = print_c_it->second.begin();
		for (; fabio != print_c_it->second.end(); fabio++)
		{
		unsigned int a = fabio->first;
		double b = fabio->second;
		f << a << "     " << b << '\n';
		}
		f.close();*/
	}
	NemoUtils::toc(tic_toc_prefix);
}



void Self_energy::get_data(const std::string& variable, std::vector <unsigned int>& data)
{
  std::string prefix = "Self_energy(\""+get_name()+"\")::get_data ";
  //1. check that the variable name makes sense
  //(i.e. corresponds to a writeable Propagator and includes the keyword "nonlocal_zeros" or "local_zeros"
  //std::map<std::string, Propagator*>::const_iterator c_it=writeable_Propagators.begin();
  //for(; c_it!=writeable_Propagators.end(); ++c_it)
  //  if(variable.find(c_it->first)!=std::string::npos) break;

  //NEMO_ASSERT(c_it!=writeable_Propagators.end(),prefix+"have not found the Self-energy that corresponds to the variable \""+variable+"\"\n");
  NEMO_ASSERT(name_of_writeable_Propagator.find(variable)!=std::string::npos,
              prefix+"have not found the Self-energy that corresponds to the variable \""+variable+"\"\n");
  if(variable.find("contact")!=std::string::npos)
  {
    //2. get the DOFs of the surface atoms coupled to the neighbor domain atoms
    //const Domain* neighbor_domain_pointer;
    std::string variable_name=name_of_writeable_Propagator+std::string("_lead_domain");
    std::string neighbor_domain_name;
    if (options.check_option(variable_name))
      neighbor_domain_name=options.get_option(variable_name,std::string(""));
    else
      throw std::invalid_argument(prefix+" define \""+variable_name+"\"\n");
    const Domain* neighbor_domain=Domain::get_domain(neighbor_domain_name);
    std::set<unsigned int> relevant_dofs;
    Hamilton_Constructor->get_data(std::string("coupling_DOFs"), neighbor_domain, relevant_dofs);
    NEMO_ASSERT(relevant_dofs.size()>0,prefix
                +"called on a process without local coupling dofs\n"); //tolerable for Petsc-distributed self-energies (not for OMEN-compression)
    //3. prepare the result container (allocate memory and fill with data)
    //3.1 prepare data
    const DOFmapInterface& temp_dofmap = Hamilton_Constructor->get_dof_map();
    unsigned int number_of_local_rows=temp_dofmap.get_number_of_dofs();
    data.clear();
    data.resize(number_of_local_rows,0);
    //3. loop over the relevant_dofs
    unsigned int relevant_dofs_size=relevant_dofs.size();
    std::set<unsigned int>::const_iterator set_cit=relevant_dofs.begin();
    for(; set_cit!=relevant_dofs.end(); ++set_cit)
    {
      //3.1 if local nonzeros: pick data index corresponding to relevant_dof and fill with relevant_dofs size
      if(variable.find("local_nonzeros")!=std::string::npos)
        data[*set_cit]=relevant_dofs_size;
      else if(variable.find("nonlocal_nonzeros")!=std::string::npos)
        data[*set_cit]=0;
      else
        throw std::invalid_argument(prefix+"called with unknown variable \""+variable+"\"\n");
    }
  }
  else
    throw std::runtime_error(prefix+"not implemented yet\n");

}

void Self_energy::get_data(const std::string& variable, std::map<unsigned int, double>& data)
{
  // This get_data is used to retrieve the chemical_potential_map from the retarded BP self_energy
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::get_data ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+get_name()+"\")::get_data() ";
  //const Domain* temp_domain=get_const_simulation_domain();

  if (variable=="chemical_potential_map")
  {
    if(bp_chemical_potential_initialized)
    {
      data = chemical_potential_map;
    }
    else
      throw runtime_error(prefix+"chemical_potential_map has not been setup yet, check retarded BP self-energy\n");
  }
  else
    Propagation::get_data(variable,data);

  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::get_data(const std::string& variable, std::map<unsigned short, unsigned short>& data)
{
  // This get_data is used to retrieve the dof_to_bp_map from the retarded BP self_energy
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::get_data ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy(\""+get_name()+"\")::get_data() ";
  //const Domain* temp_domain=get_const_simulation_domain();

  if (variable=="dof_to_bp_map")
  {
    if(bp_chemical_potential_initialized)
    {
      data = dof_to_bp_map;
    }
    else
      throw runtime_error(prefix+"dof_to_bp_map has not been setup yet, check retarded BP self-energy\n");
  }
  else
    throw runtime_error(prefix+"unknown variable " + variable + "\n");

  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_setup_chemical_potential()
{
  // Specify Buttiker probes using a region with higher priority. Loop over all active atoms and check to see which ones belong
  // to the higher priority region. These are the atoms corresponding to Buettiker probes. Add new option "scattering region" that
  // specifies which region the Buettiker probes belong to.
  // The location will be specified by region, later can add an option for sparsity

  // Under development, Kyle Aitken: kaitken17@gmail.com
  // bp_atoms_ids: Vector to hold all IDs of atoms with Buttiker probes
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::bp_setup_chemical_potential ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy("+this->get_name()+")::bp_setup_chemical_potential: ";

  // Output - for test purposes
  // Look at Propagation::print_Propagator (translate momentum vector)
  std::string filename= "bp_txt_setup_chem_pot.txt";

  //std::ofstream out_file;
  //out_file.open(output_collector.get_file_path(filename,"File for debugging, contains information of DOF of atoms that are matched to a Buettiker probe" ,
                //NemoFileSystem::DEBUG).c_str());

  const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
  const AtomicStructure& atoms   = domain->get_atoms();

  ConstActiveAtomIterator it  = atoms.active_atoms_begin();
  ConstActiveAtomIterator end = atoms.active_atoms_end();

  const DOFmapInterface* defining_DOFmap=&(Hamilton_Constructor->get_dof_map(get_const_simulation_domain()));
  //HamiltonConstructor* tb_ham = 0;

  //-----------------------------------------------------------------------------------------------------
  // 1. Finds material(s) that correspond to Buettiker probes
  std::vector<std::string> bp_material_tags;

  if(options.check_option("bp_material_tags"))
  {
    options.get_option("bp_material_tags", bp_material_tags);
  }
  else
    throw std::invalid_argument(prefix+"please define \"bp_material_tags\"\n");

  //out_file << "Number of BP Materials:\t" << bp_material_tags.size() << "\n";

  unsigned short temp_bp_counter = 0; // Counter used for multi-DOF BPs

  // 2. Go through material(s) corresponding to Buettiker probes and finds DOF positions of probes in that material
  std::vector<std::string>::const_iterator mat_it = bp_material_tags.begin();
  for(unsigned int i=0; i<bp_material_tags.size(); ++i)
  {
    // Resets atom iterator
    it  = atoms.active_atoms_begin();

    //out_file << "Material Number:\t" << i << "\n";
    // Gets the options from the corresponding material
    const Material* mat = Material::find_material_by_tag(bp_material_tags[i]);
    const InputOptions* mat_options = mat->get_material_options();

    // bp_probe_model: model of probes (e.g. constant eta, exponential, etc.)
    std::string bp_probe_model = std::string("");
    if(mat_options->check_option("bp_probe_model"))
    {
      bp_probe_model = mat_options->get_option("bp_probe_model",std::string(""));
    }
    else
      throw std::invalid_argument(prefix+"please define \"bp_probe_model\" in "+mat->get_name()+"\n");

    unsigned short n_orbital;
    // Loops over all active atoms and their respective DOF maps to find locations for Buettiker probes corresponding to material
    for (; it!=end; ++it)
    {
      const AtomStructNode& nd = it.node();
      const Material* material = nd.atom->get_material(); // Gets material type of atom on corresponding node of the iterator
      const std::string tag = material->get_tag(); // Tag of material
      const unsigned int atom_id = it.id(); // Atom ID

      if(tag==bp_material_tags[i]) // If this atom's material matches that of BPs
      {
        //out_file << "Material matched\n";
        // Gets the atom's dof map
        const map<short, unsigned int>* atom_dofs = defining_DOFmap->get_atom_dof_map(atom_id);
        // Error if dof maps is not found
        if (atom_dofs == NULL)
          throw std::runtime_error(prefix + "atom DOFs are missing\n");

        //Sets number of orbitals equal to the size of atom_dofs
        n_orbital = atom_dofs->size();

        //-----------------------------------------------------------------------------------------------------
        // 3. Sets up dof_to_bp map and chemical_potential_map based off of model
        map<short, unsigned int>::const_iterator it_dofs; // Iterator to go through atom DOF

        if(bp_probe_model == "constant_eta" ||
            bp_probe_model == "constant_eta_band_edge" ||
            bp_probe_model == "contact" ||
            bp_probe_model == "energy_eta" ||
            bp_probe_model == "impurity_scattering" ||
            bp_probe_model == "grain_boundary" ||
            bp_probe_model == "p_p_boundary_impurity" ||
            bp_probe_model == "electron_phonon_coupling") // One BP to each DOF models) // One BP to each DOF models
        {
          for (unsigned short j=0; j < n_orbital; j++)
          {
            it_dofs = atom_dofs->find(j);
            assert( it_dofs != atom_dofs->end());

            // Inserts  corresponding atom_id and AtomStructNode into BP_asn_map
            //: If there was some way to get the material type from the DOF map position this map wouldn't be needed.
            //Bp_asn_map->insert(std::pair<unsigned int, const AtomStructNode*>(it_dofs->second, &nd));
            Bp_asn_map[it_dofs->second] = &nd;

            // One-to-one correspondance between DOF position and BP index
            dof_to_bp_map[it_dofs->second] = it_dofs->second;
            //out_file << "DOF:\t" << it_dofs->second << "\tto BP:\t" << it_dofs->second << "\n";
            // Creates entry in chemical potential map
            chemical_potential_map[it_dofs->second] = 0;
          }
        }
        else if(bp_probe_model == "constant_eta_atoms" ||
            bp_probe_model == "energy_eta_atoms" ||
            bp_probe_model == "impurity_scattering_atoms" ||
            bp_probe_model == "grain_boundary_atoms"||
            bp_probe_model == "p_p_boundary_impurity_atoms" ||
            bp_probe_model == "electron_phonon_coupling_atoms") // One BP to each atom
        {
          // Creates one BP for the entire atom
          chemical_potential_map[temp_bp_counter] = 0;
          for (unsigned short j=0; j < n_orbital; j++) // Loops over orbitals
          {
            it_dofs = atom_dofs->find(j);
            assert( it_dofs != atom_dofs->end());

            // Inserts  corresponding atom_id and AtomStructNode into BP_asn_map
            //Bp_asn_map->insert(std::pair<unsigned int, const AtomStructNode*>(it_dofs->second, &nd));
            Bp_asn_map[it_dofs->second] = &nd;

            // Alls DOFs of this atom assigned toward same BP
            dof_to_bp_map[it_dofs->second] = temp_bp_counter;
            //out_file << "DOF:\t" << it_dofs->second << "\tto BP:\t" << temp_bp_counter << "\n";
          }
          // Iterates to next BP index
          temp_bp_counter++;
        }
        else
        {
          throw std::runtime_error(prefix+"called with unknown Buettiker Probe type:\""+ bp_probe_model + "\" in " + mat->get_name() + "\n");
        }
      }
      else
      {
        //out_file << "Material not matched\n";
      }
    }
  }

  NEMO_ASSERT(dof_to_bp_map.size()>0,"Self_energy(\""+this->get_name()+"\"::bp_resolve_chemical_potential: No Buettiker_probes specified within region\n");

  //-----------------------------------------------------------------------------------------------------
  // 4. For each Buettiker probe material type initializes chemical potential values based off of model
  for(unsigned short i=0; i<bp_material_tags.size(); ++i)
  {
    // Gets the options from the corresponding material
    const Material* mat = Material::find_material_by_tag(bp_material_tags[i]);
    const InputOptions* mat_options = mat->get_material_options();

    // Get the initial chemical potential model
    std::string chem_pot_model = std::string("");
    if(mat_options->check_option("bp_initial_chemical_potential_model"))
    {
      chem_pot_model = mat_options->get_option("bp_initial_chemical_potential_model", std::string("zeros"));
    }
    else
      throw std::invalid_argument(prefix+"please define \"bp_initial_chemical_potential_model\"\n");

    // Sets the intial chemical potentials based off of model
    if(chem_pot_model == "1D_linear")
    {
      // Reads in linear values
      std::vector<double> linear_cp_corners;
      if(mat_options->check_option("bp_chem_potential_linear"))
      {
        mat_options->get_option("bp_chem_potential_linear", linear_cp_corners);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_chem_potential_linear\"\n");

      //NOTE: Change this to automatically find these values
      double temp_source_location;
      if(mat_options->check_option("source_location"))
      {
        temp_source_location = mat_options->get_option("source_location", 0.0);
      }
      else
        throw std::invalid_argument(prefix+"please define \"source_location\"\n");
      double temp_drain_location;
      if(mat_options->check_option("drain_location"))
      {
        temp_drain_location = mat_options->get_option("drain_location", 0.0);
      }
      else
        throw std::invalid_argument(prefix+"please define \"drain_location\"\n");

      std::map<unsigned short, unsigned short>::iterator it2 = dof_to_bp_map.begin();
      for(; it2!=dof_to_bp_map.end(); ++it2)
      {
        // Finds BP of particular DOF and sees if chemical potential is zero, if sets chemical potential (skips chemical potentials who have already been set)
        if(chemical_potential_map.find(it2->second)->second==0)
        {
          //NOTE: For now, assumes transport to be along x direction
          double new_cp = 0;
          // Calcualtes new chemical potential based off of atom location
          //const AtomStructNode* nd = Bp_asn_map->find(it2->first)->second; //Gets node corresponding to DOF
          const AtomStructNode* nd = Bp_asn_map.find(it2->first)->second; //Gets node corresponding to DOF
          //std::vector<double> temp_vector (nd->position, nd->position + sizeof nd->position / sizeof nd->position[0]); // converts array to vector
          double relative_location = 1 - ((nd->position[0] - temp_source_location) / (temp_drain_location - temp_source_location));
          new_cp = relative_location * (linear_cp_corners[0] - linear_cp_corners[1]) + linear_cp_corners[1];

          chemical_potential_map.find(it2->second)->second = new_cp;

          // 1. Find intended transmission direction (options)?
          // 2. Get corresponding coordinate from two contacts
          // 3. For each Buettiker probe, find distance perpendicular to location
          // 4. Create a linear chemical potential drop given those distances
        }
      }
    }
    else if (chem_pot_model == "direct_input")
    {
      // This method of initializing the chemical potential values directly reads in a vector of values equal to the number of Buettiker probes
      // and assigns the values to the chemical potentials of the Buettiker probes in order of increasing index (mainly used for debugging).
      // Reads in direct input values
      std::vector<double> direct_cp_values;
      if(mat_options->check_option("bp_chem_potential_direct"))
      {
        mat_options->get_option("bp_chem_potential_direct", direct_cp_values);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_chem_potential_direct\"\n");
      NEMO_ASSERT(chemical_potential_map.size()==direct_cp_values.size(),
                  prefix+"number of direct input chemical potential values must be equal to the number of Buettiker probes!\n");

      std::map<unsigned int, double>::iterator it2 = chemical_potential_map.begin();
      for(unsigned int j=0; it2!=chemical_potential_map.end(); ++it2, ++j)
      {
        // Assigns direct cp value to corresponding chemical potential
        it2->second = direct_cp_values[j];
      }
    }
    else if (chem_pot_model == "zeros")
    {
      // Do nothing
    }
    else
      throw std::invalid_argument(prefix+"do not recognize initial chemical potential model \""+chem_pot_model+"\"\n");
  }

  // 6. Final output and closes output file
  //out_file<<"BP Index\tChemical potential\n";
  std::map<unsigned int, double>::const_iterator bp_out_it = chemical_potential_map.begin();
  //for(; bp_out_it!=chemical_potential_map.end(); ++bp_out_it)
  //{
  //  out_file<<bp_out_it->first<<"\t"<<bp_out_it->second<<"\n";
  //  //if (Bp_asn_map->find(bp_out_it->first)!=Bp_asn_map->end())
  //  if (Bp_asn_map.find(bp_out_it->first)!=Bp_asn_map.end())
  //  {
  //    out_file<<"Corresponding ASN found\n";
  //  }
  //  else
  //  {
  //    out_file<<"Corresponding ASN NOt found\n";
  //  }
  //}
  //out_file.close();

  bp_chemical_potential_initialized = true;
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_Buettiker_retarded(Propagator*& output_Propagator,const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  // Under development, Kyle Aitken: kaitken17@gmail.com
  // output_Propagator: Propagator output of the propagation
  //    propagator_map: map of momenta and energy values to corresponding matrices, this is where each bp self-energy is stored
  //    allocated_momentum_Propagator_map: map of momenta and energy values
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_Buettiker_retarded ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy("+this->get_name()+")::do_solve_Buettiker_retarded: ";

  // Output - for test purposes
  // Look at Propagation::print_Propagator (translate momentum vector)
  //std::string filename= "bp_txt_do_solve_ret.txt";

  //std::ofstream out_file;
  //out_file.open(output_collector.get_file_path(filename,"File for testing" ,NemoFileSystem::DEBUG).c_str());
  const DOFmapInterface* defining_DOFmap=&(Hamilton_Constructor->get_dof_map(get_const_simulation_domain()));

  // 1. Checks to see if this is the constructor
  std::map<std::string,Simulation*>::const_iterator c_it=pointer_to_Propagator_Constructors->find(output_Propagator->get_name());
  NEMO_ASSERT(c_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found constructor of \""+output_Propagator->get_name()+"\"\n");
  NEMO_ASSERT(c_it->second==this,prefix+"this is not the constructor of \""+output_Propagator->get_name()+"\"\n");
  // 1.1 Checks for RGF override
  bool rgf_override =(options.get_option("rgf_override", false));

  // 2. Checks if Buettiker probes need to be setup first. This will iterate through all active atoms and define the chemical potential map
  // Also sets initial guess and creates a map between each Buettiker probe index and the AtomStructNode
  if(!rgf_override)
  {
    //out_file<<"No RGF Override\n";
    // : Clarify how this will work in parallel
    if(!chemical_potential_initialized())
    {
      //out_file<<"Chemical Potential not yet initialized\n";
      // Retrieves the atom ids of those in the Buettiker probe region and sets up their corresponding chemical potential, and
      bp_setup_chemical_potential();
    }
    else
    {
      //out_file<<"Chemical Potential already initialized\n";
    }

    NEMO_ASSERT(Bp_asn_map.size()>0,prefix+"Buettiker to Atomic Structure Node map is empty!\n");
  }
  //out_file.close();

  // 3. Read in input options
  // Material(s) that correspond to Buettiker probes
  std::vector<std::string> bp_material_tags;
  options.get_option("bp_material_tags", bp_material_tags);

  // Creates iterator pointing to matrix pointer corresponding to momentum_point
  if(!is_Propagator_initialized(output_Propagator->get_name()))
    initialize_Propagators(output_Propagator->get_name());
  //Propagator::PropagatorMap::iterator it = output_Propagator->propagator_map.find(momentum_point);
  //NEMO_ASSERT(it!=output_Propagator->propagator_map.end(),prefix+"have not found momentum\n");
  if(!is_Propagator_initialized(output_Propagator->get_name()))
    initialize_Propagators(output_Propagator->get_name());
  //Propagator::PropagatorMap::iterator it = output_Propagator->propagator_map.find(momentum_point);
  //NEMO_ASSERT(it!=output_Propagator->propagator_map.end(),prefix+"have not found momentum\n");

  // 4. Creates matrix, allocates memory, and sets sparsity pattern
  // NOTE: Alternative to: allocate_propagator_matrices(defining_DOFmap,&(it->first)) - (storage type needs to be set to diagonal);
  vector<int> local_rows;
  unsigned int number_of_rows = defining_DOFmap->get_global_dof_number();
  unsigned int local_number_of_rows =  defining_DOFmap->get_number_of_dofs();
  defining_DOFmap->get_local_row_indexes(&local_rows);

  msg << "Self_energy(\""+this->get_name()+"\"): problem size (DOFs): " << number_of_rows << std::endl;
  NEMO_ASSERT(number_of_rows>0, "Propagation(\""+this->get_name()+"\": seem to have 0 DOFs (empty matrix). There is something wrong. Aborting.");

  //std::map<std::string, Propagator*>::const_iterator it2=writeable_Propagators.begin();
  //for(; it2!=writeable_Propagators.end(); ++it2)
  if(writeable_Propagator!=NULL)
  {
    // Looks for "Buettiker_probe" in writable propagators
    if(name_of_writeable_Propagator.find(std::string("Buettiker_probe"))!=std::string::npos)
    {
      //only the respective constructor of each propagator is allowed to set the storage format
      std::map<std::string, Simulation*>::iterator temp_it=pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator);
      NEMO_ASSERT(temp_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found the constructor of \""+name_of_writeable_Propagator+"\"\n");
      if(temp_it->second==this)
      {
        // Creates matrix and set number of owned rows
        //it->second = new PetscMatrixParallelComplex(number_of_rows,number_of_rows, get_simulation_domain()->get_communicator());
        result = new PetscMatrixParallelComplex(number_of_rows,number_of_rows, get_simulation_domain()->get_communicator());
        result->set_num_owned_rows(local_number_of_rows);

        // Automatically sets to one local nonzero in each local row (diagonal)
        for (unsigned int i = 0; i < local_number_of_rows; i++)
          result->set_num_nonzeros(local_rows[i],1,0);

        //// Override due to rise reallocation error (makes the matrix dense)
        //for (unsigned int i = 0; i < local_number_of_rows; i++)
        //  it->second->set_num_nonzeros(local_rows[i],number_of_rows,0);

        msg << "Propagation(\""+this->get_name()+"\"): allocating memory\n";
        result->allocate_memory();

        //std::map<const std::vector<NemoMeshPoint>,bool>::iterator alloc_it=it2->second->allocated_momentum_Propagator_map.find(it->first);
        //NEMO_ASSERT(alloc_it!=it2->second->allocated_momentum_Propagator_map.end(),
        //            "Propagation(\""+it2->first+"\")::allocate_propagator_matrices have not found momentum in allocated_momentum_Propagator_map\n");
        //alloc_it->second=true;
        writeable_Propagator->allocated_momentum_Propagator_map[momentum_point]=true;

        result->set_to_zero();
      }
    }
  }

  // 5. Sets values of Retarded matrix based on bp_probe_model

  // Copies pointer to matrix
  //PetscMatrixParallelComplex* temp_bp_matrix = it->second;
  // For each Buettiker probe material type
  for(unsigned int i=0; i<bp_material_tags.size(); ++i)
  {
    // Gets the options from the corresponding material
    const Material* mat = Material::find_material_by_tag(bp_material_tags[i]);
    const InputOptions* mat_options = mat->get_material_options();
    if (mat == NULL) 
                  throw invalid_argument("[Self_energy::do_solve_Buettiker_retarded] wrong value for bp_material_tags option\n");


    // bp_probe_model: model of probes (e.g. constant eta, exponential, etc.)
    std::string bp_probe_model = std::string("");

    if(mat_options->check_option("bp_probe_model"))
    {
      bp_probe_model = mat_options->get_option("bp_probe_model",std::string(""));
    }
    else
      throw std::invalid_argument(prefix+"please define \"bp_probe_model\" in "+mat->get_name()+"\n");

    // Assigns corresponding index based upon type of Buettiker probe model
    if(bp_probe_model == "constant_eta" || bp_probe_model == "constant_eta_atoms") // Constant coupling, energy independent
    {
      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");

      std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0],temp_coupling[1]);
      // Iterates through all DOFs and assigns value at correct position corresponding to the index
      // if it has the material type corresponding to this material
    // this is not working at all
      std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        NEMO_ASSERT(Bp_asn_map.find(dof_it->first)!=Bp_asn_map.end(), "Propagation(\""+this->get_name()+"\": DOF not found. There is something wrong. Aborting.");
        // Gets the material of the probe at the given BP index
        const Material* dof_mat = Bp_asn_map.find(dof_it->first)->second->atom->get_material();
        if(dof_mat->get_tag() == bp_material_tags[i])
        {
          result->set(dof_it->first, dof_it->first, bp_coupling);
        }
      }
    //try this
      //for(unsigned int j=0; j<local_number_of_rows; j++)
      //  result->set(local_rows[j], local_rows[j], bp_coupling);
    }
    else if(bp_probe_model == "energy_eta" || bp_probe_model == "energy_eta_atoms") // energy dependent coupling
    {
      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");

      std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0],temp_coupling[1]);
      std::complex<double> temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point, output_Propagator);
      bp_coupling *= temp_energy*temp_energy*temp_energy;
      // Iterates through all DOFs and assigns value at correct position corresponding to the index
      // if it has the material type corresponding to this material
      std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        NEMO_ASSERT(Bp_asn_map.find(dof_it->first)!=Bp_asn_map.end(), "Propagation(\""+this->get_name()+"\": DOF not found. There is something wrong. Aborting.");
        // Gets the material of the probe at the given BP index
        const Material* dof_mat = Bp_asn_map.find(dof_it->first)->second->atom->get_material();
        if(dof_mat->get_tag() == bp_material_tags[i])
        {
          result->set(dof_it->first, dof_it->first, bp_coupling);
        }
      }
    }
    else if(bp_probe_model == "electron_phonon_coupling" || bp_probe_model == "electron_phonon_coupling_atoms") // energy dependent coupling
    {
      const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
      const AtomicStructure& atoms   = domain->get_atoms();
      ConstActiveAtomIterator it  = atoms.active_atoms_begin();
      unsigned short n_orbital;
      //const AtomStructNode& nd = it.node();
      const unsigned int atom_id = it.id(); // Atom ID
      const map<short, unsigned int>* atom_dofs = defining_DOFmap->get_atom_dof_map(atom_id);
      n_orbital = atom_dofs->size();

      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");
      std::vector<double> lattice_temperature;
      if(mat_options->check_option("lattice_temperature"))
      {
        mat_options->get_option("lattice_temperature", lattice_temperature);
      }
      else
        throw std::invalid_argument(prefix+"please define \"lattice_temperature\" in "+mat->get_name()+"\n");
      std::vector<double> band_edge_bp;
      if(mat_options->check_option("band_edge_bp"))
      {
        mat_options->get_option("band_edge_bp", band_edge_bp);
      }
      else
        throw std::invalid_argument(prefix+"please define \"band_edge_bp\" in "+mat->get_name()+"\n");
      std::complex<double> band_edge_temp = std::complex<double> (band_edge_bp[0],band_edge_bp[1]);
      //std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0],temp_coupling[1]);
      //std::complex<double> temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point, output_Propagator);
      //bp_coupling *= temp_energy*temp_energy*temp_energy;
      // Iterates through all DOFs and assigns value at correct position corresponding to the index
      // if it has the material type corresponding to this material
      std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        NEMO_ASSERT(Bp_asn_map.find(dof_it->first)!=Bp_asn_map.end(), "Propagation(\""+this->get_name()+"\": DOF not found. There is something wrong. Aborting.");
        // Gets the material of the probe at the given BP index
        const Material* dof_mat = Bp_asn_map.find(dof_it->first)->second->atom->get_material();

        if(dof_mat->get_tag() == bp_material_tags[i])
        {
          int tt = 0;
          if (bp_probe_model == "electron_phonon_coupling")
            tt = dof_it->first;
		  else 
            tt = dof_it->first/n_orbital;
          double temp_in_eV = lattice_temperature[tt]*NemoPhys::kB_nemo;
          double boson_distribution = NemoMath::bose_distribution(0.0, temp_in_eV,0.019);
          std::complex<double> temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point, output_Propagator);
          double temp_bp_coupling = temp_coupling[1]*boson_distribution;
          std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0],temp_bp_coupling);
          if (real(temp_energy)>real(band_edge_temp))
            bp_coupling *= std::sqrt(temp_energy-band_edge_temp);
					else 
            bp_coupling =0.00000000001;
          result->set(dof_it->first, dof_it->first, bp_coupling);
        }
      }
    }
    else if(bp_probe_model == "p_p_boundary_impurity" || bp_probe_model == "p_p_boundary_impurity_atoms") // energy dependent coupling
    {
      std::vector<double> temp_coupling_p_p;
      if(mat_options->check_option("bp_coupling_p_p"))
      {
        mat_options->get_option("bp_coupling_p_p", temp_coupling_p_p);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling_p_p\" in "+mat->get_name()+"\n");

      std::complex<double> bp_coupling = std::complex<double> (temp_coupling_p_p[0],temp_coupling_p_p[1]);
      std::complex<double> temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point, output_Propagator);
      bp_coupling *= temp_energy*temp_energy*temp_energy;
      std::vector<double> temp_coupling_boundary;
      if(mat_options->check_option("bp_coupling_boundary"))
      {
        mat_options->get_option("bp_coupling_boundary", temp_coupling_boundary);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling_boundary\" in "+mat->get_name()+"\n");

      std::complex<double> bp_coupling_boundary = std::complex<double> (temp_coupling_boundary[0],temp_coupling_boundary[1]);
      bp_coupling += bp_coupling_boundary*temp_energy;
      std::vector<double> temp_coupling_impurity;
      if(mat_options->check_option("bp_coupling_impurity"))
      {
        mat_options->get_option("bp_coupling_impurity", temp_coupling_impurity);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling_impurity\" in "+mat->get_name()+"\n");

      std::complex<double> bp_coupling_impurity = std::complex<double> (temp_coupling_impurity[0],temp_coupling_impurity[1]);
      bp_coupling += bp_coupling_impurity*temp_energy*temp_energy*temp_energy*temp_energy*temp_energy;
      // Iterates through all DOFs and assigns value at correct position corresponding to the index
      // if it has the material type corresponding to this material
      std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        NEMO_ASSERT(Bp_asn_map.find(dof_it->first)!=Bp_asn_map.end(), "Propagation(\""+this->get_name()+"\": DOF not found. There is something wrong. Aborting.");
        // Gets the material of the probe at the given BP index
        const Material* dof_mat = Bp_asn_map.find(dof_it->first)->second->atom->get_material();
        if(dof_mat->get_tag() == bp_material_tags[i])
        {
          result->set(dof_it->first, dof_it->first, bp_coupling);
        }
      }
    }
    else if(bp_probe_model == "impurity_scattering") // energy dependent coupling
    {
      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");

      std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0],temp_coupling[1]);
      std::complex<double> temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point, output_Propagator);
      bp_coupling *= temp_energy*temp_energy*temp_energy*temp_energy*temp_energy;
      // Iterates through all DOFs and assigns value at correct position corresponding to the index
      // if it has the material type corresponding to this material
    // this is not working at all
      std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        NEMO_ASSERT(Bp_asn_map.find(dof_it->first)!=Bp_asn_map.end(), "Propagation(\""+this->get_name()+"\": DOF not found. There is something wrong. Aborting.");
        // Gets the material of the probe at the given BP index
        const Material* dof_mat = Bp_asn_map.find(dof_it->first)->second->atom->get_material();
        if(dof_mat->get_tag() == bp_material_tags[i])
        {
          result->set(dof_it->first, dof_it->first, bp_coupling);
        }
      }
    }
    else if(bp_probe_model == "grain_boundary") // energy dependent coupling
    {
      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");

      std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0],temp_coupling[1]);
      std::complex<double> temp_energy = PropagationUtilities::read_complex_energy_from_momentum(this,momentum_point, output_Propagator);
      bp_coupling *= temp_energy;
      // Iterates through all DOFs and assigns value at correct position corresponding to the index
      // if it has the material type corresponding to this material
    // this is not working at all
      std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        NEMO_ASSERT(Bp_asn_map.find(dof_it->first)!=Bp_asn_map.end(), "Propagation(\""+this->get_name()+"\": DOF not found. There is something wrong. Aborting.");
        // Gets the material of the probe at the given BP index
        const Material* dof_mat = Bp_asn_map.find(dof_it->first)->second->atom->get_material();
        if(dof_mat->get_tag() == bp_material_tags[i])
        {
          result->set(dof_it->first, dof_it->first, bp_coupling);
        }
      }
    }
    else if(bp_probe_model == "constant_eta_band_edge") // Constant coupling, cuts off under band edge
    {
      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling")) //scattering strength eta
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");
      // Gets band edge
      double temp_edge;
      if(mat_options->check_option("Bands:BandEdge:Ec"))
      {
        temp_edge = mat_options->get_option("Bands:BandEdge:Ec", 0.0);
      }
      else
        throw std::invalid_argument(prefix+"please define \"Bands:BandEdge:Ec\" in "+mat->get_name()+"\n");
      // Reads in energy
      double temp_energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, output_Propagator);

      std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0],temp_coupling[1]);
      // Iterates through all Buettiker probes and assigns value at correct position corresponding to the index
      // if it has the material type corresponding to this material
      std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        // Gets the material of the probe at the given BP index
        NEMO_ASSERT(Bp_asn_map.find(dof_it->first)!=Bp_asn_map.end(), "Propagation(\""+this->get_name()+"\": DOF not found. There is something wrong. Aborting.");
        const Material* dof_mat = Bp_asn_map.find(dof_it->first)->second->atom->get_material();
        if(dof_mat->get_tag() == bp_material_tags[i])
        {
          // Checks to see if energy is above band edge
          if (temp_energy >= temp_edge)
          {
            result->set(dof_it->first, dof_it->first, bp_coupling);
          }
        }
      }
    }
    else if(bp_probe_model == "contact")
    {
      // NOT YET FINISHED
      // Reads in energy and momentum if necessary
      //double temp_energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, output_Propagator);
      std::vector<double> temp_momentum = read_kvector_from_momentum(momentum_point, output_Propagator);
    }
    else if(bp_probe_model == "rgf") // Override for rgf, creates a constant eta at all local rows which cutsoff when inside the band
    {
      std::vector<double> eta;
      if(mat_options->check_option("rgf_eta"))
      {
        mat_options->get_option("rgf_eta", eta);
      }
      else
        throw std::invalid_argument(prefix+"please define \"rgf_eta\" in "+mat->get_name()+"\n");
      // Gets the rgf band
      std::vector<double> rgf_band;
      if(mat_options->check_option("rgf_band"))
      {
        mat_options->get_option("rgf_band", rgf_band);
      }
      else
        rgf_band = std::vector<double>(2,-1000); // Default low enough to not matter

      // Reads in energy
      double temp_energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, output_Propagator);

      std::complex<double> bp_coupling = std::complex<double> (eta[0], eta[1]);
      // Iterates through all local rows and assigns eta value at correct position corresponding to the index
      // if it has the material type corresponding to this material
      for(unsigned int j=0; j<local_number_of_rows; j++)
      {
        // Checks to see if energy is NOT between the conduction and valence band edges
        if(!(temp_energy < rgf_band[1] && temp_energy > rgf_band[0]))
        {
          //temp_bp_matrix->set(local_rows[j], local_rows[j], bp_coupling);
          result->set(local_rows[j], local_rows[j], bp_coupling);
        }
      }
    }
    else if(bp_probe_model=="eta_band_edge")
    {
      //this is the model for a band edge dependent eta
      //purpose: set eta to 0 for energies and position below the band edge
      //5.1. get the energy of this (E,k) point
      double temp_energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, output_Propagator);
      //5.2. get the potential solver
      Simulation* potential_solver=NULL;
      //find the potential solver from HamiltonConstructor (if given in its options)
      InputOptions opt=Hamilton_Constructor->get_options();
      if(opt.check_option("potential_solver"))
      {
        std::string temp_name=opt.get_option("potential_solver",std::string(""));
        potential_solver=find_simulation(temp_name);
        NEMO_ASSERT(potential_solver!=NULL,prefix+"have not found simulation \""+temp_name+"\"\n");
      }
      //5.2.1 get the bandedge solver that gives the confined bandedge
      Simulation* band_solver = NULL;
      double Ec_confined=1e3;
      double Ev_confined=-1e3;
      if(mat_options->check_option("confined_band_solver"))
      {
        std::string temp_name=mat_options->get_option("confined_band_solver",std::string(""));
        band_solver=find_simulation(temp_name);
        NEMO_ASSERT(band_solver!=NULL,prefix+"have not found simulation \""+temp_name+"\"\n");
        std::string band;
        if(band_solver->get_options().check_option("sort_like_electrons"))
          band = band_solver->get_options().get_option("sort_like_electrons",true)?"conduction":"valence";
        else
        {
          string error_msg = prefix+" option sort_like_electrons not found in " +
                             band_solver->get_name() + "\n";
          throw invalid_argument(error_msg);
        }
        //read in Ek data
        std::vector<double> E_k_data;
        band_solver->get_data("E_k_data", E_k_data); //size of Nk*NE, format E1_k1,E2_k1,...Em_k1,E1_k2,E2_k2...Em_kn
        //read in kspace
        NemoMesh* kspace = NULL;
        band_solver->get_data("k_space", kspace);
        std::vector<NemoMeshPoint*> k_points = kspace->get_mesh_points();//size of Nk, k1,k2,...kn
        unsigned int NE=E_k_data.size()/k_points.size();
        //read the transport k-point (transverse) from momentum points
        std::vector<double> temp_vector=read_kvector_from_momentum(momentum_point, output_Propagator);
        //find the transport direction
        std::string transport_direction=options.get_option("transport_direction",std::string("x"));
        //looking for the transport k-point (transverse) from the kspace
        double distance=1e3;
        unsigned int index=0;
        std::vector<unsigned int > k_index;
        if(transport_direction.find(std::string("x"))!=std::string::npos)//transport along x-direction
        {
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            double kd=(temp_vector[1]-k_points[i]->get_y())*(temp_vector[1]-k_points[i]->get_y())+
                      (temp_vector[2]-k_points[i]->get_z())*(temp_vector[2]-k_points[i]->get_z());
            if(distance>kd)
            {
              distance=kd;
              index=i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same ky,kz
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            if(std::abs(k_points[i]->get_y()-k_points[index]->get_y())<1e-10 && std::abs(k_points[i]->get_z()-k_points[index]->get_z())<1e-10)
              k_index.push_back(i);
          }
        }
        else if(transport_direction.find(std::string("y"))!=std::string::npos)//transport along y-direction
        {
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            double kd=(temp_vector[0]-k_points[i]->get_x())*(temp_vector[0]-k_points[i]->get_x())+
                      (temp_vector[2]-k_points[i]->get_z())*(temp_vector[2]-k_points[i]->get_z());
            if(distance>kd)
            {
              distance=kd;
              index=i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same kx,kz
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            if(std::abs(k_points[i]->get_x()-k_points[index]->get_x())<1e-10 && std::abs(k_points[i]->get_z()-k_points[index]->get_z())<1e-10)
              k_index.push_back(i);
          }
        }
        else if(transport_direction.find(std::string("z"))!=std::string::npos)//transport along z-direction
        {
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            double kd=(temp_vector[0]-k_points[i]->get_x())*(temp_vector[0]-k_points[i]->get_x())+
                      (temp_vector[1]-k_points[i]->get_y())*(temp_vector[1]-k_points[i]->get_y());
            if(distance>kd)
            {
              distance=kd;
              index=i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same kx,ky
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            if(std::abs(k_points[i]->get_x()-k_points[index]->get_x())<1e-10 && std::abs(k_points[i]->get_y()-k_points[index]->get_y())<1e-10)
              k_index.push_back(i);
          }
        }
        else
          throw std::invalid_argument(prefix+" transport direction in "+transport_direction+" not supported yet\n");

        //find the required indices from Ek data, format E1_k1,E2_k1,...Em_k1,E1_k2,E2_k2...Em_kn
        std::vector<double> E_k_filter;
        for(unsigned int ii = 0; ii < k_index.size(); ii++)
        {
          unsigned int start_idx = k_index[ii] * NE;
          for(unsigned int jj = 0; jj < NE; jj++)
          {
            E_k_filter.push_back(E_k_data[start_idx+jj]);
          }
        }

        //find bandedge
        for(unsigned int counter = 0; counter < E_k_filter.size(); counter++)
          if(band.compare("conduction") == 0) //conduction bands
          {
            if(Ec_confined > E_k_filter[counter]) //find the lowest conduction band
              Ec_confined = E_k_filter[counter];
          }
          else //valence bands
          {
            if(Ev_confined < E_k_filter[counter]) //find the highest valence band
              Ev_confined = E_k_filter[counter];
          }
      }
      //5.3. get the constant eta value
      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");
      std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0], temp_coupling[1]);

      //5.4. fill the self-energy matrix
      const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
      DOFmapInterface&                dof_map = Hamilton_Constructor->get_dof_map(get_const_simulation_domain()); //get_dof_map();

      const AtomicStructure& atoms   = domain->get_atoms();
      ConstActiveAtomIterator ait  = atoms.active_atoms_begin();
      ConstActiveAtomIterator end = atoms.active_atoms_end();
      for ( ; ait != end; ++ait)
      {
        //5.4.1. loop over all active atoms in the current domain
        //5.4.2. determine the potential for the current atom (using get_data to the potential_solver, if potential_solver!=NULL)
        const unsigned int    atom_id   = ait.id();
        const AtomStructNode& nd        = ait.node();

        double potential_for_this_atom=0.0;
        if(potential_solver!=NULL)
        {
          vector<unsigned int> atom_ids(1, atom_id);
          vector<double> potential_vector;
          potential_solver->get_data("electron_potential_energy", atom_ids, potential_vector);
          potential_for_this_atom=potential_vector[0];
        }
        //5.4.3. get the Ec for this current atom (using the HamiltonConstructor)
        HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (Hamilton_Constructor->get_material_properties(nd.atom->get_material()));
        if(tb_ham==NULL)
        {
          create_material_properties_for_material(nd.atom->get_material());
          tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(nd.atom->get_material()));
        }
        double Ec_edge=0.0;
        double Ev_edge=0.0;
        if(mat_options->check_option("confined_band_solver"))
        {
          Ec_edge=Ec_confined; //if confined band solver defined, take the confined band edge
          Ev_edge=Ev_confined;
        }
        else
        {
          Ec_edge=tb_ham->get_conduction_band_edge(); //if confined band solver not defined, take bulk value from database
          Ev_edge=tb_ham->get_valence_band_edge();
        }
        //5.4.3. compare the energy with the band edge at the current atom - electron type: bigger gives eta, smaller gives 0
        std::complex<double> effective_eta(0.0,0.0);
        if(temp_energy>Ec_edge+potential_for_this_atom || temp_energy<Ev_edge+potential_for_this_atom)
          effective_eta=bp_coupling;
        //5.4.4 fill the self-energy for this atom
        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
        map<short, unsigned int>::const_iterator c_dof_it=atom_dofs->begin();
        //5.4.4.1 loop over all atomDofs of this atom
        for(; c_dof_it!=atom_dofs->end(); ++c_dof_it)
        {
          //5.4.4.2 find the appropriate matrix row/column index for the current atomDOF
          unsigned int index=c_dof_it->second;
          //5.4.4.3 set the self-energy matrix value of this atomDOF
          result->set(index,index,effective_eta);
        }
      }

    }
    else if(bp_probe_model == "eta_band_edge_exponential")
    {
      //purpose: position-energy dependent i*eta, exponentially decaying into the bandgap
      //Required options: 1. "confined_band_solver" comes from this eta solver option (not material option)
      //                  2. "transport_direction" comes from this eta solver option (not material option)
      //                  3. "bp_coupling" comes from material option, specifies the eta value above the bandedge
      //                  4. "decay_length" comes from material option, specifies how fast eta decay below the bandedge
      //                  5. "potential_solver" comes from Hamilton_Constructor option
      //                  6. "sort_like_electrons" comes from confined_band_solver, sepcifies whether solver for CB or VB
      //5.1. get the energy of this (E,k) point
      double temp_energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, output_Propagator);
      //5.2. get the potential solver
      Simulation* potential_solver = NULL;
      //find the potential solver from HamiltonConstructor (if given in its options)
      InputOptions opt = Hamilton_Constructor->get_options();
      if(opt.check_option("potential_solver"))
      {
        std::string temp_name = opt.get_option("potential_solver", std::string(""));
        potential_solver = find_simulation(temp_name);
        NEMO_ASSERT(potential_solver != NULL, prefix + "have not found simulation \"" + temp_name + "\"\n");
      }
      //5.2.1 get the bandedge solver that gives the confined bandedge
      Simulation* band_solver = NULL;
      double Ec_confined = 1e3;
      double Ev_confined = -1e3;

      if(options.check_option("confined_band_data_provider"))
      {
        std::string temp_name = options.get_option("confined_band_data_provider", std::string(""));
        band_solver = find_simulation(temp_name);
        NEMO_ASSERT(band_solver != NULL, prefix + "have not found simulation \"" + temp_name + "\"\n");
        std::vector<double> k_vector = read_kvector_from_momentum(momentum_point, output_Propagator);
        double band_edge = 0.0;
        bool CB = options.get_option("CB",true);
        band_solver->get_data(std::string("confined_band_edge"), this, k_vector, band_edge, CB);
        if(CB)
          Ec_confined = band_edge;
        else
          Ev_confined = band_edge;
      }
      else if(options.check_option("confined_band_solver"))
      {
        std::string temp_name = options.get_option("confined_band_solver", std::string(""));
        band_solver = find_simulation(temp_name);
        NEMO_ASSERT(band_solver != NULL, prefix + "have not found simulation \"" + temp_name + "\"\n");
        std::string band;
        if(band_solver->get_options().check_option("sort_like_electrons"))
          band = band_solver->get_options().get_option("sort_like_electrons", true) ? "conduction" : "valence";
        else
        {
          string error_msg = prefix + " option sort_like_electrons not found in " +
            band_solver->get_name() + "\n";
          throw invalid_argument(error_msg);
        }
        //read in Ek data
        std::vector<double> E_k_data;
        band_solver->get_data("E_k_data", E_k_data); //size of Nk*NE, format E1_k1,E2_k1,...Em_k1,E1_k2,E2_k2...Em_kn
        //read in kspace
        NemoMesh* kspace = NULL;
        band_solver->get_data("k_space", kspace);
        std::vector<NemoMeshPoint*> k_points = kspace->get_mesh_points();//size of Nk, k1,k2,...kn
        unsigned int NE = E_k_data.size() / k_points.size();
        //read the transport k-point (transverse) from momentum points
        std::vector<double> temp_vector = read_kvector_from_momentum(momentum_point, output_Propagator);
        //find the transport direction
        std::string transport_direction = options.get_option("transport_direction", std::string("x"));
        //looking for the transport k-point (transverse) from the kspace
        double distance = 1e3;
        unsigned int index = 0;
        std::vector<unsigned int > k_index;
        if(transport_direction.find(std::string("x")) != std::string::npos)//transport along x-direction
        {
          for(unsigned int i = 0; i<k_points.size(); i++)
          {
            double kd = (temp_vector[1] - k_points[i]->get_y())*(temp_vector[1] - k_points[i]->get_y()) +
              (temp_vector[2] - k_points[i]->get_z())*(temp_vector[2] - k_points[i]->get_z());
            if(distance>kd)
            {
              distance = kd;
              index = i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same ky,kz
          for(unsigned int i = 0; i<k_points.size(); i++)
          {
            if(std::abs(k_points[i]->get_y()-k_points[index]->get_y())<1e-10 && std::abs(k_points[i]->get_z()-k_points[index]->get_z())<1e-10)
              k_index.push_back(i);
          }
        }
        else if(transport_direction.find(std::string("y")) != std::string::npos)//transport along y-direction
        {
          for(unsigned int i = 0; i<k_points.size(); i++)
          {
            double kd = (temp_vector[0] - k_points[i]->get_x())*(temp_vector[0] - k_points[i]->get_x()) +
              (temp_vector[2] - k_points[i]->get_z())*(temp_vector[2] - k_points[i]->get_z());
            if(distance>kd)
            {
              distance = kd;
              index = i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same kx,kz
          for(unsigned int i = 0; i<k_points.size(); i++)
          {
            if(std::abs(k_points[i]->get_x()-k_points[index]->get_x())<1e-10 && std::abs(k_points[i]->get_z()-k_points[index]->get_z())<1e-10)
              k_index.push_back(i);
          }
        }
        else if(transport_direction.find(std::string("z")) != std::string::npos)//transport along z-direction
        {
          for(unsigned int i = 0; i<k_points.size(); i++)
          {
            double kd = (temp_vector[0] - k_points[i]->get_x())*(temp_vector[0] - k_points[i]->get_x()) +
              (temp_vector[1] - k_points[i]->get_y())*(temp_vector[1] - k_points[i]->get_y());
            if(distance>kd)
            {
              distance = kd;
              index = i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same kx,ky
          for(unsigned int i = 0; i<k_points.size(); i++)
          {
            if(std::abs(k_points[i]->get_x()-k_points[index]->get_x())<1e-10 && std::abs(k_points[i]->get_y()-k_points[index]->get_y())<1e-10)
              k_index.push_back(i);
          }
        }
        else
          throw std::invalid_argument(prefix + " transport direction in " + transport_direction + " not supported yet\n");

        //find the required indices from Ek data, format E1_k1,E2_k1,...Em_k1,E1_k2,E2_k2...Em_kn
        std::vector<double> E_k_filter;
        for(unsigned int ii = 0; ii < k_index.size(); ii++)
        {
          unsigned int start_idx = k_index[ii] * NE;
          for(unsigned int jj = 0; jj < NE; jj++)
          {
            E_k_filter.push_back(E_k_data[start_idx+jj]);
          }
        }

        //find bandedge
        for(unsigned int counter = 0; counter < E_k_filter.size(); counter++)
        if(band.compare("conduction") == 0) //conduction bands
        {
          if(Ec_confined > E_k_filter[counter]) //find the lowest conduction band
            Ec_confined = E_k_filter[counter];
        }
        else //valence bands
        {
          if(Ev_confined < E_k_filter[counter]) //find the highest valence band
            Ev_confined = E_k_filter[counter];
        }
      }

      //5.3.1. get the constant eta value
      std::vector<double> bp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", bp_coupling);
      }
      else
        throw std::invalid_argument(prefix + "please define \"bp_coupling\" in " + mat->get_name() + "\n");
      //5.3.2. get the decay length
      double decay_length = 0.0;
      if(mat_options->check_option("decay_length"))
      {
        decay_length = mat_options->get_option("decay_length", 0.0);
        decay_length = decay_length + 1e-20; //avoid divide by zero
      }
      else
        throw std::invalid_argument(prefix + "please define \"decay_length\" in " + mat->get_name() + "\n");

      //5.4. fill the self-energy matrix
      const AtomisticDomain* domain = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
      DOFmapInterface&                dof_map = Hamilton_Constructor->get_dof_map(get_const_simulation_domain()); //get_dof_map();

      const AtomicStructure& atoms = domain->get_atoms();
      ConstActiveAtomIterator ait = atoms.active_atoms_begin();
      ConstActiveAtomIterator end = atoms.active_atoms_end();
      for(; ait != end; ++ait)
      {
        //5.4.1. loop over all active atoms in the current domain
        //5.4.2. determine the potential for the current atom (using get_data to the potential_solver, if potential_solver!=NULL)
        const unsigned int    atom_id = ait.id();
        const AtomStructNode& nd = ait.node();

        double potential_for_this_atom = 0.0;
        if(!options.get_option("potential_included_in_band_data",false) && potential_solver != NULL)
        {
          vector<unsigned int> atom_ids(1, atom_id);
          vector<double> potential_vector;
          potential_solver->get_data("electron_potential_energy", atom_ids, potential_vector);
          potential_for_this_atom = potential_vector[0];
        }
        //5.4.3. get the Ec for this current atom (using the HamiltonConstructor)
        HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (Hamilton_Constructor->get_material_properties(nd.atom->get_material()));
        if(tb_ham == NULL)
        {
          create_material_properties_for_material(nd.atom->get_material());
          tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(nd.atom->get_material()));
        }
        double Ec_edge = 0.0;
        double Ev_edge = 0.0;
        if(options.check_option("confined_band_solver") || options.check_option("confined_band_data_provider"))
        {
          Ec_edge = Ec_confined;
          Ev_edge = Ev_confined;
        }
        else
        {
          Ec_edge = tb_ham->get_conduction_band_edge();
          Ev_edge = tb_ham->get_valence_band_edge();
        }
        //5.4.3. compare the energy with the band edge at the current atom - electron type: bigger gives eta, smaller gives 0
        std::complex<double> effective_eta(0.0, 0.0);
        bool CB = options.get_option("CB",true);
        if(CB)
        {
          if(temp_energy > Ec_edge + potential_for_this_atom)
            effective_eta = std::complex<double>(bp_coupling[0], bp_coupling[1]);
          else
          {
            double distance = Ec_edge + potential_for_this_atom - temp_energy;
            effective_eta = std::complex<double>(bp_coupling[0] * std::exp(-distance/decay_length), bp_coupling[1] * std::exp(-distance/decay_length));
          }
        }
        else
        {
          if(temp_energy < Ev_edge + potential_for_this_atom)
            effective_eta = std::complex<double>(bp_coupling[0], bp_coupling[1]);
          else
          {
            double distance = temp_energy - Ev_edge - potential_for_this_atom;
            effective_eta = std::complex<double>(bp_coupling[0] * std::exp(-distance/decay_length), bp_coupling[1] * std::exp(-distance/decay_length));
          }      
        }

/*
        if(temp_energy > Ec_edge + potential_for_this_atom || temp_energy < Ev_edge + potential_for_this_atom)
        {
          effective_eta = std::complex<double>(bp_coupling[0], bp_coupling[1]);;
        }
        else
        {
          double distance = std::min(Ec_edge + potential_for_this_atom - temp_energy, temp_energy - Ev_edge - potential_for_this_atom);
          effective_eta = std::complex<double>(bp_coupling[0] * std::exp(-distance/decay_length), bp_coupling[1] * std::exp(-distance/decay_length));
        }
*/
        //5.4.4 fill the self-energy for this atom
        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
        map<short, unsigned int>::const_iterator c_dof_it = atom_dofs->begin();
        //5.4.4.1 loop over all atomDofs of this atom
        for(; c_dof_it != atom_dofs->end(); ++c_dof_it)
        {
          //5.4.4.2 find the appropriate matrix row/column index for the current atomDOF
          unsigned int index = c_dof_it->second;
          //5.4.4.3 set the self-energy matrix value of this atomDOF
          result->set(index, index, effective_eta);
        }
      }

    }
    // SS: This is added to make eta E,k and space dependent, in case equilibrium region consists of a heterostructure (in EqNeq method)
    // This method is same as "eta_band_edge" method except that it adds delta_Ec(or delta_Ev) to the reference energy, where
    // delta_Ec = Ec1 (conduction band edge from "confined band solver) - Ec2 (conduction band edge for the material at atom in position,x)
    // and reference energy is the energy above(conduction band) or below (valence band) which eta is nonzero.
    else if(bp_probe_model=="eta_space_energy")
    {
      //this is the model for a band edge dependent eta
      //purpose: set eta to 0 for energies and position below the band edge
      //5.1. get the energy of this (E,k) point
      double temp_energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point, output_Propagator);
      //5.2. get the potential solver
      Simulation* potential_solver=NULL;
      //find the potential solver from HamiltonConstructor (if given in its options)
      InputOptions opt=Hamilton_Constructor->get_options();
      if(opt.check_option("potential_solver"))
      {
        std::string temp_name=opt.get_option("potential_solver",std::string(""));
        potential_solver=find_simulation(temp_name);
        NEMO_ASSERT(potential_solver!=NULL,prefix+"have not found simulation \""+temp_name+"\"\n");
      }
      //5.2.1 get the bandedge solver that gives the confined bandedge
      Simulation* band_solver = NULL;
      double Ec_confined=1e3;
      double Ev_confined=-1e3;
      if(mat_options->check_option("confined_band_solver"))
      {
        std::string temp_name=mat_options->get_option("confined_band_solver",std::string(""));
        band_solver=find_simulation(temp_name);
        NEMO_ASSERT(band_solver!=NULL,prefix+"have not found simulation \""+temp_name+"\"\n");
        std::string band;
        if(band_solver->get_options().check_option("sort_like_electrons"))
          band = band_solver->get_options().get_option("sort_like_electrons",true)?"conduction":"valence";
        else
        {
          string error_msg = prefix+" option sort_like_electrons not found in " +
                             band_solver->get_name() + "\n";
          throw invalid_argument(error_msg);
        }
        //read in Ek data
        std::vector<double> E_k_data;
        band_solver->get_data("E_k_data", E_k_data); //size of Nk*NE, format E1_k1,E2_k1,...Em_k1,E1_k2,E2_k2...Em_kn
        //read in kspace
        NemoMesh* kspace = NULL;
        band_solver->get_data("k_space", kspace);
        std::vector<NemoMeshPoint*> k_points = kspace->get_mesh_points();//size of Nk, k1,k2,...kn
        unsigned int NE=E_k_data.size()/k_points.size();
        //read the transport k-point (transverse) from momentum points
        std::vector<double> temp_vector=read_kvector_from_momentum(momentum_point, output_Propagator);
        //find the transport direction
        std::string transport_direction=options.get_option("transport_direction",std::string("x"));
        //looking for the transport k-point (transverse) from the kspace
        double distance=1e3;
        unsigned int index=0;
        std::vector<unsigned int > k_index;
        if(transport_direction.find(std::string("x"))!=std::string::npos)//transport along x-direction
        {
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            double kd=(temp_vector[1]-k_points[i]->get_y())*(temp_vector[1]-k_points[i]->get_y())+
                      (temp_vector[2]-k_points[i]->get_z())*(temp_vector[2]-k_points[i]->get_z());
            if(distance>kd)
            {
              distance=kd;
              index=i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same ky,kz
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            if((k_points[i]->get_y()==k_points[index]->get_y()) && (k_points[i]->get_z()==k_points[index]->get_z()))
              k_index.push_back(i);
          }
        }
        else if(transport_direction.find(std::string("y"))!=std::string::npos)//transport along y-direction
        {
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            double kd=(temp_vector[1]-k_points[i]->get_x())*(temp_vector[1]-k_points[i]->get_x())+
                      (temp_vector[2]-k_points[i]->get_z())*(temp_vector[2]-k_points[i]->get_z());
            if(distance>kd)
            {
              distance=kd;
              index=i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same ky,kz
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            if((k_points[i]->get_x()==k_points[index]->get_x()) && (k_points[i]->get_z()==k_points[index]->get_z()))
              k_index.push_back(i);
          }
        }
        else if(transport_direction.find(std::string("z"))!=std::string::npos)//transport along z-direction
        {
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            double kd=(temp_vector[1]-k_points[i]->get_x())*(temp_vector[1]-k_points[i]->get_x())+
                      (temp_vector[2]-k_points[i]->get_y())*(temp_vector[2]-k_points[i]->get_y());
            if(distance>kd)
            {
              distance=kd;
              index=i;//find the index of kspace that is closest to the transport k-point
            }
          }
          //after the index is found, find all kpoints in kspace which have the same ky,kz
          for(unsigned int i=0; i<k_points.size(); i++)
          {
            if((k_points[i]->get_x()==k_points[index]->get_x()) && (k_points[i]->get_y()==k_points[index]->get_y()))
              k_index.push_back(i);
          }
        }
        else
          throw std::invalid_argument(prefix+" transport direction in "+transport_direction+" not supported yet\n");

        //find the required indices from Ek data
        std::vector<double> E_k_filter;
        for(unsigned int i=0; i<NE; i++)
          for(unsigned int j=0; j<k_index.size(); j++)
            E_k_filter.push_back(E_k_data[k_index[j]+i*k_points.size()]);

        //find bandedge
        for(unsigned int counter = 0; counter < E_k_filter.size(); counter++)
          if(band.compare("conduction") == 0) //conduction bands
          {
            if(Ec_confined > E_k_filter[counter]) //find the lowest conduction band
              Ec_confined = E_k_filter[counter];
          }
          else //valence bands
          {
            if(Ev_confined < E_k_filter[counter]) //find the highest valence band
              Ev_confined = E_k_filter[counter];
          }
      }
      //5.3. get the constant eta value
      std::vector<double> temp_coupling;
      if(mat_options->check_option("bp_coupling"))
      {
        mat_options->get_option("bp_coupling", temp_coupling);
      }
      else
        throw std::invalid_argument(prefix+"please define \"bp_coupling\" in "+mat->get_name()+"\n");
      std::complex<double> bp_coupling = std::complex<double> (temp_coupling[0], temp_coupling[1]);

      //5.4. fill the self-energy matrix
      const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (get_simulation_domain());
      DOFmapInterface&                dof_map = Hamilton_Constructor->get_dof_map(get_const_simulation_domain()); //get_dof_map();

      const AtomicStructure& atoms   = domain->get_atoms();
      ConstActiveAtomIterator ait  = atoms.active_atoms_begin();
      ConstActiveAtomIterator end = atoms.active_atoms_end();
      for ( ; ait != end; ++ait)
      {
        //5.4.1. loop over all active atoms in the current domain
        //5.4.2. determine the potential for the current atom (using get_data to the potential_solver, if potential_solver!=NULL)
        const unsigned int    atom_id   = ait.id();
        const AtomStructNode& nd        = ait.node();

        double potential_for_this_atom=0.0;
        if(potential_solver!=NULL)
        {
          vector<unsigned int> atom_ids(1, atom_id);
          vector<double> potential_vector;
          potential_solver->get_data("electron_potential_energy", atom_ids, potential_vector);
          potential_for_this_atom=potential_vector[0];
        }
        //5.4.3. get the Ec for this current atom (using the HamiltonConstructor)
        HamiltonConstructor* tb_ham = dynamic_cast<HamiltonConstructor*> (Hamilton_Constructor->get_material_properties(nd.atom->get_material()));
        if(tb_ham==NULL)
        {
          create_material_properties_for_material(nd.atom->get_material());
          tb_ham = dynamic_cast<HamiltonConstructor*> (this->get_material_properties(nd.atom->get_material()));
        }
        double Ec_edge=0.0;
        double Ev_edge=0.0;
        if(mat_options->check_option("confined_band_solver"))
        {
          Ec_edge=Ec_confined;
          Ev_edge=Ev_confined;
        }
        else
        {
          Ec_edge=tb_ham->get_conduction_band_edge();
          Ev_edge=tb_ham->get_valence_band_edge();
        }
        //5.4.3. compare the energy with the band edge at the current atom - electron type: bigger gives eta, smaller gives 0
        std::complex<double> effective_eta(0.0,0.0);
        if(temp_energy>Ec_edge+potential_for_this_atom || temp_energy<Ev_edge+potential_for_this_atom)
          effective_eta=bp_coupling;
        //5.4.4 fill the self-energy for this atom
        const map<short, unsigned int>* atom_dofs = dof_map.get_atom_dof_map(atom_id);
        map<short, unsigned int>::const_iterator c_dof_it=atom_dofs->begin();
        //5.4.4.1 loop over all atomDofs of this atom
        for(; c_dof_it!=atom_dofs->end(); ++c_dof_it)
        {
          //5.4.4.2 find the appropriate matrix row/column index for the current atomDOF
          unsigned int index=c_dof_it->second;
          //5.4.4.3 set the self-energy matrix value of this atomDOF
          result->set(index,index,effective_eta);
        }
      }

    }
    else
    {
      throw std::runtime_error(prefix+"called with unknown Buettiker Probe type:\""+ bp_probe_model + "\" in " + mat->get_name() + "\n");
    }
  }

  // 6. Sets result
  result->assemble();
  // it->second->save_to_matlab_file(std::string("debug_bp_R_self_energy.m"));
  //result = it->second;
  NemoUtils::toc(tic_toc_prefix);

}

void Self_energy::do_solve_Buettiker_lesser(Propagator*& output_Propagator, const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  // Under development, Kyle Aitken: kaitken17@gmail.com
  tic_toc_name = options.get_option("tic_toc_name",get_name());
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::do_solve_Buettiker_lesser ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix = "Self_energy("+this->get_name()+")::do_solve_Buettiker_lesser: ";

  // const DOFmap* defining_DOFmap=&(Hamilton_Constructor->get_dof_map(get_const_simulation_domain()));
  //unsigned int number_of_rows = defining_DOFmap->get_global_dof_number();

  // Creates iterator pointing to matrix pointer corresponding to momentum_point
  Propagator::PropagatorMap::iterator it=output_Propagator->propagator_map.find(momentum_point);
  NEMO_ASSERT(it!=output_Propagator->propagator_map.end(),prefix+"have not found momentum\n");

  // Checks to see if this is the constructor
  std::map<std::string,Simulation*>::const_iterator c_it=pointer_to_Propagator_Constructors->find(output_Propagator->get_name());
  NEMO_ASSERT(c_it!=pointer_to_Propagator_Constructors->end(),prefix+"have not found constructor of \""+output_Propagator->get_name()+"\"\n");
  NEMO_ASSERT(c_it->second==this,prefix+"this is not the constructor of \""+output_Propagator->get_name()+"\"\n");

  // Allocates room for matrix corresponding to momentum_point
  // Automatically sets storage_type to diagonal
  //std::map<std::string, Propagator*>::const_iterator it2=writeable_Propagators.begin();
  //for(; it2!=writeable_Propagators.end(); ++it2)
  //if(writeable_Propagator!=NULL)
  //{
  //  // Looks for "Buettiker_probe" in writable propagators
  //  if(name_of_writeable_Propagator.find(std::string("Buettiker_probe"))!=std::string::npos)
  //  {
  //    // Only the respective constructor of each propagator is allowed to set the storage format
  //    NEMO_ASSERT(pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator)!=pointer_to_Propagator_Constructors->end(),prefix+
  //                "have not found constructor of \""+name_of_writeable_Propagator+"\"\n");
  //    if(pointer_to_Propagator_Constructors->find(name_of_writeable_Propagator)->second==this)
  //    {
  //      // Sets option in input deck and in PMPC for allocate_propagator_matrices
  //      options.set_option(name_of_writeable_Propagator+std::string("_store"),std::string("diagonal"));
  //      writeable_Propagator->set_storage_type(std::string("diagonal"));
  //    }
  //  }
  //}

  //allocate_propagator_matrices(defining_DOFmap,&(it->first));
  //it->second->assemble();

  //PetscMatrixParallelComplex* temp_bp_matrix = it->second;
  //temp_bp_matrix->set_to_zero();

  // Checks to see if chemical potentials are resolved
  // NOTE: Add function to set bp_chemical_potenetial_resolved to false on initialization and other case
  // (i.e. during Poisson iteration)
  if(!chemical_potential_resolved())
  {
    // If not, executes method to resolve chemical potential
    bp_resolve_chemical_potential();
  }

  //Then, executes as usual
  bp_do_solve_lesser_equilibrium(output_Propagator, momentum_point, result);

  // temp_bp_matrix->save_to_matlab_file(std::string("debug_bp_L_self_energy.m"));

  // Sets result
  //result = temp_bp_matrix;

  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_resolve_chemical_potential()
{
	// Under development, Kyle Aitken: kaitken17@gmail.com
	// This method will resolve the Buettiker probe chemical potential at each node. Currently, only energy relaxed model is being implemented

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::resolve_Buettiker_probes ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::resolve_Buettiker_probes: ";

	msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes] Resolving Chemical Potential");

	// Inputs from input deck
	//    - Current threshold: Threshold to be met before Newton Raphson finishes
	double bp_current_threshold = 1e-9;
	if (options.check_option("bp_current_threshold"))
	{
		bp_current_threshold = options.get_option("bp_current_threshold", 1e-15);
	}
	else
		throw std::invalid_argument(prefix + "please define \"bp_current_threshold\"\n");
	//double temperature=options.get_option("temperature", NemoPhys::temperature);
	double kB_temp_in_eV = temperature*NemoPhys::kB_nemo;

	bool analytical_momenta = false;
	// Container for analytical integration, the type of analytical momentum, i.e. 1D or 2D
	std::string momentum_type = options.get_option("bp_analytical_momenta", std::string(""));

	// Output - for test purposes
	std::string filename = "bp_txt_resolve_chem_pot.txt";

	//std::ofstream out_file;
	//out_file.open(output_collector.get_file_path(filename,"Buettiker probe resolve chemical potential method output" ,NemoFileSystem::DEBUG).c_str());

	////out_file.open(filename.c_str());
	//out_file << "BP Resolve Chemical Potential Status\n\n";


	// 1. Calculates the transmission. The transmission matrix is (number of nonvirtual contacts + number of Buettiker probes)^2 big,
	// the contacts make up the first indices, then the Buettiker probes are listed in order of increasing index

	// ----------------------------------------------------------------------------------------------------------------------------------
	// 1.1 Finds all retarded contact self-energies (partially from from Greensolver::do_solve_inverse). All nonvirtual stored to map with simulation
	// and name of the Buettiker self-energy stored as well.

	std::map<std::string, Simulation*> nonvirtual_contacts; // Map of names to sources of non_virtual contacts
	// NOTE: Since this only accepts one Buettiker probe at the moment, possibly change to a pair in the future
	std::map<std::string, Simulation*> bp_retarded_self_sources; // Map of names to sources of buettiker probes
	std::map<std::string, const Propagator*>::const_iterator c_it = Propagators.begin();

	for (; c_it != Propagators.end(); ++c_it)
	{
		// Omit those propagators that are in the writeable_Propagators map (!=lesser_Buettiker_probe).
		// If the current propagator is not found in the list of writable propagators
		if (name_of_writeable_Propagator != c_it->first)
		{
			// Omit those Propagators with "Buettiker_probe" in them
			if (c_it->first.find(std::string("Buettiker_probe")) == std::string::npos)
			{
				// Throws exception for Boson self-energies
				// NEMO_ASSERT(Propagator_types.find(c_it->first)->second!=NemoPhys::Boson_retarded_self, prefix + "not implemented for Bosons\n");
				// If the propagator is of type Fermion_retarded_self
				if (Propagator_types.find(c_it->first)->second == NemoPhys::Fermion_retarded_self ||
					Propagator_types.find(c_it->first)->second == NemoPhys::Boson_retarded_self)
				{
					// Inserts pointer to simulation keyed by name into nonvirtual_contacts
					// NOTE: Add NEMO_ASSERT here for no option found
					Simulation* source_of_self_energy = find_source_of_data(c_it->first);
					nonvirtual_contacts[c_it->first] = source_of_self_energy;
					//out_file << "NVC:\t" + c_it->first << "\n";
					msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes] Real Contact:\t" + c_it->first);
				}
			}
			else
			{
				// Finds source of Buettiker probe retarded self-energy
				Simulation* source_of_bp_self_energy = find_source_of_data(c_it->first);
				bp_retarded_self_sources[c_it->first] = source_of_bp_self_energy;
				//out_file << "BP:\t" + c_it->first << "\n";
			}
		}
	}

	NEMO_ASSERT(bp_retarded_self_sources.size()>0, prefix + "No Buettiker_probe self-energy found\n");
	NEMO_ASSERT(bp_retarded_self_sources.size() == 1, prefix + "Not yet implemented for multiple retarded Buettiker probe self-eneriges\n");
	NEMO_ASSERT(nonvirtual_contacts.size()>0, prefix + "No real contact self-energy found\n");

	// ----------------------------------------------------------------------------------------------------------------------------------
	// 1.2 Get the retarded Green's function of the device (either the only retarded Green's function within this propagation, or read in from inputdeck
	std::map<std::string, const Propagator*>::const_iterator G_iterator;

	if (options.check_option("G_for_transmission"))
	{
		std::string name_of_G = options.get_option("G_for_transmission", std::string(""));
		G_iterator = Propagators.find(name_of_G);
		NEMO_ASSERT(G_iterator != Propagators.end(), "Self_energy(\"" + get_name() + "\")::resolve_Buettiker_probes have not found Propagator \"" + name_of_G + "\n");
		NemoPhys::Propagator_type G_type = Propagator_types.find(name_of_G)->second;
		NEMO_ASSERT(G_type == NemoPhys::Fermion_retarded_Green ||
			G_type == NemoPhys::Boson_retarded_Green, "Self_energy(\"" + get_name() + "\")::resolve_Buettiker_probes Type of \"" + name_of_G
			+ " is \"" + Propagator_type_map.find(G_type)->second + "\"\n");
	}
	else
	{
		G_iterator = Propagators.end();
		std::map<std::string, const Propagator*>::const_iterator c_it = Propagators.begin();
		for (; c_it != Propagators.end(); ++c_it)
		{
			std::map<std::string, NemoPhys::Propagator_type>::const_iterator temp_cit = Propagator_types.find(c_it->first);
			NEMO_ASSERT(temp_cit != Propagator_types.end(), "Self_energy(\"" + get_name() + "\")::resolve_Buettiker_probes Type of \"" + c_it->first + " is not found\n");
			if (temp_cit->second == NemoPhys::Fermion_retarded_Green || temp_cit->second == NemoPhys::Boson_retarded_Green)
			{
				NEMO_ASSERT(G_iterator == Propagators.end(), "Self_energy(\"" + get_name() + "\")::resolve_Buettiker_probes found more than one retarded Green's function\n");
				G_iterator = c_it;
			}
		}
	}
	// Get the retarded Green's function from its constructor
	std::map<std::string, Simulation*>::const_iterator prop_cit = pointer_to_Propagator_Constructors->find(G_iterator->first);
	NEMO_ASSERT(prop_cit != pointer_to_Propagator_Constructors->end(), prefix + "have not found constructor of \"" + G_iterator->first + "\"\n");
	const Propagator* ret_Green = NULL;
	//out_file << "Ret Green:\t" + G_iterator->first << "\n";
	prop_cit->second->get_data(G_iterator->first, ret_Green);
	NEMO_ASSERT(ret_Green != NULL, prefix + "have not found the retarded Green's function\n");

	msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes] Retarded Green's Function:\t" + ret_Green->get_name());

	// ----------------------------------------------------------------------------------------------------------------------------------
	// 1.3 Retrieves dof_to_bp_map and chemical_potential_map from the retarded-self energy. Since the retarded Green's function is
	// dependent upon the retarded BP self-energy, the chemical potential map should be set up at this point
	std::map<std::string, Simulation*>::const_iterator bp_source_it = bp_retarded_self_sources.begin();
	PetscMatrixParallelComplex* bp_retarded_self;
	Propagator::PropagatorMap::const_iterator green_matrix_iterator = ret_Green->propagator_map.begin();
	bp_source_it->second->get_data(bp_source_it->first, &(green_matrix_iterator->first), bp_retarded_self);
	std::map<unsigned int, double>& chemical_potential_map_ref = chemical_potential_map;
	std::map<unsigned short, unsigned short>& dof_to_bp_map_ref = dof_to_bp_map;
	//std::map<unsigned int, double>& energy_exchange_from_electron_ref = energy_exchange_from_electron;
	//Propagators.find(bp_source_it->first)->second->get_data("chemical_potential_map", &chemical_potential_map);
	bp_source_it->second->get_data(std::string("chemical_potential_map"), chemical_potential_map_ref);
	bp_source_it->second->get_data(std::string("dof_to_bp_map"), dof_to_bp_map_ref);
	//bp_source_it->second->get_data(std::string("energy_exchange_from_electron"), energy_exchange_from_electron_ref);


	// ----------------------------------------------------------------------------------------------------------------------------------
	// 1.4 For each real contact, finds the index locations that correspond to Buettiker probe locations. At these indices the current is NOT
	// conserved because these indices correspond to contacts. Therefore they are removed from the chemical potential map for the time being
	// and added to the "constant_cp_bp" map.
	// CAUTION: Assumes each self-energy has nonzero values in the same locations for all momenta
	msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes] Building real contact to Buettiker probe map...");

	// This is a map between the name of a real contact and a set of unsigned ints. The unsigned ints correspond to the
	// (single) indices of the diagonal nonzero values.
	//std::map<std::string, std::set<unsigned int> > nvc_nonzero_diag_map;
	// This is a map of constant chemical potential Buettiker probes, they are not included in the Newton-Raphson because their chemical potentials
	// do not need to be resolved, instead the chemical potentials are set equal to their corresponding contact
	std::map<unsigned int, double> const_cp_bp;

	//: Could zero energies cause problems?
	std::map<std::string, Simulation*>::const_iterator c_it2 = nonvirtual_contacts.begin();
	for (; c_it2 != nonvirtual_contacts.end(); ++c_it2)
	{
		// Retrieves the self-energy matrix (assumes to be same size as the Green's function matrices)
		// PetscMatrixParallelComplex* self_matrix = new PetscMatrixParallelComplex (*(green_matrix_iterator->second));
		PetscMatrixParallelComplex* self_matrix = NULL; //new PetscMatrixParallelComplex (*bp_retarded_self);
		c_it2->second->get_data(c_it2->first, &(green_matrix_iterator->first), self_matrix, &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
		// self_matrix->save_to_matlab_file(std::string("self_matrix.m"));
		// Retrieves the chemical potential for the real contact

		// This part of the code is not used
		/*
		double nvc_chemical_potential = 0.0;
		if(options.check_option(c_it2->first+"_chemical_potential"))
		{
		nvc_chemical_potential = options.get_option(c_it2->first + "_chemical_potential", 0.0);
		}
		else
		throw std::invalid_argument(prefix+"please define \""+c_it2->first+"_chemical_potential\"\n");
		*/
		std::complex<double> zero = std::complex<double>(0.0, 0.0);
		std::set<unsigned int> temp_set;
		// Iterates over all DOFs corresponding to BPs
		std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
		for (; dof_it != dof_to_bp_map.end(); ++dof_it)
		{
			// If the corresponding self-energy location does not equal zero, goes through a series of steps:
			if (self_matrix->get(dof_it->first, dof_it->first) != zero)
			{
				// 1. Finds the BP the index belongs to
				//unsigned short temp_bp_index = dof_it->second;

				// 2. Checks to see if BP has already gone through steps of assigning it to constant chemical potential
				//if(const_cp_bp.find(temp_bp_index)==const_cp_bp.end()) // If not...
				//{
				//  std::stringstream index_to_string;
				//  index_to_string << temp_bp_index;

				//  msg.print_message(NemoUtils::MSG_LEVEL_1,"[Buettiker Probes]   Setting Buettiker Probe " + index_to_string.str() + " to chemical potential of " + c_it2->first);
				//  // 3. Adds all DOFs that belong to this BP to the set of indices that correspond to Buettiker probes for THIS nonvirtual contact
				//  std::map<unsigned short, unsigned short>::const_iterator dof_it2 = dof_to_bp_map.begin();
				//  for(; dof_it2!=dof_to_bp_map.end(); ++dof_it2) // Iterates through all BP DOFs to find those that belong to BP
				//  {
				//    if(dof_it2->second == temp_bp_index)
				//    {
				//      temp_set.insert(dof_it2->first);

				//      std::stringstream index_to_string2;
				//      index_to_string2 << dof_it2->first;

				//      msg.print_message(NemoUtils::MSG_LEVEL_1,"[Buettiker Probes]     Including DOF " + index_to_string2.str());
				//    }
				//  }
				// 4. Adds BP to the constant chemical potential Buettiker probe map
				//const_cp_bp[temp_bp_index] = nvc_chemical_potential;
				// 5. Removes BP from the chemical potential map
				//chemical_potential_map.erase(temp_bp_index);
				//}
			}
		}

		// Assigns temp_set to nvc_nonzero_diag_map
		//nvc_nonzero_diag_map[c_it2->first] = temp_set;
		//delete self_matrix;
	}

	// ----------------------------------------------------------------------------------------------------------------------------------
	// 2. Finds the transmission for each pair of contacts for each momenta tuple. Buettiker probe values and real contacts stored seperately
	// The size of bp_trans_map is defined by the DOFmap and the transmission between two Buettiker probes is stored at the corresponding
	// point between two indices.
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*> bp_trans_map;
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>* bp_trans_map_address = &bp_trans_map;

	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*> nvc_trans_map;
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>* nvc_trans_map_address = &nvc_trans_map;

	// Loop over all momenta tuples and create a pointer to a transmission matrix for each value, but only point to NULL for the time being
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator trans_it = ret_Green->propagator_map.begin();
	for (; trans_it != ret_Green->propagator_map.end(); ++trans_it)
	{
		bp_trans_map[trans_it->first] = NULL;
		nvc_trans_map[trans_it->first] = NULL;
	}

	msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes] Computing Transmissions...");
	if (options.get_option("bp_full_transmission_calculation", false))
		msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes]   Full Calculation");
	else
		msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes]   Using Single Nonzero Shortcut");
	// Calculates the transmission for ALL Buettiker probes for each local momenta tuple

	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*> contact_bp_trans_map;
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>* contact_bp_trans_map_address = &contact_bp_trans_map;

	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*> contact_nvc_trans_map;
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>* contact_nvc_trans_map_address = &contact_nvc_trans_map;

	// Loop over all momenta tuples and create a pointer to a transmission matrix for each value, but only point to NULL for the time being
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator trans_it2 = ret_Green->propagator_map.begin();
	for (; trans_it2 != ret_Green->propagator_map.end(); ++trans_it2)
	{
		contact_bp_trans_map[trans_it2->first] = NULL;
		contact_nvc_trans_map[trans_it2->first] = NULL;
	}

	bool not_calculate_contact_current = true;
	if (options.check_option("not_calculate_contact_current"))
	{
		not_calculate_contact_current = options.get_option("not_calculate_contact_current", true);
	}
	if (not_calculate_contact_current)
		bp_calculate_contact_transmission(ret_Green, nonvirtual_contacts, bp_retarded_self_sources, contact_bp_trans_map_address, contact_nvc_trans_map_address);

	bp_calculate_transmission(ret_Green, nonvirtual_contacts, bp_retarded_self_sources, bp_trans_map_address, nvc_trans_map_address);

	//DEBUG: Prints out all transmission map values for all energies
	//std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator trans_it2 = bp_trans_map.begin();
	//for(unsigned int i=0; trans_it2!=bp_trans_map.end(); ++trans_it2, ++i)
	//{
	//  ostringstream convert;
	//  convert << i;
	//  trans_it2->second->save_to_matlab_file(std::string("energy_bp_trans_map_" + convert.str() + ".m"));
	//  nvc_trans_map.find(trans_it2->first)->second->save_to_matlab_file(std::string("energy_nvc_trans_map_" + convert.str() + ".m"));
	//}

	// ----------------------------------------------------------------------------------------------------------------------------------
	// 3.1 Calculates the local current for each BP (based on model). Uses get_data to call local current density solver
	std::vector<std::complex<double> > local_current_density(chemical_potential_map.size(), 0.0);
	std::vector<std::complex<double> >* local_current_density_address = &local_current_density;
	std::vector<std::complex<double> > local_energy_current_density(chemical_potential_map.size(), 0.0);
	std::vector<std::complex<double> >* local_energy_current_density_address = &local_energy_current_density;

	bp_energy_relaxed_local_current(nonvirtual_contacts, ret_Green, bp_trans_map_address, nvc_trans_map_address, local_current_density_address, local_energy_current_density_address);

	// 3.2 Integrate local current across all processes
	std::vector<std::complex<double> > total_current_density(chemical_potential_map.size(), 0.0);

	MPI_Allreduce(&(local_current_density[0]), &(total_current_density[0]), chemical_potential_map.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, simulation_communicator);
	// Check with calculate_local_current
	// calculate_local_current();

	// ----------------------------------------------------------------------------------------------------------------------------------
	// 4. Checks if maximum of total current is below threshold
	//out_file << "Newton Raphson Method:\n";
	msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes] Newton-Raphson Method...");
	if (options.check_option("bp_analytical_momenta"))
	{
		analytical_momenta = true;
		if (momentum_type == std::string("1D"))
		{
			throw std::invalid_argument(prefix + "\"1D\" analytical momentum not yet implemented for Buettiker probes\n");
			msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes]   Analytical Momenta: 1D");
		}
		else if (momentum_type == std::string("2D")) // Uses fermi integral
		{
			msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes]   Analytical Momenta: 2D");
		}
		else
			throw std::invalid_argument(prefix + "the value of analytical_momenta can be either \"1D\" or \"2D\"\n");
	}
	// Variable to keep track of values of interest
	double total_current = 0;
	double max_bp_current = 0;
	unsigned int newton_raphson_count = 1;
	// Calculates maximum and total current
	for (unsigned int i = 0; i<chemical_potential_map.size(); ++i)
	{
		total_current += real(total_current_density[i]);
		if (abs(total_current_density[i])>max_bp_current)
			max_bp_current = abs(total_current_density[i]);
	}
	// Prints status
	ostringstream convert;
	convert << newton_raphson_count << "\tMax J:\t" << max_bp_current << "\tTotal J:\t" << total_current;
	msg.print_message(NemoUtils::MSG_LEVEL_1, "  " + convert.str());

	//out_file << newton_raphson_count << "\tMax J:\t" << max_bp_current << "\tTotal J:\t" << total_current << "\n";

	while (bp_current_threshold < max_bp_current)
	{
		// 5. Current is not below threshold, computes the Jacobian
		// J_pq = -(e/h) * int(dE, df_q/dmu_q * sum(trans(E))) for p=q
		//         (e/h) * int(dE, df_q/dmu_q * trans_qp(E)) for p!=q

		PetscMatrixParallelComplex jacobian = PetscMatrixParallelComplex(chemical_potential_map.size(), chemical_potential_map.size(),
			get_const_simulation_domain()->get_communicator());

		//std::vector<std::complex<double> > local_current_density_new(chemical_potential_map.size(), 0.0);
		//std::vector<std::complex<double> >* local_current_density_address_new = &local_current_density_new;
		//std::vector<std::complex<double> > local_energy_current_density_new(chemical_potential_map.size(), 0.0);
		//std::vector<std::complex<double> >* local_energy_current_density_address_new = &local_energy_current_density_new;

		//bp_energy_relaxed_local_current(nonvirtual_contacts, ret_Green, bp_trans_map_address, nvc_trans_map_address, local_current_density_address_new, local_energy_current_density_address_new);
		//MPI_Allreduce(&(local_current_density_new[0]), &(total_current_density[0]), chemical_potential_map.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, simulation_communicator);


		jacobian.set_num_owned_rows(chemical_potential_map.size());
		//// Sets the number of nonzeros to all elements in each row (workaround for consder_as_full)
		//for (unsigned int i = 0; i<chemical_potential_map.size(); ++i)
		//{
		//  jacobian.set_num_nonzeros(i, chemical_potential_map.size(), 0);
		//}
		jacobian.consider_as_full();
		jacobian.allocate_memory();
		jacobian.assemble();
		jacobian.set_to_zero();

		// Gets effective mass (for analytical momenta integration)
		//: See  about how effective mass works for TB?
		std::vector<double> effective_mass;
		std::vector<unsigned int> dummy_input(1, 1);
		if (particle_type_is_Fermion) Hamilton_Constructor->get_data("effective_mass", dummy_input, effective_mass);

		// Sum over all local momenta.
		std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator momenta_it = bp_trans_map.begin();
		for (unsigned int momentum_num = 0; momenta_it != bp_trans_map.end(); ++momenta_it, ++momentum_num)
		{
			//out_file << "\n\n-----------------------MOMENTA POINT (";
			//for(unsigned int i=0; i<momenta_it->first.size(); ++i) // Index for each momenta
			//{
			//  out_file << momenta_it->first[i].get_idx() << " ";
			//}
			//out_file << ")-----------------------\n";

			// Local mometa jacobian
			PetscMatrixParallelComplex momenta_jacobian = PetscMatrixParallelComplex(chemical_potential_map.size(), chemical_potential_map.size(),
				get_const_simulation_domain()->get_communicator());

			// Gets the energy for particular momenta
			double energy_in_eV = PropagationUtilities::read_energy_from_momentum(this, momenta_it->first, ret_Green);

			// Variables to determine process rows
			std::vector<int> local_rows(chemical_potential_map.size(), 0); // rows on process
			std::vector<int> local_nonzeros(chemical_potential_map.size(), 0); // number of local_nonzeros for corresponding row
			std::vector<int> nonlocal_nonzeros(chemical_potential_map.size(), 0); // number of nonlocal_nonzeros for corresponding row
			//DEBUG: Serial override for now
			for (unsigned int i = 0; i<chemical_potential_map.size(); ++i)
			{
				local_rows[i] = i;
				local_nonzeros[i] = chemical_potential_map.size();
				nonlocal_nonzeros[i] = 0;
			}

			// Initialize momenta jacobian
			momenta_jacobian.set_num_owned_rows(chemical_potential_map.size()); // DEBUG: Serial override
			momenta_jacobian.consider_as_full();
			momenta_jacobian.allocate_memory();
			momenta_jacobian.assemble();
			momenta_jacobian.set_to_zero();

			std::map<unsigned int, double>::const_iterator it1 = chemical_potential_map.begin();
			for (unsigned int i = 0; it1 != chemical_potential_map.end(); ++it1, ++i)
			{
				//out_file << "  Row:\t" << i << "\n";

				std::map<unsigned int, double>::const_iterator it2 = chemical_potential_map.begin();
				for (unsigned int j = 0; it2 != chemical_potential_map.end(); ++it2, ++j)
				{
					//out_file << "    Column:\t" << j << "\n";

					// Temporary complex variable to hold transmission values
					std::complex<double> trans(std::complex<double>(0.0, 0.0));

					if (it1->first == it2->first) // Diagonal case
					{
						//out_file << "     Diagonal Case:\n";
						// Sums all transmissions to the given buettiker probe from other probes
						std::map<unsigned int, double>::const_iterator it3 = chemical_potential_map.begin();
						std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator momenta_bp_trans = bp_trans_map.find(momenta_it->first);
						for (; it3 != chemical_potential_map.end(); ++it3)
						{
							// Exception for transmission of the probe to itself
							if (it3->first != it1->first)
							{
								// Note this is the sum of negative transmissions
								trans -= momenta_bp_trans->second->get(it1->first, it3->first);
								//out_file << "       Trans(" << it1->first << ", " << it3->first << "):\t" << momenta_bp_trans->second->get(it1->first, it3->first) << "\n";
							}
						}
						// Sums all transmissions to the given buettiker probe from real contacts
						std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator momenta_nvc_trans = nvc_trans_map.find(momenta_it->first);
						for (unsigned int k = 0; k<nonvirtual_contacts.size(); ++k)
						{
							trans -= momenta_nvc_trans->second->get(it1->first, k);
							//out_file << "       Trans(" << it1->first << ", Contact" << k << "):\t" << momenta_nvc_trans->second->get(it1->first, k) << "\n";
						}
						//out_file << "     Trans sum:\t" << trans << "\n";
					}
					else // Non-diagonal case
					{
						trans += bp_trans_map.find(momenta_it->first)->second->get(it2->first, it1->first);
						//out_file << "     Trans(" << it2->first << ", " << it1->first << "):\t" << trans << "\n";
					}

					// Multiplies by different distributions based upon momenta model
					if (options.check_option("bp_analytical_momenta")) // Analytical case
					{
						// Find the type of analytical momentum, i.e. 1D or 2D
						if (momentum_type == std::string("1D"))
						{
							throw std::invalid_argument(prefix + "\"1D\" analytical momentum not yet implemented for Buettiker probes\n");
						}
						else if (momentum_type == std::string("2D")) // Uses fermi integral
						{
							// Multiplies by appropriate effective mass
							trans *= effective_mass[it2->first] * NemoPhys::electron_mass;
							//out_file << "       Eff Mass:\t" << effective_mass[it2->first] << "\n";
							// Multiplies by dF_q/dmu_q
							double temp = (it2->second - energy_in_eV) / kB_temp_in_eV;
							double dFermi_dmu = std::exp(temp) / (1 + std::exp(temp));
							//out_file << "       dF/dmu:\t" << dFermi_dmu << "\n";
							trans *= dFermi_dmu;
						}
					}
					else // Nonanalytical case
					{
						// Multiplies by df_q/dmu_q
						double dfermi_dmu;
						//double dbose_dmu;

						if (particle_type_is_Fermion)
						{
							if (options.get_option("electron_temperature_driven", false))
							{
								double chemical_potential = 0.0;
								if (options.check_option("chemical_potential"))
								{
									chemical_potential = options.get_option("chemical_potential", 0.0);
								}
								else
									throw std::invalid_argument(prefix + "please define chemical_potential\"\n");
								double kB_temp_in_eV_bp = it2->second*NemoPhys::kB_nemo;
								dfermi_dmu = NemoMath::dfermi_distribution_over_dT(chemical_potential, kB_temp_in_eV_bp, energy_in_eV)*NemoPhys::kB_nemo;
							}
							else
								dfermi_dmu = NemoMath::dfermi_distribution_over_dmu(it2->second, kB_temp_in_eV, energy_in_eV);
						}
						else if (!particle_type_is_Fermion)
						{
							double kB_temp_in_eV_ph = it2->second*NemoPhys::kB_nemo;
							dfermi_dmu = energy_in_eV*NemoMath::dbose_distribution_over_dT(0.0, kB_temp_in_eV_ph, energy_in_eV)*NemoPhys::kB_nemo;
						}
						//out_file << "       df/dmu:\t" << dfermi_dmu << "\n";
						trans *= dfermi_dmu;
					}

					//out_file << "         Temp:\t" << kB_temp_in_eV << "\n";
					//out_file << "         Energy:\t" << energy_in_eV << "\n";
					//out_file << "         Mu:\t" << it2->second << "\n";

					// Saves value to momenta jacobian
					//NOTE: Saves real value for the time being
					momenta_jacobian.set(i, j, real(trans));
				}
			}
			// Reassembles Jacobian after set commands
			momenta_jacobian.assemble();

			// Multiplies mometa_jacobian by appropriate integration weight (accounts for analytical integration)

			double weight = bp_get_integration_weight(ret_Green, momenta_it->first, analytical_momenta, momentum_type);
			//out_file << "Weight:\t" << weight << "\nMomenta Jacobian:\n";

			// Prints the momenta jacobian to output file
			//for(unsigned int i=0; i<chemical_potential_map.size(); ++i)
			//{
			//  //out_file << i << ":\t";
			//  for(unsigned int j=0; j<chemical_potential_map.size(); ++j)
			//  {
			//    out_file << real(momenta_jacobian.get(i,j)) << "\t";
			//  }
			//  out_file << "\n";
			//}

			// Do the integral - i.e. weighted sum
			jacobian.add_matrix(momenta_jacobian, DIFFERENT_NONZERO_PATTERN, weight);

			// Assemble after addition
			jacobian.assemble();
			// Print out Jacobian so far
			//out_file << "Jacobian:\n";
			//for(unsigned int i=0; i<chemical_potential_map.size(); ++i)
			//{
			//  out_file << i << ":\t";
			//  for(unsigned int j=0; j<chemical_potential_map.size(); ++j)
			//  {
			//    out_file << real(jacobian.get(i,j)) << "\t";
			//  }
			//  out_file << "\n";
			//}

			//DEBUG: Prints out Jacobian thus far
			/*ostringstream convert;
			convert << momentum_num;
			jacobian.save_to_matlab_file(std::string("debug_jacobian_" + convert.str() + ".m"));*/
			//jacobian.save_to_matlab_file(std::string("debug_jacobian1.m"));
		}

		// Prints out final local Jacobian
		//out_file << "\nFinal Local Jacobian:\n";
		//for(unsigned int i=0; i<chemical_potential_map.size(); ++i)
		//{
		//  out_file << i << ":\t";
		//  for(unsigned int j=0; j<chemical_potential_map.size(); ++j)
		//  {
		//    out_file << real(jacobian.get(i,j)) << "\t";
		//  }
		//  out_file << "\n";
		//}

		jacobian.assemble();
		//jacobian.save_to_matlab_file(std::string("debug_jacobian.m"));

		// Multiplies entire jacobian by appropriate prefactor
		if (options.check_option("bp_analytical_momenta"))
		{
			if (momentum_type == std::string("1D"))
			{
				throw std::invalid_argument(prefix + "\"1D\" analytical momentum not yet implemented for Buettiker probes\n");
			}
			else if (momentum_type == std::string("2D")) // Uses fermi integral
			{
				jacobian *= std::complex<double>(8.0 * NemoMath::pi * NemoPhys::elementary_charge / NemoPhys::hbar_nemo / NemoPhys::hbar_nemo / NemoPhys::hbar_nemo, 0.0);
			}
		}
		else
		{
			// e/h (extra e for conversion of h to eV*s)
			jacobian *= std::complex<double>(NemoPhys::elementary_charge*NemoPhys::elementary_charge / NemoPhys::h, 0.0);
		}

		jacobian.assemble();

		// Retrieves the PETSc pointer to matrix to combine
		std::complex<double>* pointer_to_jacobian = NULL;
		jacobian.get_array_for_matrix(pointer_to_jacobian);

		MPI_Allreduce(MPI_IN_PLACE, pointer_to_jacobian, chemical_potential_map.size()*chemical_potential_map.size(), MPI_DOUBLE_COMPLEX, MPI_SUM,
			simulation_communicator);
		//DEBUG: Break of output file
		//if(newton_raphson_count==3)
		//    out_file.close();

		PetscMatrixParallelComplex* inv_jacobian = NULL;
		// Inverts the Jacobian
		/*
		//jacobian.save_to_matlab_file("jacobian.m");
		ofstream outfile;
		//std::string filename = "jacobian.dat";
		outfile.open(filename.c_str());
		outfile << "\njacobian\n ";
		std::map<unsigned int, double>::iterator it_jacobian = chemical_potential_map.begin();
		for(unsigned int i=0; i<chemical_potential_map.size(); ++i, ++it_jacobian)
		{
		outfile << i << ":\t";
		for(unsigned int j=0; j<chemical_potential_map.size(); ++j)
		{
		outfile << jacobian.get(i,j) << "\t";
		}
		outfile << "\n";
		}
		outfile.close();
		*/
		PetscMatrixParallelComplex::invert(jacobian, &inv_jacobian);
		NEMO_ASSERT(inv_jacobian != NULL, prefix + "inversion failed\n");

		// Prints out inverted Jacobian
		//out_file << "\nInverted Jacobian:\n";
		//for(unsigned int i=0; i<chemical_potential_map.size(); ++i)
		//{
		//  out_file << i << ":\t";
		//  for(unsigned int j=0; j<chemical_potential_map.size(); ++j)
		//  {
		//    out_file << real(inv_jacobian->get(i,j)) << "\t";
		//  }
		//  out_file << "\n";
		//}

		// 4. Uses Jacobian to find needed change in chemical potentials and adjusts chemical potentials and returns to step #1.
		// mu_p^(i+1) = mu_p^(i) - sum(all elements in row p of inv Jacobian, (J_pk^-1)^(i) * F_k^(i)
		std::map<unsigned int, double>::iterator it_test = chemical_potential_map.begin();
		//for(unsigned int i=0; i<chemical_potential_map.size(); ++i, ++it_test)
		//{
		//  out_file << "Local Current Densities\n";
		//  out_file << "Probe " << it_test->first << " LCD:\t" << total_current_density[i] << "\n";
		//}

		std::map<unsigned int, double>::iterator it = chemical_potential_map.begin();
		std::vector<double> energy_exchange_from_electron(chemical_potential_map.size(), 0.0);
		if (options.check_option("energy_exchange_from_electron"))
		{
			options.get_option("energy_exchange_from_electron", energy_exchange_from_electron);
		}
		//std::vector<std::complex<double> > energy_exchange_from_electron(chemical_potential_map.size(),0.0);
		//energy_exchange_from_electron = options.get_option("energy_exchange_from_electron", energy_exchange_from_electron);
		for (unsigned int i = 0; i<chemical_potential_map.size(); ++i, ++it)
		{
			double sum = 0;
			for (unsigned int j = 0; j<chemical_potential_map.size(); ++j)
			{
				sum += real(inv_jacobian->get(i, j) * (total_current_density[j] - energy_exchange_from_electron[j]));
			}
			// Adjusts chemical potential accordingly
			//out_file << "Probe " << it->first << " Mu:\t" << it->second << "\tdMu:\t" << sum << "\n";
			it->second -= sum;
		}

		delete inv_jacobian;

		// MPI Broadcast new mu's

		// Recalculates local current density for new chemical potentials
		// NOTE: Make sure total_current_density is zeroed
		bp_energy_relaxed_local_current(nonvirtual_contacts, ret_Green, bp_trans_map_address, nvc_trans_map_address, local_current_density_address, local_energy_current_density_address);

		// Integrate local current across all processes

		MPI_Allreduce(&(local_current_density[0]), &(total_current_density[0]), chemical_potential_map.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, simulation_communicator);
		// Calculates new maximum current
		++newton_raphson_count;
		max_bp_current = 0;
		total_current = 0;

		for (unsigned int i = 0; i<chemical_potential_map.size(); ++i)
		{
			total_current += real(total_current_density[i] - energy_exchange_from_electron[i]);
			if (abs(total_current_density[i])>max_bp_current)
				max_bp_current = abs(total_current_density[i] - energy_exchange_from_electron[i]);
		}

		// Prints status
		ostringstream convert;
		convert << newton_raphson_count << "\tMax J:\t" << max_bp_current << "\tTotal J:\t" << total_current;
		msg.print_message(NemoUtils::MSG_LEVEL_1, "  " + convert.str());
	}

	msg.print_message(NemoUtils::MSG_LEVEL_1, "[Buettiker Probes] Newton-Raphson Threshold Met");
	//out_file << "Newton Raphson Threshold Met\n";

	std::vector<std::complex<double> > contact_local_current_density(nonvirtual_contacts.size(), 0.0);
	std::vector<std::complex<double> >* contact_local_current_density_address = &contact_local_current_density;

	// 3.2 Integrate local current across all processes
	std::vector<std::complex<double> > contact_total_current_density(nonvirtual_contacts.size(), 0.0);
	if (not_calculate_contact_current)
	{
		bp_contact_energy_relaxed_local_current(nonvirtual_contacts, ret_Green, contact_bp_trans_map_address, contact_nvc_trans_map_address, contact_local_current_density_address);
		MPI_Allreduce(&(contact_local_current_density[0]), &(contact_total_current_density[0]), nonvirtual_contacts.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, simulation_communicator);
	}
	if (particle_type_is_Fermion && (options.get_option("electron_temperature_driven", false)))
	{
		std::vector<std::complex<double> > local_energy_current_density(chemical_potential_map.size(), 0.0);
		std::vector<std::complex<double> >* local_energy_current_density_address = &local_energy_current_density;
		bp_energy_relaxed_local_energy_current(nonvirtual_contacts, ret_Green, bp_trans_map_address, nvc_trans_map_address, local_energy_current_density_address);
		std::vector<std::complex<double> > total_energy_current_density(chemical_potential_map.size(), 0.0);
		MPI_Allreduce(&(local_energy_current_density[0]), &(total_energy_current_density[0]), chemical_potential_map.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, simulation_communicator);
		for (unsigned int i = 0; i<chemical_potential_map.size(); ++i)
		{
			OUT_NORMAL << "energy current" << total_energy_current_density[i] << "\n";
		}
	}
	else if (particle_type_is_Fermion && (options.get_option("electron_potential_driven", false)))
	{
		std::vector<std::complex<double> > total_energy_current_density(chemical_potential_map.size(), 0.0);
		MPI_Allreduce(&(local_energy_current_density[0]), &(total_energy_current_density[0]), chemical_potential_map.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, simulation_communicator);
		for (unsigned int i = 0; i<chemical_potential_map.size(); ++i)
		{
			OUT_NORMAL << "energy current" << total_energy_current_density[i] << "\n";
		}
	}

	for (unsigned int i = 0; i<chemical_potential_map.size(); ++i)
	{
		total_current += real(total_current_density[i]);
		if (abs(total_current_density[i])>max_bp_current)
			max_bp_current = abs(total_current_density[i]);
	}
	if (not_calculate_contact_current)
	{
		OUT_NORMAL << "\tSource J:\t" << contact_total_current_density[0] << "\n";
		OUT_NORMAL << "\tDrain J:\t" << contact_total_current_density[1] << "\n";
	}
	// Adds the constant chemical potential Buettiker probes back into the chemical potential map
	std::map<unsigned int, double>::const_iterator const_cp_it = const_cp_bp.begin();
	for (; const_cp_it != const_cp_bp.end(); const_cp_it++)
	{
		chemical_potential_map[const_cp_it->first] = const_cp_it->second;
	}

	// Prints mu's
	std::map<unsigned int, double>::const_iterator mu_it = chemical_potential_map.begin();
	for (; mu_it != chemical_potential_map.end(); ++mu_it)
	{
		OUT_NORMAL << "BP" << mu_it->first << ":\t" << mu_it->second << "\n";
	}

	// Deletes bp_trans_map and nvc_trans_map
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator momenta_it = bp_trans_map.begin();
	for (; momenta_it != bp_trans_map.end(); ++momenta_it)
	{
		delete nvc_trans_map.find(momenta_it->first)->second;
		delete momenta_it->second;
	}

	// Deletes bp_trans_map and nvc_trans_map
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator momenta_it1 = contact_nvc_trans_map.begin();
	for (; momenta_it1 != contact_nvc_trans_map.end(); ++momenta_it1)
	{
		delete contact_bp_trans_map.find(momenta_it1->first)->second;
		delete momenta_it1->second;
	}

	// At this point the chemical potentials are resolved given the user-defined threshold.
	set_chemical_potential_status(true);

	//out_file.close();
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_calculate_local_current_density(const std::vector<NemoMeshPoint>& momentum_point, std::map<std::string, Simulation*> nonvirtual_contacts,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map,
	std::vector<std::complex<double> >*& momenta_lcd, std::vector<std::complex<double> >*& momenta_lecd)
{

	// Calculates the current_divergence for ALL Buettiker probes for a particular momenta(for Buettiker probes)
	// j_p(momentum_point) = (e/h) * sum_(l = all contacts)(T_lp * f_l(E) - T_pl * f_p(E))
	// Under development, Kyle Aitken: kaitken17@gmail.com

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::calculate_bp_local_current_density ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::calculate_bp_local_current_density: ";

	std::string momentum_type = options.get_option("bp_analytical_momenta", std::string(""));
	//double temperature=options.get_option("temperature", NemoPhys::temperature);
	double kB_temp_in_eV = temperature*NemoPhys::kB_nemo;
	double energy_in_eV = PropagationUtilities::read_energy_from_momentum(this, momentum_point, writeable_Propagator);

	//lcd_out_file << "Energy:\t" << energy_in_eV << "\n";

	if (options.check_option("bp_analytical_momenta")) // Analytical momenta use different distributions
	{
		// Find the type of analytical momentum, i.e. 1D or 2D
		if (momentum_type == std::string("1D"))
		{
			throw std::invalid_argument(prefix + "\"1D\" analytical momentum not yet implemented for Buettiker probes\n");
		}
		else if (momentum_type == std::string("2D")) // Uses fermi integral
		{
			double index_independent_prefactor = 8.0 * NemoMath::pi * NemoPhys::elementary_charge / NemoPhys::hbar_nemo / NemoPhys::hbar_nemo / NemoPhys::hbar_nemo;
			// BP trans matrix of momenta tuple
			std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_momenta_trans_matrix = bp_trans_map->find(momentum_point);
			// NVC trans matrix of momenta tuple
			std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_momenta_trans_matrix = nvc_trans_map->find(momentum_point);

			// Loop over all Buettiker probes
			std::map<unsigned int, double>::const_iterator bp_it1 = chemical_potential_map.begin();
			for (unsigned int i = 0; bp_it1 != chemical_potential_map.end(); ++bp_it1, ++i)
			{
				//lcd_out_file << "Probe:\t" << bp_it1->first << "\n";

				// Temporary variable to hold sum of transmissions
				double sum = 0;

				// Gets effective mass
				std::vector<double> effective_mass;
				std::vector<unsigned int> dummy_input(1, 1);
				Hamilton_Constructor->get_data("effective_mass", dummy_input, effective_mass);
				// Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
				// Negative half here for efficiency (because it doesn't change between iterations)
				double temp_n = (bp_it1->second - energy_in_eV) / kB_temp_in_eV;
				// This fermi integral contains the kb_temp prefactor
				double fermi_integral_n = kB_temp_in_eV * std::log(1.0 + std::exp(temp_n));

				// Loop over all Buettiker probes
				std::map<unsigned int, double>::const_iterator bp_it2 = chemical_potential_map.begin();
				for (; bp_it2 != chemical_potential_map.end(); ++bp_it2)
				{
					if (bp_it1->first != bp_it2->first) // Exception for transmission between probe and itself
					{
						// Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
						double temp_p = (bp_it2->second - energy_in_eV) / kB_temp_in_eV;
						// This fermi integral contains the kb_temp prefactor
						double fermi_integral_p = kB_temp_in_eV * std::log(1.0 + std::exp(temp_p));

						//: Is this effective mass implemented correctly?
						double temp_p_sum = effective_mass[bp_it2->first] * NemoPhys::electron_mass * real(bp_momenta_trans_matrix->second->get(bp_it2->first,
							bp_it1->first)) * fermi_integral_p;
						double temp_n_sum = effective_mass[bp_it1->first] * NemoPhys::electron_mass * real(bp_momenta_trans_matrix->second->get(bp_it1->first,
							bp_it2->first)) * fermi_integral_n;
						// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
						sum += temp_p_sum;
						// Subtract T_pl * f_p(E)
						sum -= temp_n_sum;

						//lcd_out_file << "  BP" << bp_it2->first << "_P:\t" << temp_p_sum << "\tCP:\t" << bp_it2->second << "\tEM:\t" << effective_mass[bp_it2->first] << "\n";
						//lcd_out_file << "    Fermi Int:\t" << fermi_integral_p << "\tTemp_p:\t" << temp_p << "\n";
						//lcd_out_file << "  BP" << bp_it2->first << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\tEM:\t" << effective_mass[bp_it1->first] << "\n";
						//lcd_out_file << "    Fermi Int:\t" << fermi_integral_n << "\tTemp_n:\t" << temp_n << "\n";

					}
				}
				// Loop over all non-virtual contacts
				std::map<std::string, Simulation*>::const_iterator nvc_it = nonvirtual_contacts.begin();
				for (unsigned int j = 0; nvc_it != nonvirtual_contacts.end(); ++nvc_it, ++j) // Nonvirtual Contacts
				{
					// Add T_lp * f_l(E)
					// NOTE: This can be made more efficient by moving this get_option outside the loop
					double nvc_chemical_potential = 0.0;
					if (options.check_option(nvc_it->first + "_chemical_potential"))
					{
						nvc_chemical_potential = options.get_option(nvc_it->first + "_chemical_potential", 0.0);
					}
					else
						throw std::invalid_argument(prefix + "please define \"" + nvc_it->first + "_chemical_potential\"\n");

					// Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
					double temp_p = (nvc_chemical_potential - energy_in_eV) / kB_temp_in_eV;
					// This fermi integral contains the kb_temp prefactor
					double fermi_integral_p = kB_temp_in_eV * std::log(1.0 + std::exp(temp_p));

					//: Is this effective mass implemented correctly?
					//NOTE: Temporary corresponding effective mass location for contacts
					unsigned int temporary_nvc_locations[2] = { 0, 11 };
					double temp_p_sum = effective_mass[temporary_nvc_locations[j]] * NemoPhys::electron_mass * real(nvc_momenta_trans_matrix->second->get(bp_it1->first,
						j)) * fermi_integral_p;
					double temp_n_sum = effective_mass[bp_it1->first] * NemoPhys::electron_mass * real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * fermi_integral_n;
					// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
					sum += temp_p_sum;
					// Subtract T_pl * f_p(E)
					sum -= temp_n_sum;

					//lcd_out_file << "  NVC" << j << "_P:\t" << temp_p_sum << "\tCP:\t" << nvc_chemical_potential << "\tEM:\t" << effective_mass[temporary_nvc_locations[j]] << "\n";
					//lcd_out_file << "    Fermi Int:\t" << fermi_integral_p << "\tTemp_p:\t" << temp_p << "\n";
					//lcd_out_file << "  NVC" << j << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\tEM:\t" << effective_mass[bp_it1->first] << "\n";
					//lcd_out_file << "    Fermi Int:\t" << fermi_integral_n << "\tTemp_n:\t" << temp_n << "\n";

				}

				// lcd_out_file << "  Pre-constant sum:\t" << sum << "\n";
				//lcd_out_file << "  Constant:\t" << index_independent_prefactor << "\n";

				// Multiplies by e/h to give the current density for a particular probe at a given tuple
				sum *= index_independent_prefactor;

				// Assigns temporary sum variable to output variable
				// DEBUG: Change if necessary to assign to point in vector
				//NOTE: Could potentially change this to simply a vector<double> at a later point
				(*momenta_lcd)[i] = std::complex<double>(sum, 0.0);

				//lcd_out_file << "Probe " << bp_it1->first << " Momenta Current:\t" << sum << "\n";
			}
		}
		else
			throw std::invalid_argument(prefix + "the value of analytical_momenta can be either \"1D\" or \"2D\"\n");
	}
	else // Case for nonanalytical (normal) momenta
	{
		double index_independent_prefactor = NemoPhys::elementary_charge*NemoPhys::elementary_charge / NemoPhys::h;
		// BP trans matrix of momenta tuple
		std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_momenta_trans_matrix = bp_trans_map->find(momentum_point);
		// NVC trans matrix of momenta tuple
		std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_momenta_trans_matrix = nvc_trans_map->find(momentum_point);

		// Loop over all Buettiker probes
		std::map<unsigned int, double>::const_iterator bp_it1 = chemical_potential_map.begin();
		for (unsigned int i = 0; bp_it1 != chemical_potential_map.end(); ++bp_it1, ++i)
		{
			//lcd_out_file << "Probe:\t" << bp_it1->first << "\n";

			// Temporary variable to hold sum of transmissions
			double sum = 0;
			double sum_e = 0;
			// Loop over all Buettiker probes
			std::map<unsigned int, double>::const_iterator bp_it2 = chemical_potential_map.begin();
			for (; bp_it2 != chemical_potential_map.end(); ++bp_it2)
			{
				if (bp_it1->first != bp_it2->first) // Exception for transmission between probe and itself
				{
					double temp_p_sum;
					double temp_n_sum;
					double temp_p_sum_e=0;
					double temp_n_sum_e=0;
					if (particle_type_is_Fermion)
					{
						temp_p_sum = real(bp_momenta_trans_matrix->second->get(bp_it2->first, bp_it1->first)) * NemoMath::fermi_distribution(bp_it2->second, kB_temp_in_eV,
							energy_in_eV);
						temp_n_sum = real(bp_momenta_trans_matrix->second->get(bp_it1->first, bp_it2->first)) * NemoMath::fermi_distribution(bp_it1->second, kB_temp_in_eV,
							energy_in_eV);
						temp_p_sum_e = (energy_in_eV - bp_it1->second)*real(bp_momenta_trans_matrix->second->get(bp_it2->first, bp_it1->first)) * NemoMath::fermi_distribution(bp_it2->second, kB_temp_in_eV,
							energy_in_eV);
						temp_n_sum_e = (energy_in_eV - bp_it1->second)*real(bp_momenta_trans_matrix->second->get(bp_it1->first, bp_it2->first)) * NemoMath::fermi_distribution(bp_it1->second, kB_temp_in_eV,
							energy_in_eV);
					}
					/*
					else if (particle_type_is_Fermion && options.get_option("electron_potential_driven",false))
					{
					temp_p_sum = (energy_in_eV-bp_it2->second)*real(bp_momenta_trans_matrix->second->get(bp_it2->first, bp_it1->first)) * NemoMath::fermi_distribution(bp_it2->second, kB_temp_in_eV,
					energy_in_eV);
					temp_n_sum = (energy_in_eV-bp_it2->second)*real(bp_momenta_trans_matrix->second->get(bp_it1->first, bp_it2->first)) * NemoMath::fermi_distribution(bp_it1->second, kB_temp_in_eV,
					energy_in_eV);
					}
					*/
					else if (!particle_type_is_Fermion)
					{
						double kB_temp_in_eV_bp1 = bp_it1->second*NemoPhys::kB_nemo;
						double kB_temp_in_eV_bp2 = bp_it2->second*NemoPhys::kB_nemo;
						temp_p_sum = energy_in_eV*real(bp_momenta_trans_matrix->second->get(bp_it2->first, bp_it1->first)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_bp2,
							energy_in_eV);
						temp_n_sum = energy_in_eV*real(bp_momenta_trans_matrix->second->get(bp_it1->first, bp_it2->first)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_bp1,
							energy_in_eV);
					}

					// Add T_lp * f_l(E)
					sum += temp_p_sum;
					// Subtract T_pl * f_p(E)
					sum -= temp_n_sum;
					// Add T_lp * f_l(E)
					sum_e += temp_p_sum_e;
					// Subtract T_pl * f_p(E)
					sum_e -= temp_n_sum_e;

					//lcd_out_file << "  BP" << bp_it2->first << "_P:\t" << temp_p_sum << "\tCP:\t" << bp_it2->second << "\n";
					//lcd_out_file << "  BP" << bp_it2->first << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\n";
				}
			}
			// Loop over all non-virtual contacts
			std::map<std::string, Simulation*>::const_iterator nvc_it = nonvirtual_contacts.begin();
			for (unsigned int j = 0; nvc_it != nonvirtual_contacts.end(); ++nvc_it, ++j) // Nonvirtual Contacts
			{
				// Add T_lp * f_l(E)
				// NOTE: This can be made more efficient by moving this get_option outside the loop
				double nvc_chemical_potential = 0.0;
				double temp_p_sum;
				double temp_n_sum;
				double temp_p_sum_e=0;
				double temp_n_sum_e=0;
				if (options.check_option(nvc_it->first + "_chemical_potential"))
				{
					nvc_chemical_potential = options.get_option(nvc_it->first + "_chemical_potential", 0.0);
				}
				else
					throw std::invalid_argument(prefix + "please define \"" + nvc_it->first + "_chemical_potential\"\n");
				if (particle_type_is_Fermion)
				{
					temp_p_sum = real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::fermi_distribution(nvc_chemical_potential, kB_temp_in_eV,
						energy_in_eV);
					temp_n_sum = real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::fermi_distribution(bp_it1->second, kB_temp_in_eV, energy_in_eV);
					temp_p_sum_e = (energy_in_eV - bp_it1->second)*real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::fermi_distribution(nvc_chemical_potential, kB_temp_in_eV,
						energy_in_eV);
					temp_n_sum_e = (energy_in_eV - bp_it1->second)*real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::fermi_distribution(bp_it1->second, kB_temp_in_eV, energy_in_eV);
				}
				else if (!particle_type_is_Fermion)
				{
					double kB_temp_in_eV_nvc = nvc_chemical_potential*NemoPhys::kB_nemo;
					double kB_temp_in_eV_bp = bp_it1->second*NemoPhys::kB_nemo;
					temp_p_sum = energy_in_eV*real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_nvc,
						energy_in_eV);
					temp_n_sum = energy_in_eV*real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_bp,
						energy_in_eV);
				}
				// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
				sum += temp_p_sum;
				// Subtract T_pl * f_p(E)
				sum -= temp_n_sum;
				sum_e += temp_p_sum_e;
				// Subtract T_pl * f_p(E)
				sum_e -= temp_n_sum_e;

				//lcd_out_file << "  NVC" << j << "_P:\t" << temp_p_sum << "\tCP:\t" << nvc_chemical_potential << "\n";
				//lcd_out_file << "  kB_temp_in_eV:\t" << kB_temp_in_eV << "\tTV1:\t" << NemoMath::fermi_distribution(nvc_chemical_potential, kB_temp_in_eV, energy_in_eV) << "\tTrans:\t" << nvc_trans_map->find(momentum_point)->second->get(bp_it1->first, j) << "\n";
				//lcd_out_file << "  NVC" << j << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\n";
			}

			//lcd_out_file << "  Pre-constant sum:\t" << sum << "\n";
			//lcd_out_file << "  Constant:\t" << index_independent_prefactor << "\n";

			// Multiplies by e/h to give the current density for a particular probe at a given tuple
			sum *= index_independent_prefactor;
			sum_e *= index_independent_prefactor;

			// Assigns temporary sum variable to output variable
			// DEBUG: Change if necessary to assign to point in vector
			//NOTE: Could potentially change this to simply a vector<double> at a later point
			(*momenta_lcd)[i] = std::complex<double>(sum, 0.0);
			(*momenta_lecd)[i] = std::complex<double>(sum_e, 0.0);
			//lcd_out_file << "Probe " << bp_it1->first << " Momenta Current:\t" << sum << "\n";
		}
	}
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_calculate_temperature_local_current_density(const std::vector<NemoMeshPoint>& momentum_point, std::map<std::string, Simulation*> nonvirtual_contacts,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map,
	std::vector<std::complex<double> >*& momenta_lcd, std::vector<std::complex<double> >*& momenta_lecd)
{

	// Calculates the current_divergence for ALL Buettiker probes for a particular momenta(for Buettiker probes)
	// j_p(momentum_point) = (e/h) * sum_(l = all contacts)(T_lp * f_l(E) - T_pl * f_p(E))
	// Under development, Kyle Aitken: kaitken17@gmail.com

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::calculate_bp_local_current_density ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::calculate_bp_local_current_density: ";

	std::string momentum_type = options.get_option("bp_analytical_momenta", std::string(""));
	//double temperature=options.get_option("temperature", NemoPhys::temperature);
	double energy_in_eV = PropagationUtilities::read_energy_from_momentum(this, momentum_point, writeable_Propagator);

	double index_independent_prefactor = NemoPhys::elementary_charge*NemoPhys::elementary_charge / NemoPhys::h;
	// BP trans matrix of momenta tuple
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_momenta_trans_matrix = bp_trans_map->find(momentum_point);
	// NVC trans matrix of momenta tuple
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_momenta_trans_matrix = nvc_trans_map->find(momentum_point);
	// Loop over all Buettiker probes
	std::map<unsigned int, double>::const_iterator bp_it1 = chemical_potential_map.begin();
	for (unsigned int i = 0; bp_it1 != chemical_potential_map.end(); ++bp_it1, ++i)
	{
		// Temporary variable to hold sum of transmissions
		double sum = 0;
		double sum_e = 0;
		// Loop over all Buettiker probes
		std::map<unsigned int, double>::const_iterator bp_it2 = chemical_potential_map.begin();
		double kB_temp_in_eV_bp1 = bp_it1->second*NemoPhys::kB_nemo;
		double chemical_potential = 0.0;
		if (options.check_option("chemical_potential"))
		{
			chemical_potential = options.get_option("chemical_potential", 0.0);
		}
		else
			throw std::invalid_argument(prefix + "please define chemical_potential\"\n");
		for (; bp_it2 != chemical_potential_map.end(); ++bp_it2)
		{
			if (bp_it1->first != bp_it2->first) // Exception for transmission between probe and itself
			{
				double temp_p_sum;
				double temp_n_sum;
				double kB_temp_in_eV_bp2 = bp_it2->second*NemoPhys::kB_nemo;

				temp_p_sum = real(bp_momenta_trans_matrix->second->get(bp_it2->first, bp_it1->first)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_bp2,
					energy_in_eV);
				temp_n_sum = real(bp_momenta_trans_matrix->second->get(bp_it1->first, bp_it2->first)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_bp1,
					energy_in_eV);
				// Add T_lp * f_l(E)
				sum += temp_p_sum;
				// Subtract T_pl * f_p(E)
				sum -= temp_n_sum;
				// Add T_lp * f_l(E)
				sum_e += temp_p_sum*(energy_in_eV - chemical_potential);
				// Subtract T_pl * f_p(E)
				sum_e -= temp_n_sum*(energy_in_eV - chemical_potential);
			}
		}
		// Loop over all non-virtual contacts
		std::map<std::string, Simulation*>::const_iterator nvc_it = nonvirtual_contacts.begin();
		for (unsigned int j = 0; nvc_it != nonvirtual_contacts.end(); ++nvc_it, ++j) // Nonvirtual Contacts
		{
			// Add T_lp * f_l(E)
			// NOTE: This can be made more efficient by moving this get_option outside the loop
			double nvc_temperature = 0.0;
			double temp_p_sum;
			double temp_n_sum;
			if (options.check_option(nvc_it->first + "_temperature"))
			{
				nvc_temperature = options.get_option(nvc_it->first + "_temperature", 0.0);
			}
			else
				throw std::invalid_argument(prefix + "please define \"" + nvc_it->first + "_temperature\"\n");
			double kB_temp_in_eV_nvc = nvc_temperature*NemoPhys::kB_nemo;
			temp_p_sum = real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_nvc,
				energy_in_eV);
			temp_n_sum = real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_bp1, energy_in_eV);
			// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
			sum += temp_p_sum;
			// Subtract T_pl * f_p(E)
			sum -= temp_n_sum;

			sum_e += temp_p_sum*(energy_in_eV - chemical_potential);
			sum_e -= temp_n_sum*(energy_in_eV - chemical_potential);

		}
		// Multiplies by e/h to give the current density for a particular probe at a given tuple
		sum *= index_independent_prefactor;
		sum_e *= index_independent_prefactor;
		// Assigns temporary sum variable to output variable
		// DEBUG: Change if necessary to assign to point in vector
		//NOTE: Could potentially change this to simply a vector<double> at a later point
		(*momenta_lcd)[i] = std::complex<double>(sum, 0.0);
		(*momenta_lecd)[i] = std::complex<double>(sum_e, 0.0);
	}
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_calculate_contact_local_current_density(const std::vector<NemoMeshPoint>& momentum_point, std::map<std::string, Simulation*> nonvirtual_contacts,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map,
	std::vector<std::complex<double> >*& momenta_lcd)
{

	// Calculates the current_divergence for ALL Buettiker probes for a particular momenta(for Buettiker probes)
	// j_p(momentum_point) = (e/h) * sum_(l = all contacts)(T_lp * f_l(E) - T_pl * f_p(E))
	// Under development, 

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::calculate_bp_local_current_density ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::calculate_bp_local_current_density: ";

	std::string momentum_type = options.get_option("bp_analytical_momenta", std::string(""));
	//double temperature=options.get_option("temperature", NemoPhys::temperature);
	double kB_temp_in_eV = temperature*NemoPhys::kB_nemo;
	double energy_in_eV = PropagationUtilities::read_energy_from_momentum(this, momentum_point, writeable_Propagator);

	//lcd_out_file << "Energy:\t" << energy_in_eV << "\n";

	if (options.check_option("bp_analytical_momenta")) // Analytical momenta use different distributions
	{
		// Find the type of analytical momentum, i.e. 1D or 2D
		if (momentum_type == std::string("1D"))
		{
			throw std::invalid_argument(prefix + "\"1D\" analytical momentum not yet implemented for Buettiker probes\n");
		}
		else if (momentum_type == std::string("2D")) // Uses fermi integral
		{
			double index_independent_prefactor = 8.0 * NemoMath::pi * NemoPhys::elementary_charge / NemoPhys::hbar_nemo / NemoPhys::hbar_nemo / NemoPhys::hbar_nemo;
			// BP trans matrix of momenta tuple
			std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_momenta_trans_matrix = bp_trans_map->find(momentum_point);
			// NVC trans matrix of momenta tuple
			std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_momenta_trans_matrix = nvc_trans_map->find(momentum_point);

			// Loop over all Buettiker probes
			std::map<unsigned int, double>::const_iterator bp_it1 = chemical_potential_map.begin();
			for (unsigned int i = 0; bp_it1 != chemical_potential_map.end(); ++bp_it1, ++i)
			{
				//lcd_out_file << "Probe:\t" << bp_it1->first << "\n";

				// Temporary variable to hold sum of transmissions
				double sum = 0;

				// Gets effective mass
				std::vector<double> effective_mass;
				std::vector<unsigned int> dummy_input(1, 1);
				Hamilton_Constructor->get_data("effective_mass", dummy_input, effective_mass);
				// Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
				// Negative half here for efficiency (because it doesn't change between iterations)
				double temp_n = (bp_it1->second - energy_in_eV) / kB_temp_in_eV;
				// This fermi integral contains the kb_temp prefactor
				double fermi_integral_n = kB_temp_in_eV * std::log(1.0 + std::exp(temp_n));

				// Loop over all Buettiker probes
				std::map<unsigned int, double>::const_iterator bp_it2 = chemical_potential_map.begin();
				for (; bp_it2 != chemical_potential_map.end(); ++bp_it2)
				{
					if (bp_it1->first != bp_it2->first) // Exception for transmission between probe and itself
					{
						// Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
						double temp_p = (bp_it2->second - energy_in_eV) / kB_temp_in_eV;
						// This fermi integral contains the kb_temp prefactor
						double fermi_integral_p = kB_temp_in_eV * std::log(1.0 + std::exp(temp_p));

						//: Is this effective mass implemented correctly?
						double temp_p_sum = effective_mass[bp_it2->first] * NemoPhys::electron_mass * real(bp_momenta_trans_matrix->second->get(bp_it2->first,
							bp_it1->first)) * fermi_integral_p;
						double temp_n_sum = effective_mass[bp_it1->first] * NemoPhys::electron_mass * real(bp_momenta_trans_matrix->second->get(bp_it1->first,
							bp_it2->first)) * fermi_integral_n;
						// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
						sum += temp_p_sum;
						// Subtract T_pl * f_p(E)
						sum -= temp_n_sum;

						//lcd_out_file << "  BP" << bp_it2->first << "_P:\t" << temp_p_sum << "\tCP:\t" << bp_it2->second << "\tEM:\t" << effective_mass[bp_it2->first] << "\n";
						//lcd_out_file << "    Fermi Int:\t" << fermi_integral_p << "\tTemp_p:\t" << temp_p << "\n";
						//lcd_out_file << "  BP" << bp_it2->first << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\tEM:\t" << effective_mass[bp_it1->first] << "\n";
						//lcd_out_file << "    Fermi Int:\t" << fermi_integral_n << "\tTemp_n:\t" << temp_n << "\n";

					}
				}
				// Loop over all non-virtual contacts
				std::map<std::string, Simulation*>::const_iterator nvc_it = nonvirtual_contacts.begin();
				for (unsigned int j = 0; nvc_it != nonvirtual_contacts.end(); ++nvc_it, ++j) // Nonvirtual Contacts
				{
					// Add T_lp * f_l(E)
					// NOTE: This can be made more efficient by moving this get_option outside the loop
					double nvc_chemical_potential = 0.0;
					if (options.check_option(nvc_it->first + "_chemical_potential"))
					{
						nvc_chemical_potential = options.get_option(nvc_it->first + "_chemical_potential", 0.0);
					}
					else
						throw std::invalid_argument(prefix + "please define \"" + nvc_it->first + "_chemical_potential\"\n");

					// Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
					double temp_p = (nvc_chemical_potential - energy_in_eV) / kB_temp_in_eV;
					// This fermi integral contains the kb_temp prefactor
					double fermi_integral_p = kB_temp_in_eV * std::log(1.0 + std::exp(temp_p));

					//: Is this effective mass implemented correctly?
					//NOTE: Temporary corresponding effective mass location for contacts
					unsigned int temporary_nvc_locations[2] = { 0, 11 };
					double temp_p_sum = effective_mass[temporary_nvc_locations[j]] * NemoPhys::electron_mass * real(nvc_momenta_trans_matrix->second->get(bp_it1->first,
						j)) * fermi_integral_p;
					double temp_n_sum = effective_mass[bp_it1->first] * NemoPhys::electron_mass * real(nvc_momenta_trans_matrix->second->get(bp_it1->first, j)) * fermi_integral_n;
					// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
					sum += temp_p_sum;
					// Subtract T_pl * f_p(E)
					sum -= temp_n_sum;

					//lcd_out_file << "  NVC" << j << "_P:\t" << temp_p_sum << "\tCP:\t" << nvc_chemical_potential << "\tEM:\t" << effective_mass[temporary_nvc_locations[j]] << "\n";
					//lcd_out_file << "    Fermi Int:\t" << fermi_integral_p << "\tTemp_p:\t" << temp_p << "\n";
					//lcd_out_file << "  NVC" << j << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\tEM:\t" << effective_mass[bp_it1->first] << "\n";
					//lcd_out_file << "    Fermi Int:\t" << fermi_integral_n << "\tTemp_n:\t" << temp_n << "\n";

				}

				//lcd_out_file << "  Pre-constant sum:\t" << sum << "\n";
				//lcd_out_file << "  Constant:\t" << index_independent_prefactor << "\n";

				// Multiplies by e/h to give the current density for a particular probe at a given tuple
				sum *= index_independent_prefactor;

				// Assigns temporary sum variable to output variable
				// DEBUG: Change if necessary to assign to point in vector
				//NOTE: Could potentially change this to simply a vector<double> at a later point
				(*momenta_lcd)[i] = std::complex<double>(sum, 0.0);

				//lcd_out_file << "Probe " << bp_it1->first << " Momenta Current:\t" << sum << "\n";
			}
		}
		else
			throw std::invalid_argument(prefix + "the value of analytical_momenta can be either \"1D\" or \"2D\"\n");
	}
	else // Case for nonanalytical (normal) momenta
	{
		double index_independent_prefactor = NemoPhys::elementary_charge*NemoPhys::elementary_charge / NemoPhys::h;
		// BP trans matrix of momenta tuple
		std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_momenta_trans_matrix = bp_trans_map->find(momentum_point);
		// NVC trans matrix of momenta tuple
		std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_momenta_trans_matrix = nvc_trans_map->find(momentum_point);

		// Loop over all Buettiker probes
		std::map<std::string, Simulation*>::const_iterator nvc_it1 = nonvirtual_contacts.begin();
		for (unsigned int i = 0; nvc_it1 != nonvirtual_contacts.end(); ++nvc_it1, ++i)
		{
			//lcd_out_file << "Probe:\t" << nvc_it1->first << "\n";
			double nvc_chemical_potential1 = 0.0;
			if (options.check_option(nvc_it1->first + "_chemical_potential"))
			{
				nvc_chemical_potential1 = options.get_option(nvc_it1->first + "_chemical_potential", 0.0);
			}
			else
				throw std::invalid_argument(prefix + "please define \"" + nvc_it1->first + "_chemical_potential\"\n");
			// Temporary variable to hold sum of transmissions
			double sum = 0;
			// Loop over all Buettiker probes
			std::map<std::string, Simulation*>::const_iterator nvc_it2 = nonvirtual_contacts.begin();
			for (unsigned int j = 0; nvc_it2 != nonvirtual_contacts.end(); ++nvc_it2, ++j)
			{
				if (i != j) // Exception for transmission between probe and itself
				{
					double nvc_chemical_potential2 = 0.0;
					if (options.check_option(nvc_it2->first + "_chemical_potential"))
					{
						nvc_chemical_potential2 = options.get_option(nvc_it2->first + "_chemical_potential", 0.0);
					}
					else
						throw std::invalid_argument(prefix + "please define \"" + nvc_it2->first + "_chemical_potential\"\n");
					double temp_p_sum;
					double temp_n_sum;
					if (particle_type_is_Fermion)
					{
						temp_p_sum = real(nvc_momenta_trans_matrix->second->get(j, i)) * NemoMath::fermi_distribution(nvc_chemical_potential1, kB_temp_in_eV,
							energy_in_eV);
						temp_n_sum = real(nvc_momenta_trans_matrix->second->get(i, j)) * NemoMath::fermi_distribution(nvc_chemical_potential2, kB_temp_in_eV,
							energy_in_eV);
					}
					else if (!particle_type_is_Fermion)
					{
						double kB_temp_in_eV_nvc1 = nvc_chemical_potential1*NemoPhys::kB_nemo;
						double kB_temp_in_eV_nvc2 = nvc_chemical_potential2*NemoPhys::kB_nemo;
						temp_p_sum = energy_in_eV*real(nvc_momenta_trans_matrix->second->get(j, i)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_nvc2,
							energy_in_eV);
						temp_n_sum = energy_in_eV*real(nvc_momenta_trans_matrix->second->get(i, j)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_nvc1,
							energy_in_eV);
					}

					// Add T_lp * f_l(E)
					sum += temp_p_sum;
					// Subtract T_pl * f_p(E)
					sum -= temp_n_sum;

					//lcd_out_file << "  BP" << bp_it2->first << "_P:\t" << temp_p_sum << "\tCP:\t" << bp_it2->second << "\n";
					//lcd_out_file << "  BP" << bp_it2->first << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\n";
				}
			}
			// Loop over all non-virtual contacts
			std::map<unsigned int, double>::const_iterator bp_it = chemical_potential_map.begin();
			for (; bp_it != chemical_potential_map.end(); ++bp_it) // Nonvirtual Contacts
			{
				double temp_p_sum;
				double temp_n_sum;
				if (particle_type_is_Fermion)
				{
					temp_p_sum = real(bp_momenta_trans_matrix->second->get(bp_it->first, i)) * NemoMath::fermi_distribution(nvc_chemical_potential1, kB_temp_in_eV,
						energy_in_eV);
					temp_n_sum = real(bp_momenta_trans_matrix->second->get(bp_it->first, i)) * NemoMath::fermi_distribution(bp_it->second, kB_temp_in_eV, energy_in_eV);
				}
				else if (!particle_type_is_Fermion)
				{
					double kB_temp_in_eV_nvc = nvc_chemical_potential1*NemoPhys::kB_nemo;
					double kB_temp_in_eV_bp = bp_it->second*NemoPhys::kB_nemo;
					temp_p_sum = energy_in_eV*real(bp_momenta_trans_matrix->second->get(bp_it->first, i)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_bp,
						energy_in_eV);
					temp_n_sum = energy_in_eV*real(bp_momenta_trans_matrix->second->get(bp_it->first, i)) * NemoMath::bose_distribution(0.0, kB_temp_in_eV_nvc,
						energy_in_eV);
				}

				// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
				sum += temp_p_sum;
				// Subtract T_pl * f_p(E)
				sum -= temp_n_sum;

				//lcd_out_file << "  NVC" << j << "_P:\t" << temp_p_sum << "\tCP:\t" << nvc_chemical_potential << "\n";
				////lcd_out_file << "  kB_temp_in_eV:\t" << kB_temp_in_eV << "\tTV1:\t" << NemoMath::fermi_distribution(nvc_chemical_potential, kB_temp_in_eV, energy_in_eV) << "\tTrans:\t" << nvc_trans_map->find(momentum_point)->second->get(bp_it1->first, j) << "\n";
				//lcd_out_file << "  NVC" << j << "_N:\t" << -1 * temp_n_sum << "\tCP:\t" << bp_it1->second << "\n";
			}

			//lcd_out_file << "  Pre-constant sum:\t" << sum << "\n";
			//lcd_out_file << "  Constant:\t" << index_independent_prefactor << "\n";

			// Multiplies by e/h to give the current density for a particular probe at a given tuple
			sum *= index_independent_prefactor;

			// Assigns temporary sum variable to output variable
			// DEBUG: Change if necessary to assign to point in vector
			//NOTE: Could potentially change this to simply a vector<double> at a later point
			(*momenta_lcd)[i] = std::complex<double>(sum, 0.0);

			//lcd_out_file << "Probe " << nvc_it1->first << " Momenta Current:\t" << sum << "\n";
		}
	}
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_calculate_contact_temperature_local_current_density(const std::vector<NemoMeshPoint>& momentum_point, std::map<std::string, Simulation*> nonvirtual_contacts,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map,
	std::vector<std::complex<double> >*& momenta_lcd)
{

	// Calculates the current_divergence for ALL Buettiker probes for a particular momenta(for Buettiker probes)
	// j_p(momentum_point) = (e/h) * sum_(l = all contacts)(T_lp * f_l(E) - T_pl * f_p(E))
	// Under development,

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::calculate_bp_local_current_density ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::calculate_bp_local_current_density: ";

	double energy_in_eV = PropagationUtilities::read_energy_from_momentum(this, momentum_point, writeable_Propagator);
	double chemical_potential = 0.0;
	if (options.check_option("chemical_potential"))
	{
		chemical_potential = options.get_option("chemical_potential", 0.0);
	}
	else
		throw std::invalid_argument(prefix + "please define chemical_potential\"\n");
	double index_independent_prefactor = NemoPhys::elementary_charge*NemoPhys::elementary_charge / NemoPhys::h;
	// BP trans matrix of momenta tuple
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_momenta_trans_matrix = bp_trans_map->find(momentum_point);
	// NVC trans matrix of momenta tuple
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_momenta_trans_matrix = nvc_trans_map->find(momentum_point);
	// Loop over all Buettiker probes
	std::map<std::string, Simulation*>::const_iterator nvc_it1 = nonvirtual_contacts.begin();
	for (unsigned int i = 0; nvc_it1 != nonvirtual_contacts.end(); ++nvc_it1, ++i)
	{
		//lcd_out_file << "Probe:\t" << nvc_it1->first << "\n";
		double temperature1 = 0.0;
		if (options.check_option(nvc_it1->first + "_temperature"))
		{
			temperature1 = options.get_option(nvc_it1->first + "_temperature", 0.0);
		}
		else
			throw std::invalid_argument(prefix + "please define \"" + nvc_it1->first + "_temperature\"\n");
		// Temporary variable to hold sum of transmissions
		double kB_temp_in_eV_nvc1 = temperature1*NemoPhys::kB_nemo;
		double sum = 0;
		// Loop over all Buettiker probes
		std::map<std::string, Simulation*>::const_iterator nvc_it2 = nonvirtual_contacts.begin();
		for (unsigned int j = 0; nvc_it2 != nonvirtual_contacts.end(); ++nvc_it2, ++j)
		{
			if (i != j) // Exception for transmission between probe and itself
			{
				double temperature2 = 0.0;
				if (options.check_option(nvc_it2->first + "_temperature"))
				{
					temperature2 = options.get_option(nvc_it2->first + "_temperature", 0.0);
				}
				else
					throw std::invalid_argument(prefix + "please define \"" + nvc_it2->first + "_temperature\"\n");
				double kB_temp_in_eV_nvc2 = temperature2*NemoPhys::kB_nemo;
				double temp_p_sum;
				double temp_n_sum;
				temp_p_sum = real(nvc_momenta_trans_matrix->second->get(j, i)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_nvc1,
					energy_in_eV);
				temp_n_sum = real(nvc_momenta_trans_matrix->second->get(i, j)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_nvc2,
					energy_in_eV);
				// Add T_lp * f_l(E)
				sum += temp_p_sum;
				// Subtract T_pl * f_p(E)
				sum -= temp_n_sum;
			}
		}
		// Loop over all non-virtual contacts
		std::map<unsigned int, double>::const_iterator bp_it = chemical_potential_map.begin();
		for (; bp_it != chemical_potential_map.end(); ++bp_it) // Nonvirtual Contacts
		{
			double temp_p_sum;
			double temp_n_sum;
			double kB_temp_in_eV_bp = bp_it->second*NemoPhys::kB_nemo;
			temp_p_sum = real(bp_momenta_trans_matrix->second->get(bp_it->first, i)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_nvc1,
				energy_in_eV);
			temp_n_sum = real(bp_momenta_trans_matrix->second->get(bp_it->first, i)) * NemoMath::fermi_distribution(chemical_potential, kB_temp_in_eV_bp, energy_in_eV);
			// NOTE: For now we assume the transmission from the real contact to the BP and vice versa is the same, add a fix in later (switch first get to get(j, bp_it1->first))
			sum += temp_p_sum;
			// Subtract T_pl * f_p(E)
			sum -= temp_n_sum;
		}
		// Multiplies by e/h to give the current density for a particular probe at a given tuple
		sum *= index_independent_prefactor;

		// Assigns temporary sum variable to output variable
		// DEBUG: Change if necessary to assign to point in vector
		//NOTE: Could potentially change this to simply a vector<double> at a later point
		(*momenta_lcd)[i] = std::complex<double>(sum, 0.0);
	}
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_energy_relaxed_local_current(std::map<std::string, Simulation*> nonvirtual_contacts, const Propagator* ret_Green,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map,
	std::vector<std::complex<double> >*& local_current_density, std::vector<std::complex<double> >*& local_energy_current_density)
{
	// Calculates the total (integrated over all momenta) local current at each virtual contact (for Buettiker probes)
	// F_p^(i) = int(dE, j_p[mu_1^(i),...](E)) for all p in P
	// Under development, Kyle Aitken: kaitken17@gmail.com

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::bp_energy_relaxed_local_current ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::bp_energy_relaxed_local_current: ";

	// Output - for test purposes
	std::string filename = "bp_txt_lcd_all_output.txt";
	//std::ofstream lcd_out_file;
	//lcd_out_file.open(output_collector.get_file_path(filename,
	//"File for testing, Contains information of the local current between all BP and contacts as well as the chemical potentials",NemoFileSystem::DEBUG).c_str());
	//lcd_out_file << "BP Local Current Density Status\n\n";

	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*> momenta_lcd_map;
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>* momenta_lcd_map_pointer = &momenta_lcd_map;
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*> momenta_lecd_map;
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>* momenta_lecd_map_pointer = &momenta_lecd_map;
	//std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>* momenta_lecd_map_pointer = &momenta_lecd_map;

	// Loop over all momenta tuples and create a pointer to a complex for each value, but only point to NULL for the time being
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator it = ret_Green->propagator_map.begin();
	for (; it != ret_Green->propagator_map.end(); ++it)
	{
		momenta_lcd_map[it->first] = NULL;
		momenta_lecd_map[it->first] = NULL;
	}

	// Loop over all momenta
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>::iterator momenta_it = momenta_lcd_map.begin();
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>::iterator momenta_it_e = momenta_lecd_map.begin();
	for (; momenta_it != momenta_lcd_map.end(); ++momenta_it, ++momenta_it_e)
	{
		// DEBUG: Prints information for mesh point
		//lcd_out_file << "\n\n-----------------------MOMENTA POINT (";
		//for(unsigned int i=0; i<momenta_it->first.size(); ++i) // Index for each momenta
		//{
		//lcd_out_file << momenta_it->first[i].get_idx() << " ";
		//}
		//lcd_out_file << ")-----------------------\n";

		// Initializes vector and assigns results
		momenta_it->second = new std::vector<std::complex<double> >(chemical_potential_map.size());
		momenta_it_e->second = new std::vector<std::complex<double> >(chemical_potential_map.size());
		if (options.get_option("electron_temperature_driven", false) && particle_type_is_Fermion)
			//bp_calculate_local_current_density(momenta_it->first, nonvirtual_contacts, bp_trans_map, nvc_trans_map, momenta_it->second);      
			bp_calculate_temperature_local_current_density(momenta_it->first, nonvirtual_contacts, bp_trans_map, nvc_trans_map, momenta_it->second, momenta_it_e->second);
		else
			bp_calculate_local_current_density(momenta_it->first, nonvirtual_contacts, bp_trans_map, nvc_trans_map, momenta_it->second, momenta_it_e->second);
		//NOTE: Add NEMO_ASSERT here

		// DEBUG: Prints result
		//for(unsigned int i=0;i<momenta_it->second->size();++i)
		//{
		//  lcd_out_file << i << ":\t" << (*(momenta_it->second))[i] << "\n";
		//}
	}
	double prefactor = 1;
	//const std::vector<std::string>& temp_mesh_names=ret_Green->momentum_mesh_names;
	/*for (unsigned int i=0; i<temp_mesh_names.size(); i++)
	{
	if(temp_mesh_names[i].find("energy")!=std::string::npos || temp_mesh_names[i].find("momentum_1D")!=std::string::npos)
	prefactor/=2.0*NemoMath::pi;
	else if(temp_mesh_names[i].find("momentum_2D")!=std::string::npos)
	prefactor/=4.0*NemoMath::pi*NemoMath::pi;
	else if(temp_mesh_names[i].find("momentum_3D")!=std::string::npos)
	prefactor/=8.0*NemoMath::pi*NemoMath::pi*NemoMath::pi;
	}*/
	double real_prefactor = options.get_option("_energy_resolved_output_real", 1.0);
	double imag_prefactor = options.get_option("_energy_resolved_output_imag", 0.0);
	std::complex<double> total_input_prefactor(real_prefactor, imag_prefactor);

	std::map<double, std::vector<std::complex<double> > > energy_resolved_data;
	std::map<double, std::vector<std::complex<double> > >* energy_resolved_data_pointer = &energy_resolved_data;

	std::map<double, std::map<unsigned int, double>, Compare_double_or_complex_number > energy_resolved_density_output;
	if (options.get_option("output_energy_resolved_current", false))
	{
		integrate_vector_for_energy_resolved(momenta_lcd_map_pointer, energy_resolved_data_pointer);

		std::map<double, std::vector<std::complex<double> > >::iterator it = energy_resolved_data.begin();
		for (; it != energy_resolved_data.end(); ++it)
		{
			std::map<unsigned int, double> temp_map;
			translate_vector_into_map(it->second, total_input_prefactor*prefactor*std::complex<double>(1.0, 0.0), true, temp_map);

			energy_resolved_density_output[it->first] = temp_map;

			std::vector<std::complex<double> >::iterator vec_it = it->second.begin();
			for (; vec_it != it->second.end(); ++vec_it)
				cerr << "E[eV] " << it->first << " value " << *vec_it << " \n";
		}

		int my_local_rank = -1;
		MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);
		std::string filename = "energy_resolved_current";
		if (my_local_rank == 0)
		{
			print_atomic_maps(energy_resolved_density_output, filename);
			reset_output_counter();

		}
	}

	// Integrate over all local momenta using integrate_diagonal
	std::vector<std::complex<double> > result(chemical_potential_map.size());
	std::vector<std::complex<double> >* result_pointer = &result;
	bp_integrate_over_momenta(ret_Green, momenta_lcd_map_pointer, result_pointer);

	// Assign to output
	*local_current_density = result;

	std::vector<std::complex<double> > result_e(chemical_potential_map.size());
	std::vector<std::complex<double> >* result_e_pointer = &result_e;
	bp_integrate_over_momenta(ret_Green, momenta_lecd_map_pointer, result_e_pointer);

	// Assign to output
	*local_energy_current_density = result_e;

	//lcd_out_file.close();
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_energy_relaxed_local_energy_current(std::map<std::string, Simulation*> nonvirtual_contacts, const Propagator* ret_Green,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map,
	std::vector<std::complex<double> >*& local_energy_current_density)
{
	// Calculates the total (integrated over all momenta) local current at each virtual contact (for Buettiker probes)
	// F_p^(i) = int(dE, j_p[mu_1^(i),...](E)) for all p in P
	// Under development, Kyle Aitken: kaitken17@gmail.com

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::bp_energy_relaxed_local_current ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::bp_energy_relaxed_local_current: ";

	// Output - for test purposes
	std::string filename = "bp_txt_lcd_all_output.txt";
	//std::ofstream lcd_out_file;
	//lcd_out_file.open(output_collector.get_file_path(filename,
	//"File for testing, Contains information of the local current between all BP and contacts as well as the chemical potentials",NemoFileSystem::DEBUG).c_str());
	//lcd_out_file << "BP Local Current Density Status\n\n";
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*> momenta_lcd_map;
	//std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>* momenta_lcd_map_pointer = &momenta_lcd_map;
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*> momenta_lecd_map;
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>* momenta_lecd_map_pointer = &momenta_lecd_map;

	// Loop over all momenta tuples and create a pointer to a complex for each value, but only point to NULL for the time being
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator it = ret_Green->propagator_map.begin();
	for (; it != ret_Green->propagator_map.end(); ++it)
	{
		momenta_lcd_map[it->first] = NULL;
		momenta_lecd_map[it->first] = NULL;
	}

	// Loop over all momenta
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>::iterator momenta_it = momenta_lcd_map.begin();
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>::iterator momenta_it_e = momenta_lecd_map.begin();
	for (; momenta_it != momenta_lcd_map.end(); ++momenta_it, ++momenta_it_e)
	{
		// DEBUG: Prints information for mesh point
		//lcd_out_file << "\n\n-----------------------MOMENTA POINT (";
		//for(unsigned int i=0; i<momenta_it->first.size(); ++i) // Index for each momenta
		//{
		//lcd_out_file << momenta_it->first[i].get_idx() << " ";
		//}
		//lcd_out_file << ")-----------------------\n";

		// Initializes vector and assigns results
		momenta_it->second = new std::vector<std::complex<double> >(chemical_potential_map.size());
		momenta_it_e->second = new std::vector<std::complex<double> >(chemical_potential_map.size());
		bp_calculate_temperature_local_current_density(momenta_it->first, nonvirtual_contacts, bp_trans_map, nvc_trans_map, momenta_it->second, momenta_it_e->second);

		//NOTE: Add NEMO_ASSERT here

		// DEBUG: Prints result
		//for(unsigned int i=0;i<momenta_it->second->size();++i)
		//{
		//  lcd_out_file << i << ":\t" << (*(momenta_it->second))[i] << "\n";
		//}
	}
	//double prefactor = 1;


	std::map<double, std::vector<std::complex<double> > > energy_resolved_data;
	//std::map<double, std::vector<std::complex<double> > >* energy_resolved_data_pointer = &energy_resolved_data;

	std::map<double, std::map<unsigned int, double>, Compare_double_or_complex_number > energy_resolved_density_output;
	// Integrate over all local momenta using integrate_diagonal
	std::vector<std::complex<double> > result(chemical_potential_map.size());
	std::vector<std::complex<double> >* result_pointer = &result;
	bp_integrate_over_momenta(ret_Green, momenta_lecd_map_pointer, result_pointer);

	//lcd_out_file << "\n\n-----------------------TOTAL CURRENT-----------------------\n";
	//for(unsigned int i=0; i<result.size(); ++i)
	//{
	//lcd_out_file << i << ":\t" << result[i] << "\n";
	//}

	// Assign to output
	*local_energy_current_density = result;

	//lcd_out_file.close();
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_contact_energy_relaxed_local_current(std::map<std::string, Simulation*> nonvirtual_contacts, const Propagator* ret_Green,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map,
	std::vector<std::complex<double> >*& contact_local_current_density)
{
	// Calculates the total (integrated over all momenta) local current at each virtual contact (for Buettiker probes)
	// F_p^(i) = int(dE, j_p[mu_1^(i),...](E)) for all p in P
	// Under development, Kyle Aitken: kaitken17@gmail.com

	tic_toc_name = options.get_option("tic_toc_name", get_name());
	std::string tic_toc_prefix = "Self_energy(\"" + tic_toc_name + "\")::bp_energy_relaxed_local_current ";
	NemoUtils::tic(tic_toc_prefix);
	std::string prefix = "Self_energy(" + this->get_name() + ")::bp_energy_relaxed_local_current: ";

	// Output - for test purposes
	std::string filename = "bp_txt_lcd_all_output.txt";
	//std::ofstream lcd_out_file;
	//lcd_out_file.open(output_collector.get_file_path(filename,
	//                  "File for testing, Contains information of the local current between all BP and contacts as well as the chemical potentials",NemoFileSystem::DEBUG).c_str());
	//lcd_out_file << "BP Local Current Density Status\n\n";

	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*> momenta_lcd_map;
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>* momenta_lcd_map_pointer = &momenta_lcd_map;

	// Loop over all momenta tuples and create a pointer to a complex for each value, but only point to NULL for the time being
	std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::const_iterator it = ret_Green->propagator_map.begin();
	for (; it != ret_Green->propagator_map.end(); ++it)
	{
		momenta_lcd_map[it->first] = NULL;
	}

	// Loop over all momenta
	std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>::iterator momenta_it = momenta_lcd_map.begin();
	for (; momenta_it != momenta_lcd_map.end(); ++momenta_it)
	{
		// DEBUG: Prints information for mesh point
		//lcd_out_file << "\n\n-----------------------MOMENTA POINT (";
		//for(unsigned int i=0; i<momenta_it->first.size(); ++i) // Index for each momenta
		//{
		//  lcd_out_file << momenta_it->first[i].get_idx() << " ";
		//}
		//lcd_out_file << ")-----------------------\n";

		// Initializes vector and assigns results
		momenta_it->second = new std::vector<std::complex<double> >(nonvirtual_contacts.size());
		if (options.get_option("electron_temperature_driven", false) && particle_type_is_Fermion)
			bp_calculate_contact_temperature_local_current_density(momenta_it->first, nonvirtual_contacts, bp_trans_map, nvc_trans_map, momenta_it->second);
		else
			bp_calculate_contact_local_current_density(momenta_it->first, nonvirtual_contacts, bp_trans_map, nvc_trans_map, momenta_it->second);
		//NOTE: Add NEMO_ASSERT here

		// DEBUG: Prints result
		//for(unsigned int i=0;i<momenta_it->second->size();++i)
		//{
		//  lcd_out_file << i << ":\t" << (*(momenta_it->second))[i] << "\n";
		//}
	}

	std::map<double, std::vector<std::complex<double> > > energy_resolved_data;
	std::map<double, std::vector<std::complex<double> > >* energy_resolved_data_pointer = &energy_resolved_data;
	if (options.get_option("output_energy_resolved_current", false))
	{
		integrate_vector_for_energy_resolved(momenta_lcd_map_pointer, energy_resolved_data_pointer);

		std::map<double, std::vector<std::complex<double> > >::iterator it = energy_resolved_data.begin();
		for (; it != energy_resolved_data.end(); ++it)
		{
			std::vector<std::complex<double> >::iterator vec_it = it->second.begin();

			int my_local_rank = -1;
			MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);

			if (my_local_rank == 0)
			{
				for (; vec_it != it->second.end(); ++vec_it)
					cerr << "E[eV] " << it->first << " value " << *vec_it << " \n";
			}
		}
	}


	// Integrate over all local momenta using integrate_diagonal
	std::vector<std::complex<double> > result(nonvirtual_contacts.size());
	std::vector<std::complex<double> >* result_pointer = &result;
	bp_integrate_over_momenta(ret_Green, momenta_lcd_map_pointer, result_pointer);

	//lcd_out_file << "\n\n-----------------------TOTAL CURRENT-----------------------\n";
	//for(unsigned int i=0; i<result.size(); ++i)
	//{
	//lcd_out_file << i << ":\t" << result[i] << "\n";
	//}

	// Assign to output
	*contact_local_current_density = result;

	//lcd_out_file.close();
	NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::set_chemical_potential(int indx, double new_cp)
{
  chemical_potential_map.at(indx) = new_cp;
}

void Self_energy::bp_calculate_transmission(const Propagator* ret_Green, std::map<std::string, Simulation*> nonvirtual_contacts,
    std::map<std::string, Simulation*> bp_retarded_self_sources,
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& bp_trans_map,
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& nvc_trans_map)
{
  // Under development, Kyle Aitken: kaitken17@gmail.com
  // This function calculates the transmission between all buettiker probes and other contacts for ALL momenta
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::bp_calculate_transmission ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::bp_calculate_transmission: ";
  msg<<prefix<<"\n";

  // Output - for test purposes
  //std::string filename = "bp_txt_trans_output.txt";

  //std::ofstream trans_out_file;
  //trans_out_file.open(output_collector.get_file_path(filename,
                      //"File for testing, This file contains the Self energy real and imaginary part for each BP and the transmission between the BP",NemoFileSystem::DEBUG).c_str());
  //trans_out_file << "BP Transmission Status\n\n";

  // DOF map used to create bp_trans_map
  //const DOFmap* defining_DOFmap=&(Hamilton_Constructor->get_dof_map());

  // Optional: Read in sparsity pattern of transmission map. Sparse if few Buettiker probes, dense if many.
  // If undefined, assumes to be sparse
  std::string bp_trans_map_storage = std::string("");
  bp_trans_map_storage = options.get_option("bp_trans_map_storage",std::string("sparse"));

  //NOTE: check that the self-energies and the retarded Green's function are ready - to be covered by get_data calls...

  // Loop over all momenta
  //NOTE: These transmissions maps are bigger than they need to be, they are the size up to the maximum Buettiker probe index and are
  // sparse. They could be made the size of the number of Buettiker probes and dense.
  Propagator::PropagatorMap::const_iterator green_matrix_iterator;
  green_matrix_iterator = ret_Green->propagator_map.begin();
  for(; green_matrix_iterator!=ret_Green->propagator_map.end(); ++green_matrix_iterator)
  {

    // Iterator to corresponding momenta in bp_trans_map
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_trans_it = bp_trans_map->find(green_matrix_iterator->first);
    // Iterator to corresponding momenta in bp_trans_map
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_trans_it = nvc_trans_map->find(green_matrix_iterator->first);

    // Variables to determine process rows
    // compare to direct_iterative_leads
    std::vector<int> local_rows; // rows on process
    std::vector<int> local_nonzeros; // number of local_nonzeros for corresponding row
    std::vector<int> nonlocal_nonzeros; // number of nonlocal_nonzeros for corresponding row

    //defining_DOFmap->calculate_non_zeros(&local_nonzeros, &nonlocal_nonzeros );
    //defining_DOFmap->get_local_row_indexes(&local_rows);
    //unsigned int number_of_rows = defining_DOFmap->get_global_dof_number();
    //unsigned int local_number_of_rows =  defining_DOFmap->get_number_of_dofs();

    //DEBUG: Temporary serial override
    std::map<unsigned int, double>::const_iterator temp_it = chemical_potential_map.end();
    --temp_it;
    // Sets number of rows to maximum BP index
    unsigned int number_of_rows = temp_it->first + 1;
    //trans_out_file << "Number of rows:\t" << number_of_rows << "\n";
    unsigned int local_number_of_rows = number_of_rows;
    local_rows = std::vector<int>(number_of_rows, 0);
    local_nonzeros = std::vector<int>(number_of_rows, chemical_potential_map.size());
    nonlocal_nonzeros = std::vector<int>(number_of_rows, 0);
    for(unsigned int i=0; i<number_of_rows; ++i)
    {
      local_rows[i] = i;
    }

    NEMO_ASSERT(number_of_rows>0, prefix + "seem to have 0 DOFs (empty matrix). There is something wrong. Aborting.");

    // Initializes bp_trans_map matrix corresponding to momenta point
    bp_trans_it->second = new PetscMatrixParallelComplex (number_of_rows, number_of_rows, get_simulation_domain()->get_communicator());
    bp_trans_it->second->set_num_owned_rows(local_number_of_rows);
    // Initializes nvc_trans_map matrix corresponding to momenta point
    nvc_trans_it->second = new PetscMatrixParallelComplex (number_of_rows, number_of_rows, get_simulation_domain()->get_communicator());
    nvc_trans_it->second->set_num_owned_rows(local_number_of_rows);

    // For the bp_trans_map, sets number of non-zeros based on storage type
    if(bp_trans_map_storage==std::string("dense"))
    {
      bp_trans_it->second->consider_as_full();
    }
    else if(bp_trans_map_storage==std::string("sparse"))
    {
      for (unsigned int i = 0; i < local_number_of_rows; i++)
        bp_trans_it->second->set_num_nonzeros(local_rows[i],local_nonzeros[i],nonlocal_nonzeros[i]);
    }
    else
      throw std::invalid_argument(prefix + "unknown storage format\n");

    // For the nvc_trans_map, automatically sets to sparse matrix.
    for (unsigned int i = 0; i < local_number_of_rows; i++)
      nvc_trans_it->second->set_num_nonzeros(local_rows[i],local_nonzeros[i],nonlocal_nonzeros[i]);

    // Allocates memory for the given matrices
    bp_trans_it->second->allocate_memory();
    nvc_trans_it->second->allocate_memory();

    bp_trans_it->second->set_to_zero();
    bp_trans_it->second->assemble();
    nvc_trans_it->second->set_to_zero();
    nvc_trans_it->second->assemble();

    // Retrieves the Buettiker probe retarded self energy for the particular momenta point
    // Iterate over local DOFs
    PetscMatrixParallelComplex* bp_retarded_self;
    std::map<std::string, Simulation*>::const_iterator bp_source_it = bp_retarded_self_sources.begin();
    bp_source_it->second->get_data(bp_source_it->first,&(green_matrix_iterator->first),bp_retarded_self);

    //bp_retarded_self->save_to_matlab_file(std::string("debug_trans_whole_retarded_self.m"));

    // DEBUG: Prints information for mesh point
    //trans_out_file << "\n\n-----------------------MOMENTA POINT (";
    //for(unsigned int i=0; i<green_matrix_iterator->first.size(); ++i) // Index for each momenta
    //{
    //  trans_out_file << green_matrix_iterator->first[i].get_idx() << " ";
    //}
    //trans_out_file << ")-----------------------\n";

    //std::map<unsigned int, double>::const_iterator bp_it1 = chemical_potential_map.begin();
    // Iterates through local_rows corresponding to those BP DOFs on process
    for(unsigned int i=0; i<local_rows.size(); ++i)
    {
      // Sees if row corresponds to a Buettiker probe
      if(chemical_potential_map.find(local_rows[i])!=chemical_potential_map.end())
      {
        //trans_out_file << "\nBuettiker Probe at row: " << local_rows[i] << "\n  ---Inter BP Transmissions---\n";
        // " \t(" << real(self1_value) << ", " << imag(self1_value) << ")\n  ---Inter-BP Transmissions---\n";

        // Timing analysis of assembling matrices
        std::string tic_toc_prefix_int1 = "Self_energy(\""+tic_toc_name+"\")::bp_calculate_transmission_self_assemble_1";
        NemoUtils::tic(tic_toc_prefix_int1);
        std::map<unsigned short, unsigned short>::const_iterator dof_it1 = dof_to_bp_map.begin();
        std::set<unsigned short> self_dofs1;
        for(; dof_it1!=dof_to_bp_map.end(); ++dof_it1)
        {
          if(dof_it1->second == local_rows[i]) // If the DOF belongs to the current BP
            self_dofs1.insert(dof_it1->first);
        }
        int temp_block_size = dof_to_bp_map.size()/local_rows.size();
        PetscMatrixParallelComplex* self_matrix1_block = new PetscMatrixParallelComplex (temp_block_size,temp_block_size,get_simulation_domain()->get_communicator());
        self_matrix1_block->allocate_memory();
        self_matrix1_block->set_to_zero();


        // 1. Creates Matrix self_matrix_1
        int start_own_rows1;
        int end_own_rows1;
        green_matrix_iterator->second->get_ownership_range(start_own_rows1,end_own_rows1);

        PetscMatrixParallelComplex* self_matrix1 = new PetscMatrixParallelComplex (end_own_rows1,end_own_rows1,get_simulation_domain()->get_communicator());

        // 2. Defines the number of nonzeros (local and nonlocal) for each local row. Local rows same as Green's matrix
        self_matrix1->set_num_owned_rows(green_matrix_iterator->second->get_num_owned_rows());
        // NOTE: Need some future check here if any of the DOFs are not on the process's owned rows

        for(int j=start_own_rows1; j<end_own_rows1; j++)
        {
          if(self_dofs1.find(j)!=self_dofs1.end()) // If the local row corresponds to a DOF of the particular self-energy
          {
            self_matrix1->set_num_nonzeros(j,1,0); // One nonzero
            //trans_out_file << "Found: \t" << j << "\tsetting single nonzero\n";
          }
          else
          {
            self_matrix1->set_num_nonzeros(j,0,0); // All zeros
            //trans_out_file << " Not Found:\t" << j << "\tsetting all zeros\n";
          }
        }
        // 3. Allocates memory initializes matrix elements to zero
        self_matrix1->allocate_memory();
        self_matrix1->set_to_zero();
        // 4. Sets elements of self_matrix_1 and assembles
        std::set<unsigned short>::const_iterator set_it1 = self_dofs1.begin();
        for(int tt = 0; set_it1!=self_dofs1.end(); ++set_it1, tt++)
        {
          self_matrix1_block->set(tt, tt, bp_retarded_self->get(*set_it1, *set_it1));
          self_matrix1->set(*set_it1, *set_it1, bp_retarded_self->get(*set_it1, *set_it1));
        }
        self_matrix1_block->assemble();
        self_matrix1->assemble();

        //self_matrix1->save_to_matlab_file(std::string("self_matrix1.m"));
        NemoUtils::toc(tic_toc_prefix_int1);

        std::map<unsigned int, double>::const_iterator bp_it2 = chemical_potential_map.begin();
        // Iterates over all Buettiker probes contacts (equivalent to all columns)
        // Determines if the calculation should be computed using the full NEGF formula or the shortcut for single valued self-energies
        if (options.get_option("bp_full_transmission_calculation", true))
        {
          for(; bp_it2!=chemical_potential_map.end(); ++bp_it2) // Loop over all Buettiker probes
          {
            if(local_rows[i]!=(int)bp_it2->first)// && local_rows[i]-(int)bp_it2->first<12 && local_rows[i]-(int)bp_it2->first>-12) // Does not compute transmission between a probe and itself
            {
              // DEBUG: Prints BP header
              //trans_out_file << "  BP" << local_rows[i] << " to BP" << bp_it2->first << " - "; //<< " (" << real(self2_value) << ", " << imag(self2_value) << ") - ";

              // Timing analysis of assembling matrices
              std::string tic_toc_prefix_int2 = "Self_energy(\""+tic_toc_name+"\")::bp_calculate_transmission_self_assemble_2";
              NemoUtils::tic(tic_toc_prefix_int2);

              std::map<unsigned short, unsigned short>::const_iterator dof_it2 = dof_to_bp_map.begin();
              std::set<unsigned short> self_dofs2;
              for(; dof_it2!=dof_to_bp_map.end(); ++dof_it2)
              {
                if(dof_it2->second == bp_it2->first) // If the DOF belongs to the current BP
                {
                  self_dofs2.insert(dof_it2->first);
                  //trans_out_file << dof_it2->first << ", ";
                }
              }
              PetscMatrixParallelComplex* self_matrix2_block = new PetscMatrixParallelComplex (temp_block_size,temp_block_size,get_simulation_domain()->get_communicator());
              self_matrix2_block->allocate_memory();
              self_matrix2_block->set_to_zero();

              PetscMatrixParallelComplex* ret_green_block = new PetscMatrixParallelComplex (temp_block_size,temp_block_size,get_simulation_domain()->get_communicator());
              ret_green_block->allocate_memory();
              ret_green_block->set_to_zero();

              std::set<unsigned short>::const_iterator set_it2 = self_dofs2.begin();
              for(int tt = 0; set_it2!=self_dofs2.end(); ++set_it2, tt++)
              {
                std::set<unsigned short>::const_iterator set_it11 = self_dofs1.begin();
                self_matrix2_block->set(tt, tt, bp_retarded_self->get(*set_it2, *set_it2));

                for(int ttt = 0; set_it11!=self_dofs1.end(); ++set_it11, ttt++)
                {
                  ret_green_block->set(tt, ttt, green_matrix_iterator->second->get(*set_it2, *set_it11));
        
                }
		      }
              self_matrix2_block->assemble();
              ret_green_block->assemble();


              // Calculates transmission and stores to result
              std::complex<double> result(std::complex<double>(0.0,0.0));
              bp_calculate_transmission_momenta(self_matrix1_block, self_matrix2_block, ret_green_block, result);
              /*
              // 1. Creates Matrix self_matrix_2
              PetscMatrixParallelComplex* self_matrix2 = new PetscMatrixParallelComplex (*(green_matrix_iterator->second));
              // Iterates over BP DOF map to see which DOFs belong to BP, adds corresponding self-eneriges
              //trans_out_file << "DOFs: ";
              std::map<unsigned short, unsigned short>::const_iterator dof_it2 = dof_to_bp_map.begin();
              std::set<unsigned short> self_dofs2;
              for(; dof_it2!=dof_to_bp_map.end(); ++dof_it2)
              {
                if(dof_it2->second == bp_it2->first) // If the DOF belongs to the current BP
                {
                  self_dofs2.insert(dof_it2->first);
                  //trans_out_file << dof_it2->first << ", ";
                }
              }
              //trans_out_file << "\t";
              // 2. Defines the number of nonzeros (local and nonlocal) for each local row. Local rows same as Green's matrix
              self_matrix2->set_num_owned_rows(green_matrix_iterator->second->get_num_owned_rows());
              // NOTE: Need some future check here if any of the DOFs are not on the process's owned rows
              int start_own_rows2;
              int end_own_rows2;
              green_matrix_iterator->second->get_ownership_range(start_own_rows2, end_own_rows2);
              for(int j=start_own_rows2; j<end_own_rows2; j++)
              {
                if(self_dofs2.find(j)!=self_dofs2.end()) // If the local row corresponds to a DOF of the particular self-energy
                  self_matrix2->set_num_nonzeros(j,1,0); // One local nonzero, zero nonlocal nonzeros
                else
                  self_matrix2->set_num_nonzeros(j,0,0); // All zeros (local and nonlocal)
              }
              // 3. Allocates memory initializes matrix elements to zero
              self_matrix2->allocate_memory();
              self_matrix2->set_to_zero();
              // 4. Sets elements of self_matrix_2 and assembles
              std::set<unsigned short>::const_iterator set_it2 = self_dofs2.begin();
              for(; set_it2!=self_dofs2.end(); ++set_it2)
              {
                self_matrix2->set(*set_it2, *set_it2, bp_retarded_self->get(*set_it2, *set_it2));
              }
              self_matrix2->assemble();
							//self_matrix2->save_to_matlab_file(std::string("self_matrix2.m"));
              NemoUtils::toc(tic_toc_prefix_int2);

              // Calculates transmission and stores to result
              std::complex<double> result(std::complex<double>(0.0,0.0));
              bp_calculate_transmission_momenta(self_matrix1, self_matrix2, green_matrix_iterator->second, result);
*/
              // DEBUG: Prints result
              //trans_out_file << real(result) << "\n";

              // Stores the trace to location in bp_trans_map
              bp_trans_it->second->set(local_rows[i], bp_it2->first, result);

			  delete self_matrix2_block;
			  delete ret_green_block;
            }
            else // Transmission betwen a probe and itself set to zero (so it does not disrupt sum later on)
            {
              // bp_trans_it->second->set(local_rows[i], bp_it2->first, 0);
              // DEBUG: Prints result
              //trans_out_file << "  BP" << local_rows[i] << " to BP" << bp_it2->first << " - Diagonal Case\n";
            }
          }
          std::map<std::string, Simulation*>::const_iterator nvc_it = nonvirtual_contacts.begin();
          for(unsigned int j=0; j<nonvirtual_contacts.size(); ++j, ++nvc_it)
          {
            //self_matrix1->set(local_rows[i], local_rows[i], bp_retarded_self->get(local_rows[i], local_rows[i]));
            //self_matrix1->assemble();
            PetscMatrixParallelComplex* self_matrix2;

            // Gets self energy of non-virtual contact
            nvc_it->second->get_data(nvc_it->first,&(green_matrix_iterator->first),self_matrix2,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
            self_matrix2->assemble();
            // self_matrix2->save_to_matlab_file(std::string("debug_pre_trans_self2.m"));
            // Calculates transmission and stores to result
            std::complex<double> result(std::complex<double>(0.0,0.0));
            bp_calculate_transmission_momenta(self_matrix1, self_matrix2, green_matrix_iterator->second, result);

            //DEBUG: Prints the result
            //trans_out_file << "    Total Transmission:\t" << real(result) << "\n";

            // Stores the trace to location in nvc_trans_map
            //nvc_trans_it->second->set(local_rows[i], temporary_nvc_locations[j], result);
            nvc_trans_it->second->set(local_rows[i], j, result);
        }
        }
        else //Analytical caclulation for single-element self-energies
        {
          std::complex<double> self1_value = bp_retarded_self->get(local_rows[i], local_rows[i]);

          for(; bp_it2!=chemical_potential_map.end(); ++bp_it2) // Loop over all Buettiker probes
          {
            if(local_rows[i]!=(int)bp_it2->first)// && local_rows[i]-(int)bp_it2->first<12 && local_rows[i]-(int)bp_it2->first>-12) // Does not compute transmission between a probe and itself
            {
              std::complex<double> self2_value = bp_retarded_self->get(bp_it2->first, bp_it2->first);

              std::complex<double> trans(std::complex<double>(1.0,0.0));

              // DEBUG: Prints BP header
              //trans_out_file << "  BP" << local_rows[i] << " to BP" << bp_it2->first << " (" << real(self2_value) << ", " << imag(self2_value) << ") - ";

              // Equivalent to Tr[gamma1*G^R*gamma2*G^A]
              double gamma1 = -2 * std::imag(self1_value);
              double gamma2 = -2 * std::imag(self2_value);
              double green_value = std::abs(green_matrix_iterator->second->get(local_rows[i], bp_it2->first));

              trans *= gamma1 * gamma2 * green_value * green_value;

              // DEBUG: Prints result
              //trans_out_file << trans << "\n";

              // Stores the trace to location in bp_trans_map
              bp_trans_it->second->set(local_rows[i], bp_it2->first, trans);
            }
            else // Transmission betwen a probe and itself set to zero (so it does not disrupt sum later on)
            {
              bp_trans_it->second->set(local_rows[i], bp_it2->first, 0);
              // DEBUG: Prints result
              //trans_out_file << "  BP" << local_rows[i] << " to BP" << bp_it2->first << " - Diagonal Case\n";
            }
          }
        std::map<std::string, Simulation*>::const_iterator nvc_it = nonvirtual_contacts.begin();
        for(unsigned int j=0; j<nonvirtual_contacts.size(); ++j, ++nvc_it)
        {
          self_matrix1->set(local_rows[i], local_rows[i], bp_retarded_self->get(local_rows[i], local_rows[i]));
          self_matrix1->assemble();
          PetscMatrixParallelComplex* self_matrix2;

          // Gets self energy of non-virtual contact
          nvc_it->second->get_data(nvc_it->first,&(green_matrix_iterator->first),self_matrix2,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
          self_matrix2->assemble();
          // self_matrix2->save_to_matlab_file(std::string("debug_pre_trans_self2.m"));

          // Calculates transmission and stores to result
          std::complex<double> result(std::complex<double>(0.0,0.0));
          bp_calculate_transmission_momenta(self_matrix1, self_matrix2, green_matrix_iterator->second, result);

          nvc_trans_it->second->set(local_rows[i], j, result);
        }
        }

        delete self_matrix1;
        delete self_matrix1_block;
      }
      else
      {
        //trans_out_file << "No Buettiker Probe found at row: " << local_rows[i] <<"\n";
      }
    }
    // Reassembles matrix after sets.
    bp_trans_it->second->assemble();
    nvc_trans_it->second->assemble();
    //bp_trans_it->second->save_to_matlab_file(std::string("bp_trans_it.m"));
    //nvc_trans_it->second->save_to_matlab_file(std::string("nvc_trans_it.m"));
  }

  //trans_out_file.close();
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_calculate_contact_transmission(const Propagator* ret_Green, std::map<std::string, Simulation*> nonvirtual_contacts,
    std::map<std::string, Simulation*> bp_retarded_self_sources,
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& contact_bp_trans_map,
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>*& contact_nvc_trans_map)
{
  // Under development,  
  // This function calculates the transmission between all buettiker probes and other contacts for ALL momenta
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::bp_calculate_transmission ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::bp_calculate_transmission: ";
  msg<<prefix<<"\n";

  // Output - for test purposes
  //std::string filename = "bp_txt_trans_output.txt";

  //std::ofstream trans_out_file;
  //trans_out_file.open(output_collector.get_file_path(filename,
                      //"File for testing, This file contains the Self energy real and imaginary part for each BP and the transmission between the BP",NemoFileSystem::DEBUG).c_str());
  //trans_out_file << "BP Transmission Status\n\n";

  // DOF map used to create bp_trans_map
  //const DOFmap* defining_DOFmap=&(Hamilton_Constructor->get_dof_map());

  // Optional: Read in sparsity pattern of transmission map. Sparse if few Buettiker probes, dense if many.
  // If undefined, assumes to be sparse
  std::string bp_trans_map_storage = std::string("");
  bp_trans_map_storage = options.get_option("bp_trans_map_storage",std::string("sparse"));

  //NOTE: check that the self-energies and the retarded Green's function are ready - to be covered by get_data calls...

  // Loop over all momenta
  //NOTE: These transmissions maps are bigger than they need to be, they are the size up to the maximum Buettiker probe index and are
  // sparse. They could be made the size of the number of Buettiker probes and dense.
  Propagator::PropagatorMap::const_iterator green_matrix_iterator;
  green_matrix_iterator = ret_Green->propagator_map.begin();
  for(; green_matrix_iterator!=ret_Green->propagator_map.end(); ++green_matrix_iterator)
  {

    // Iterator to corresponding momenta in bp_trans_map
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator bp_trans_it = contact_bp_trans_map->find(green_matrix_iterator->first);
    // Iterator to corresponding momenta in bp_trans_map
    std::map<std::vector<NemoMeshPoint>, PetscMatrixParallelComplex*>::iterator nvc_trans_it = contact_nvc_trans_map->find(green_matrix_iterator->first);

    // Variables to determine process rows
    // compare to direct_iterative_leads
    std::vector<int> local_rows; // rows on process
    std::vector<int> local_nonzeros; // number of local_nonzeros for corresponding row
    std::vector<int> nonlocal_nonzeros; // number of nonlocal_nonzeros for corresponding row

    //defining_DOFmap->calculate_non_zeros(&local_nonzeros, &nonlocal_nonzeros );
    //defining_DOFmap->get_local_row_indexes(&local_rows);
    //unsigned int number_of_rows = defining_DOFmap->get_global_dof_number();
    //unsigned int local_number_of_rows =  defining_DOFmap->get_number_of_dofs();

    //DEBUG: Temporary serial override
    std::map<unsigned int, double>::const_iterator temp_it = chemical_potential_map.end();
    --temp_it;
    // Sets number of rows to maximum BP index
    unsigned int number_of_rows = temp_it->first + 1;
    //trans_out_file << "Number of rows:\t" << number_of_rows << "\n";
    unsigned int local_number_of_rows = number_of_rows;
    local_rows = std::vector<int>(number_of_rows, 0);
    local_nonzeros = std::vector<int>(number_of_rows, chemical_potential_map.size());
    nonlocal_nonzeros = std::vector<int>(number_of_rows, 0);
    for(unsigned int i=0; i<number_of_rows; ++i)
    {
      local_rows[i] = i;
    }

    NEMO_ASSERT(number_of_rows>0, prefix + "seem to have 0 DOFs (empty matrix). There is something wrong. Aborting.");

    // Initializes bp_trans_map matrix corresponding to momenta point
    bp_trans_it->second = new PetscMatrixParallelComplex (number_of_rows, number_of_rows, get_simulation_domain()->get_communicator());
    bp_trans_it->second->set_num_owned_rows(local_number_of_rows);
    // Initializes nvc_trans_map matrix corresponding to momenta point
    nvc_trans_it->second = new PetscMatrixParallelComplex (number_of_rows, number_of_rows, get_simulation_domain()->get_communicator());
    nvc_trans_it->second->set_num_owned_rows(local_number_of_rows);

    // For the bp_trans_map, sets number of non-zeros based on storage type
    if(bp_trans_map_storage==std::string("dense"))
    {
      bp_trans_it->second->consider_as_full();
    }
    else if(bp_trans_map_storage==std::string("sparse"))
    {
      for (unsigned int i = 0; i < local_number_of_rows; i++)
        bp_trans_it->second->set_num_nonzeros(local_rows[i],local_nonzeros[i],nonlocal_nonzeros[i]);
    }
    else
      throw std::invalid_argument(prefix + "unknown storage format\n");

    // For the nvc_trans_map, automatically sets to sparse matrix.
    for (unsigned int i = 0; i < local_number_of_rows; i++)
      nvc_trans_it->second->set_num_nonzeros(local_rows[i],local_nonzeros[i],nonlocal_nonzeros[i]);

    // Allocates memory for the given matrices
    bp_trans_it->second->allocate_memory();
    nvc_trans_it->second->allocate_memory();

    bp_trans_it->second->set_to_zero();
    bp_trans_it->second->assemble();
    nvc_trans_it->second->set_to_zero();
    nvc_trans_it->second->assemble();

    // Retrieves the Buettiker probe retarded self energy for the particular momenta point
    // Iterate over local DOFs
    PetscMatrixParallelComplex* bp_retarded_self;
    std::map<std::string, Simulation*>::const_iterator bp_source_it = bp_retarded_self_sources.begin();
    bp_source_it->second->get_data(bp_source_it->first,&(green_matrix_iterator->first),bp_retarded_self);

    //bp_retarded_self->save_to_matlab_file(std::string("debug_trans_whole_retarded_self.m"));

    // DEBUG: Prints information for mesh point
    //trans_out_file << "\n\n-----------------------MOMENTA POINT (";
    //for(unsigned int i=0; i<green_matrix_iterator->first.size(); ++i) // Index for each momenta
    //{
    //  trans_out_file << green_matrix_iterator->first[i].get_idx() << " ";
    //}
    //trans_out_file << ")-----------------------\n";

    //std::map<unsigned int, double>::const_iterator bp_it1 = chemical_potential_map.begin();
    // Iterates through local_rows corresponding to those BP DOFs on process
    std::map<std::string, Simulation*>::const_iterator nvc_it = nonvirtual_contacts.begin();
    for(unsigned int i=0; i<nonvirtual_contacts.size(); ++i, ++nvc_it)
    {

      std::string tic_toc_prefix_int1 = "Self_energy(\""+tic_toc_name+"\")::bp_calculate_transmission_self_assemble_1";
      PetscMatrixParallelComplex* self_matrix1;

      // Gets self energy of non-virtual contact
      nvc_it->second->get_data(nvc_it->first,&(green_matrix_iterator->first),self_matrix1,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
       //trans_out_file << "  BP" << local_rows[i] << " to " << nvc_it->first << "\n";

      // Adds the appropriate "constant chemical potential" Buettiker probes to the self-energy
      //std::set<unsigned int> temp_set = nvc_nonzero_diag_map->find(nvc_it->first)->second;
      //std::set<unsigned int>::const_iterator set_it = temp_set.begin();
      //for(; set_it!=temp_set.end(); ++set_it)
      //{
      //  //trans_out_file << "    Contact Value " << *set_it << ":\t" << self_matrix1->get(*set_it, *set_it) << "\n";
      //  //trans_out_file << "    Adding DOF " << *set_it << ":\t" << bp_retarded_self->get(*set_it, *set_it) << "\n";
      //  // Adds the Buettiker probe self-energy value
      //  self_matrix1->add(*set_it, *set_it, bp_retarded_self->get(*set_it, *set_it));
      //  //self_matrix1->set(*set_it, *set_it, self_matrix1->get(*set_it, *set_it) + bp_retarded_self->get(*set_it, *set_it));
      //}
      self_matrix1->assemble();
      //self_matrix1->save_to_matlab_file(std::string("contact_self1.m"));

      std::map<std::string, Simulation*>::const_iterator nvc_it1 = nonvirtual_contacts.begin();
      for(unsigned int j=0; j<nonvirtual_contacts.size(); ++j, ++nvc_it1)
      {
        if (i!=j)
        {
          PetscMatrixParallelComplex* self_matrix2;

          // Gets self energy of non-virtual contact
          nvc_it1->second->get_data(nvc_it1->first,&(green_matrix_iterator->first),self_matrix2,&(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
          self_matrix2->assemble();
          //self_matrix2->save_to_matlab_file(std::string("contact_self2.m"));
          std::complex<double> result(std::complex<double>(0.0,0.0));
          bp_calculate_transmission_momenta(self_matrix2, self_matrix1, green_matrix_iterator->second, result);
          nvc_trans_it->second->set(i, j, result);
          //delete self_matrix2;
        }
      }
      std::map<unsigned int, double>::const_iterator bp_it2 = chemical_potential_map.begin();
      if (options.get_option("bp_full_transmission_calculation", true))
	  {
        for(; bp_it2!=chemical_potential_map.end(); ++bp_it2) // Loop over all Buettiker probes
        {
          // 1. Creates Matrix self_matrix_2
          PetscMatrixParallelComplex* self_matrix4 = new PetscMatrixParallelComplex (*(green_matrix_iterator->second));
          // Iterates over BP DOF map to see which DOFs belong to BP, adds corresponding self-eneriges
          //trans_out_file << "DOFs: ";
          std::map<unsigned short, unsigned short>::const_iterator dof_it2 = dof_to_bp_map.begin();
          std::set<unsigned short> self_dofs2;
          for(; dof_it2!=dof_to_bp_map.end(); ++dof_it2)
          {
            if(dof_it2->second == bp_it2->first) // If the DOF belongs to the current BP
            {
              self_dofs2.insert(dof_it2->first);
              //trans_out_file << dof_it2->first << ", ";
            }
          }
          // 2. Defines the number of nonzeros (local and nonlocal) for each local row. Local rows same as Green's matrix
          self_matrix4->set_num_owned_rows(green_matrix_iterator->second->get_num_owned_rows());
          // NOTE: Need some future check here if any of the DOFs are not on the process's owned rows
          int start_own_rows2;
          int end_own_rows2;
          green_matrix_iterator->second->get_ownership_range(start_own_rows2, end_own_rows2);
          for(int j=start_own_rows2; j<end_own_rows2; j++)
          {
            if(self_dofs2.find(j)!=self_dofs2.end()) // If the local row corresponds to a DOF of the particular self-energy
              self_matrix4->set_num_nonzeros(j,1,0); // One local nonzero, zero nonlocal nonzeros
            else
              self_matrix4->set_num_nonzeros(j,0,0); // All zeros (local and nonlocal)
          }
          // 3. Allocates memory initializes matrix elements to zero
          self_matrix4->allocate_memory();
          self_matrix4->set_to_zero();
          // 4. Sets elements of self_matrix_2 and assembles
          std::set<unsigned short>::const_iterator set_it2 = self_dofs2.begin();
          for(; set_it2!=self_dofs2.end(); ++set_it2)
          {
            self_matrix4->set(*set_it2, *set_it2, bp_retarded_self->get(*set_it2, *set_it2));
          }
          self_matrix4->assemble();

          // Calculates transmission and stores to result
          std::complex<double> result(std::complex<double>(0.0,0.0));
          bp_calculate_transmission_momenta(self_matrix1, self_matrix4, green_matrix_iterator->second, result);
          // Stores the trace to location in bp_trans_map
          bp_trans_it->second->set(bp_it2->first,  i,result);
          delete self_matrix4;
		}
      }
	  else
	  {
      for(unsigned int k=0; k<local_rows.size(); ++k)
      {
        // Sees if row corresponds to a Buettiker probe
        if(chemical_potential_map.find(local_rows[k])!=chemical_potential_map.end())
        {
          PetscMatrixParallelComplex* self_matrix3 = new PetscMatrixParallelComplex (*(green_matrix_iterator->second));
          // Iterates over BP DOF map to see which DOFs belong to BP
          std::map<unsigned short, unsigned short>::const_iterator dof_it1 = dof_to_bp_map.begin();
          std::set<unsigned short> self_dofs1;
          for(; dof_it1!=dof_to_bp_map.end(); ++dof_it1)
          {
            if(dof_it1->second == local_rows[k]) // If the DOF belongs to the current BP
            self_dofs1.insert(dof_it1->first);
          }
          // 2. Defines the number of nonzeros (local and nonlocal) for each local row. Local rows same as Green's matrix
          self_matrix3->set_num_owned_rows(green_matrix_iterator->second->get_num_owned_rows());
          // NOTE: Need some future check here if any of the DOFs are not on the process's owned rows
          int start_own_rows1;
          int end_own_rows1;
          green_matrix_iterator->second->get_ownership_range(start_own_rows1,end_own_rows1);
          for(int j=start_own_rows1; j<end_own_rows1; j++)
          {
            if(self_dofs1.find(j)!=self_dofs1.end()) // If the local row corresponds to a DOF of the particular self-energy
            {
              self_matrix3->set_num_nonzeros(j,1,0); // One nonzero
              //trans_out_file << "Found: \t" << j << "\tsetting single nonzero\n";
            }
            else
            {
              self_matrix3->set_num_nonzeros(j,0,0); // All zeros
              //trans_out_file << " Not Found:\t" << j << "\tsetting all zeros\n";
            }
          }
          // 3. Allocates memory initializes matrix elements to zero
          self_matrix3->allocate_memory();
          self_matrix3->set_to_zero();
          self_matrix3->set(local_rows[k], local_rows[k], bp_retarded_self->get(local_rows[k], local_rows[k]));
          self_matrix3->assemble();
		    //self_matrix3->save_to_matlab_file(std::string("self_matrix3.m"));
          std::complex<double> result(std::complex<double>(0.0,0.0));
          bp_calculate_transmission_momenta(self_matrix1, self_matrix3, green_matrix_iterator->second, result);
          delete self_matrix3;
          bp_trans_it->second->set(local_rows[k], i, result);
         }
       }
	  }
      //delete self_matrix1;
    }

    // Reassembles matrix after sets.
    bp_trans_it->second->assemble();
    nvc_trans_it->second->assemble();
    //bp_trans_it->second->save_to_matlab_file(std::string("contact_bp_trans_it.m"));
    //nvc_trans_it->second->save_to_matlab_file(std::string("contact_nvc_trans_it.m"));

  }
  //trans_out_file.close();
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_calculate_transmission_momenta(PetscMatrixParallelComplex* self_matrix1, PetscMatrixParallelComplex* self_matrix2,
    PetscMatrixParallelComplex* green_matrix, std::complex<double>& result)
{
  // Under development, Kyle Aitken: kaitken17@gmail.com
  // This function calculates the transmission between self_matrix1
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::bp_calculate_transmission_momenta ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::bp_calculate_transmission_momenta: ";
  msg<<prefix<<"\n";

  //transmission = tr(gamma_p*Gr*gama_q*Ga); Gamma=i*(sigma_R-sigma_A)
  PetscMatrixParallelComplex* gamma1 = new PetscMatrixParallelComplex (*self_matrix1);
  PetscMatrixParallelComplex* gamma2 = new PetscMatrixParallelComplex (*self_matrix2);
  PetscMatrixParallelComplex* temp   = new PetscMatrixParallelComplex (self_matrix1->get_num_cols(), self_matrix1->get_num_rows(),
      self_matrix1->get_communicator());

  //gamma1
  self_matrix1->hermitian_transpose_matrix(*temp,MAT_INITIAL_MATRIX);
  *temp *= std::complex<double>(-1.0,0.0);
  gamma1->add_matrix(*temp,DIFFERENT_NONZERO_PATTERN);
  *gamma1 *= std::complex<double>(0.0,1.0);
  delete temp;

  //gamma1->save_to_matlab_file(std::string("debug_trans_gamma1.m"));

  //gamma2
  temp = new PetscMatrixParallelComplex (self_matrix2->get_num_cols(), self_matrix2->get_num_rows(), self_matrix2->get_communicator() );
  self_matrix2->hermitian_transpose_matrix(*temp,MAT_INITIAL_MATRIX);
  *temp *= std::complex<double>(-1.0,0.0);
  gamma2->add_matrix(*temp,DIFFERENT_NONZERO_PATTERN);
  *gamma2 *= std::complex<double>(0.0,1.0);
  delete temp;

  //gamma2->save_to_matlab_file(std::string("debug_trans_gamma2.m"));

  //temp=gamma_p*Gr*gama_q*Ga
  temp = new PetscMatrixParallelComplex (green_matrix->get_num_cols(), green_matrix->get_num_rows(), green_matrix->get_communicator());

  green_matrix->hermitian_transpose_matrix(*temp,MAT_INITIAL_MATRIX);
  PetscMatrixParallelComplex* temp2=NULL;
  PetscMatrixParallelComplex::mult(*gamma2,*temp,&temp2);//GR_inv1=VL'*(E-H)
  delete temp;
  temp=NULL;
  PetscMatrixParallelComplex::mult(*green_matrix,*temp2,&temp);
  delete temp2;
  temp2=NULL;
  PetscMatrixParallelComplex::mult(*gamma1,*temp,&temp2);
  delete gamma1;
  delete gamma2;
  delete temp;

  //temp2->save_to_matlab_file(std::string("debug_trans_temp2.m"));

  // Get the trace of temp2
  std::complex<double> trace(std::complex<double>(0.0,0.0));
  trace=temp2->get_trace();

  delete temp2;

  result = trace;

  // DEBUG: Checks to see if this is being called with the correct self-energies and retarded Green's functions
  /*self_matrix1->save_to_matlab_file(std::string("debug_trans_self1.m"));
  self_matrix2->save_to_matlab_file(std::string("debug_trans_self2.m"));
  green_matrix->save_to_matlab_file(std::string("debug_trans_R_green.m"));*/
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::bp_integrate_over_momenta(const Propagator* ret_Green,
    std::map<std::vector<NemoMeshPoint>, std::vector<std::complex<double> >*>*& input_vector,
    std::vector<std::complex<double> >*& result)
{
  // This function integrates the input vector (any complex vector) over all momenta, but unlike
  // Propagation::integrate_diagonal it gets the momenta weights from an existing propagator, eliminating the need for the
  // diagonal to be an actual diagonal of a matrix belonging to a propagator
  // Under development, Kyle Aitken: kaitken17@gmail.com
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::bp_integrate_over_momenta ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy::(\""+this->get_name()+"\")::bp_integrate_over_momenta: ";

  //lcd_out_file << "\n\n===========================INTEGRATE DIAGONAL===========================\n";

  Propagator::PropagatorMap::const_iterator momentum_c_it=ret_Green->propagator_map.begin();

  // temp_result holds the integrated diagonal

  std::vector<std::complex<double> > temp_result(result->size(), std::complex<double>(0.0,0.0));
  // Throw error if diagonal and result are not of equal size
  NEMO_ASSERT(temp_result.size() == input_vector->find(momentum_c_it->first)->second->size(),prefix+"mismatch of result and diagonal size\n");

  //1. Loop over all momenta
  for(; momentum_c_it!=ret_Green->propagator_map.end(); ++momentum_c_it)
  {
    //lcd_out_file << "\n\n-----------------------MOMENTA POINT (";
    //for(unsigned int i=0; i<momentum_c_it->first.size(); ++i) // Index for each momenta
    //{
      //lcd_out_file << momentum_c_it->first[i].get_idx() << " ";
    //}
    //lcd_out_file << ")-----------------------\n";

    // Temporary pointer to momenta's complex vector
    std::vector<std::complex<double> >* diag_pointer = input_vector->find(momentum_c_it->first)->second;

    bool analytical_momenta=false;
    if(options.check_option("bp_analytical_momenta"))
      analytical_momenta=true;
    // Container for analytical integration, the type of analytical momentum, i.e. 1D or 2D
    std::string momentum_type = options.get_option("bp_analytical_momenta",std::string(""));

    // Gets the integration weight
    double weight = bp_get_integration_weight(ret_Green, momentum_c_it->first, analytical_momenta, momentum_type);

    /*if(!analytical_momenta)
    {*/
    //lcd_out_file << "Weight:\t" << weight << "\n";

    //3. Do the integral - i.e. weighted sum
    for(unsigned int i=0; i<temp_result.size(); i++)
    {
      //lcd_out_file << "  Node:\t" << i << "\tValue:\t" << (*diag_pointer)[i] << "\n";
      temp_result[i] += (*diag_pointer)[i] * weight;
    }
    //}
    // Analytical momenta correction handled directly
    //else
    //{
    //  std::vector<double> effective_mass;
    //  std::vector<unsigned int> dummy_input(1,1);
    //  Hamilton_Constructor->get_data("effective_mass",dummy_input,effective_mass);
    //  //3. Do the integral - i.e. weighted sum
    //  if(momentum_type=="1D")
    //  {
    //    for(unsigned int i=0; i<temp_result.size(); i++)
    //      temp_result[i] += (*diag_pointer)[i] * weight * std::sqrt(effective_mass[i] * NemoPhys::electron_mass);
    //  }
    //  else
    //  {
    //    for(unsigned int i=0; i<temp_result.size(); i++)
    //      temp_result[i] += (*diag_pointer)[i] * weight * effective_mass[i] * NemoPhys::electron_mass;
    //  }
    //}
  }

  //: How does this change?
  // Sum up the results of all other MPI processes
  /*const MPI_Comm& topcomm=Mesh_tree_topdown.begin()->first->get_global_comm();
  MPI_Barrier(topcomm);
  MPI_Allreduce(&(temp_result[0]),&((*pointer_to_result)[0]),temp_result.size(), MPI_DOUBLE_COMPLEX, MPI_SUM ,topcomm);*/

  *result = temp_result;

  NemoUtils::toc(tic_toc_prefix);
}

double Self_energy::bp_get_integration_weight(const Propagator* ret_Green, std::vector<NemoMeshPoint> momenta_tuple, bool /*analytical_momenta*/,
    std::string /*momentum_type*/)
{
  //Determining the integration weight of this momentum vector, i.e. this (E, k1,k2,...)
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::bp_get_integration_weight ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy::(\""+this->get_name()+"\")::bp_get_integration_weight: ";

  double weight=1.0;

  // Loops over all momenta in the tuple (E, k1,k2,...) and gets corresponding integration weight
  for(unsigned int i=0; i<momenta_tuple.size(); i++)
  {
    // Get the constructor of this meshpoint
    std::string momentum_name = ret_Green->momentum_mesh_names[i];
    weight *= find_integration_weight(momentum_name, momenta_tuple, writeable_Propagator);
    //find the momentum name that does not contintegration_weightain "energy"
    if (ret_Green->momentum_mesh_names[i].find("energy") == std::string::npos)
    {
      std::map<std::string, Simulation*>::const_iterator c_it2 = Propagation::Mesh_Constructors.begin();
      c_it2 = Propagation::Mesh_Constructors.find(momentum_name);
      NEMO_ASSERT(c_it2!=Propagation::Mesh_Constructors.end(), prefix+"have not found constructor for momentum mesh \""+momentum_name+"\"\n");
      //check whether the momentum solver is providing a non rectangular mesh
      if(momentum_name.find("momentum_1D")!=std::string::npos)
        weight/=2.0*NemoMath::pi;
      if(momentum_name.find("momentum_2D")!=std::string::npos)
        weight/=4.0*NemoMath::pi*NemoMath::pi;
    }
    /*std::map<std::string, Simulation*>::const_iterator temp_cit = Mesh_Constructors.find(momentum_mesh_name);
    NEMO_ASSERT(temp_cit!=Mesh_Constructors.end(),prefix+"have not found constructor of mesh \""+momentum_mesh_name+"\"\n");
    Simulation* mesh_constructor = temp_cit->second;
    // Ask the constructor for the weighting of this point
    double temp_double=0;
    InputOptions& mesh_options=mesh_constructor->get_reference_to_options();
    if(!mesh_options.get_option(std::string("non_rectangular"),false))
      mesh_constructor->get_data("integration_weight",momenta_tuple[i],temp_double);
    else
      mesh_constructor->get_data("integration_weight",momenta_tuple,momenta_tuple[i],temp_double);
    // Adjust weight accordingly
    weight *= temp_double;*/

    //2.1 Check whether 2D momenta are to be integrated analytically (i.e. assuming parabolic transverse dispersions)
    //if(analytical_momenta)
    //{
    //  double chemical_potential=options.get_option("chemical_potential", 0.0);
    //     double temperature=options.get_option("temperature", 300.0);
    //     int degeneracy_factor=options.get_option("degeneracy_factor", 1);
    //     // Find the energy of this momentum
    //     double energy = PropagationUtilities::read_energy_from_momentum(this,momenta_tuple,ret_Green);

    //     // Find the type of analytical momentum, i.e. 1D or 2D
    //     if(momentum_type==std::string("1D"))
    //     {
    //       // Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation, Ref. Thesis of Mathias Sabathil, Eq. 3.10)
    //       double temp = (chemical_potential-energy)*NemoPhys::elementary_charge/NemoPhys::boltzmann_constant/temperature;
    //       double fermi_integral = NemoMath::fermi_int(-0.5,temp);
    //       //double index_independent_prefactor=0.5*std::sqrt(NemoPhys::boltzmann_constant*temperature/NemoPhys::planck_constant/NemoPhys::planck_constant/NemoMath::pi/2.0); //Junzhe: bug identified
    //       double hbar = NemoPhys::planck_constant/NemoMath::pi/2.0;
    //       double index_independent_prefactor=0.5*std::sqrt(NemoPhys::boltzmann_constant*temperature/hbar/hbar/NemoMath::pi/2.0);
    //       // Modify the weight of this momentum according to the 1D integrated momenta (without the index-dependent effective mass - that is done below)
    //       weight *= index_independent_prefactor*fermi_integral*degeneracy_factor;
    //     }
    //     else if (momentum_type==std::string("2D"))
    //     {
    //       // Calculate the Fermi integral for the analytical momentum integration (based on parabolic dispersion relation)
    //       double temp = (chemical_potential-energy)*NemoPhys::elementary_charge/NemoPhys::boltzmann_constant/temperature;
    //       double fermi_integral = std::log(1.0+std::exp(temp));
    //       double hbar = NemoPhys::planck_constant/NemoMath::pi/2.0;
    //       double index_independent_prefactor=NemoPhys::boltzmann_constant*temperature/hbar/hbar/NemoMath::pi/2.0;
    //       // Modify the weight of this momentum according to the 2D integrated momenta (without the index-dependent effective mass - that is done below)
    //       weight *= index_independent_prefactor*fermi_integral*degeneracy_factor*NemoPhys::nm_in_m*NemoPhys::nm_in_m; //1e-18; //Junzhe: convert from 1/m^2 to 1/nm^2

    //     }
    //     else
    //       throw std::invalid_argument(prefix+"the value of analytical_momenta can be either \"1D\" or \"2D\"\n");
    //   }
  }

  NemoUtils::toc(tic_toc_prefix);

  return weight;
}

void Self_energy::bp_do_solve_lesser_equilibrium(Propagator*& output_Propagator,const std::vector<NemoMeshPoint>& momentum_point,
    PetscMatrixParallelComplex*& result)
{
  // Specialized do_solve_lesser_equilibrium for Buettiker probes. This is almost identicle to the general do_solve_lesser_equilibrium
  // with a few exceptions:
  // 1. It does not accept retarded/advanced Green's functions or Boson self-energies
  // 2. It uses the chemical potential map instead of reading a single chemical potential from the input deck
  // 3. Does not have the Jacobian "hack"
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::bp_do_solve_lesser_equilibrium ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::bp_do_solver_lesser_equilibrium(): ";

  //1. Find the retarded and/or the advanced self-energies in the list of Propagators
  const Propagator* retarded_Propagator=NULL;
  const Propagator* advanced_Propagator=NULL;
  PetscMatrixParallelComplex* temp_advanced=NULL;
  PetscMatrixParallelComplex* temp_retarded=NULL;
  std::map<std::string, const Propagator*>::iterator it=Propagators.begin();
  for(; it!=Propagators.end(); it++)
  {
    NemoPhys::Propagator_type p_type=Propagator_types.find(it->first)->second;
    //1.1 Find the retarded and/or advanced Self energy containing "Buettiker_probe"
    if(it->first.find(std::string("Buettiker_probe"))!=std::string::npos)
    {
      if(p_type == NemoPhys::Fermion_retarded_self)
      {
        if(it->second==NULL)
        {
          NEMO_ASSERT(pointer_to_Propagator_Constructors->find(it->first)!=pointer_to_Propagator_Constructors->end(),prefix+
                      "have not found constructor of \""+it->first+"\"\n");
          pointer_to_Propagator_Constructors->find(it->first)->second->get_data(it->first,retarded_Propagator);
          it->second=retarded_Propagator;
        }
        else
          retarded_Propagator=it->second;
      }
      else if(p_type == NemoPhys::Fermion_advanced_self)
      {
        if(it->second==NULL)
        {
          NEMO_ASSERT(pointer_to_Propagator_Constructors->find(it->first)!=pointer_to_Propagator_Constructors->end(),prefix+
                      "have not found constructor of \""+it->first+"\"\n");
          pointer_to_Propagator_Constructors->find(it->first)->second->get_data(it->first,advanced_Propagator);
          it->second=advanced_Propagator;
        }
        else
          advanced_Propagator=it->second;
      }
      else if(p_type == NemoPhys::Boson_retarded_self)
      {
        if(it->second==NULL)
        {
          NEMO_ASSERT(pointer_to_Propagator_Constructors->find(it->first)!=pointer_to_Propagator_Constructors->end(),prefix+
                      "have not found constructor of \""+it->first+"\"\n");
          pointer_to_Propagator_Constructors->find(it->first)->second->get_data(it->first,retarded_Propagator);
          it->second=retarded_Propagator;
        }
        else
          retarded_Propagator=it->second;
      }
      else if(p_type == NemoPhys::Boson_advanced_self)
      {
        if(it->second==NULL)
        {
          NEMO_ASSERT(pointer_to_Propagator_Constructors->find(it->first)!=pointer_to_Propagator_Constructors->end(),prefix+
                      "have not found constructor of \""+it->first+"\"\n");
          pointer_to_Propagator_Constructors->find(it->first)->second->get_data(it->first,advanced_Propagator);
          it->second=advanced_Propagator;
        }
        else
          advanced_Propagator=it->second;
      }

      //else if(p_type == NemoPhys::Boson_advanced_self || p_type == NemoPhys::Boson_retarded_self) // Throws error for Bosons
      //{
      //  throw std::invalid_argument(prefix+"Not implemented for Bosons!\n");
      //}
    }
  }

  //2. Find the energy value and propagator type of output
  double energy = PropagationUtilities::read_energy_from_momentum(this,momentum_point,output_Propagator);
  NemoPhys::Propagator_type lesser_type=Propagator_types.find(output_Propagator->get_name())->second;

  //DEBUG: Output file for monitoring data

  ostringstream convert;
  convert << energy * 1000000;
  //std::string filename="bp_txt_do_solve_lesser_eq.txt";

  //std::ofstream out_file;
  //out_file.open(output_collector.get_file_path(filename,"File for testing, contains the fermi matrix, gamma matrix and the lesser self energy",
  //              NemoFileSystem::DEBUG).c_str());
  //out_file << "Energy:\t" << energy << "\n";

  //Get the retarded and advanced matrices
  if(retarded_Propagator!=NULL) // If there is a retarded propagator
  {
    Simulation* retarded_solver = find_source_of_data(retarded_Propagator);
    retarded_solver->get_data(retarded_Propagator->get_name(),&momentum_point,temp_retarded,
                              &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    //out_file << "Retarded Propagator:\t" << retarded_Propagator->get_name() << "\n";

    //temp_retarded->save_to_matlab_file(std::string(get_name()+"_test_retarded.m"));
    if(advanced_Propagator==NULL) // If there is no advanced propagator, makes one from hermitian transpose of retarded
    {
      temp_advanced=new PetscMatrixParallelComplex(temp_retarded->get_num_cols(), temp_retarded->get_num_rows(), temp_retarded->get_communicator());
      temp_retarded->hermitian_transpose_matrix(*temp_advanced,MAT_INITIAL_MATRIX);
      //out_file << "Creating advanced from retarded\n";
    }
    else // If there is a retarded and advanced propagator
    {
      Simulation* advanced_solver = find_source_of_data(advanced_Propagator);
      advanced_solver->get_data(advanced_Propagator->get_name(),&momentum_point,temp_advanced,
                                &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      //out_file << "Advanced Propagator:\t" << advanced_Propagator->get_name() << "\n";
    }
  }
  else if(advanced_Propagator!=NULL) // If there is not a retarded propagator, but there is an advanced propagator
  {
    Simulation* advanced_solver = find_source_of_data(advanced_Propagator);
    advanced_solver->get_data(advanced_Propagator->get_name(),&momentum_point,temp_advanced,
                              &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
    //out_file << "Advanced Propagator:\t" << advanced_Propagator->get_name() << "\n";

    if(retarded_Propagator==NULL) // If there is no retarded propagator, makes one from hermitian transpose of advanced
    {
      temp_retarded=new PetscMatrixParallelComplex(temp_advanced->get_num_cols(), temp_advanced->get_num_rows(), temp_advanced->get_communicator());
      temp_advanced->hermitian_transpose_matrix(*temp_retarded,MAT_INITIAL_MATRIX);
      //out_file << "Creating retarded from advanced\n";
    }
    else
    {
      Simulation* retarded_solver = find_source_of_data(retarded_Propagator);
      retarded_solver->get_data(retarded_Propagator->get_name(),&momentum_point,temp_retarded,
                                &(Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain())));
      //out_file << "Retarded Propagator:\t" << retarded_Propagator->get_name() << "\n";
    }
  }
  else
    throw std::invalid_argument(prefix+"at least one, either a retarded or an advanced Propagator must be given as input\n");

  //3. Read in the temperature
  //double temperature=options.get_option("temperature", NemoPhys::temperature);
  //temperature*=NemoPhys::boltzmann_constant/NemoPhys::elementary_charge; //[eV]
  //double chemical_potential=options.get_option("chemical_potential", 0.0); //default is 0 = value for mass less Bosons;

  //4. Fill the Matrix with -fermi(or bose)*(retarded_Propagator-advanced_Propagator) or with the 1-fermi for holes if user specified so
  //result = new PetscMatrixParallelComplex (temp_retarded->get_num_cols(), temp_retarded->get_num_rows(), temp_retarded->get_communicator());
  PetscMatrixParallelComplex* gamma = new PetscMatrixParallelComplex (*temp_retarded); //gamma = PR
  gamma->add_matrix(*temp_advanced,DIFFERENT_NONZERO_PATTERN,std::complex<double>(-1.0,0.0));   //gamma = PR - PA
  //: Why is this not needed in the normal do_solve_lesser_equilibrium?
  //*gamma *= std::complex<double>(0.0,1.0); //gamma = i(PR - PA)

  if(lesser_type==NemoPhys::Fermion_lesser_self)
  {
    // Multiplies each matrix element by fermi function with the corresponding chemical potential using matrix-matrix multiplication

    // Constructs a diagonal matrix of fermi function values to be muliply the matrix
    PetscMatrixParallelComplex* fermi_matrix = new PetscMatrixParallelComplex (*temp_retarded);

    std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
    if(!options.get_option("hole_distribution",false)) //weight with Fermi distribution
    {
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        // Retrieves chemical potential from chemical_potential map corresponding to DOFs BP
        double temp_chem_pot = chemical_potential_map.find(dof_it->second)->second;
        //: Why is this negative?
        //fermi_matrix->set(dof_it->first, dof_it->first, -NemoMath::fermi_distribution(dof_it->second,temperature,energy));
        fermi_matrix->set(dof_it->first, dof_it->first, -NemoMath::fermi_distribution(temp_chem_pot,temperature_in_eV,energy));
        //*result *= -NemoMath::fermi_distribution(chemical_potential,temperature,energy);  //result = -Fermi*(PR-PA)
      }
    }
    else //weight with 1-Fermi distribution
    {
      for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
      {
        double temp_chem_pot = chemical_potential_map.find(dof_it->second)->second;
        fermi_matrix->set(dof_it->first, dof_it->first, 1.0-NemoMath::fermi_distribution(temp_chem_pot,temperature_in_eV,energy));
        //*result *= (1.0-NemoMath::fermi_distribution(chemical_potential,temperature,energy));  //result = 1-Fermi*(PR-PA)
      }
    }

    fermi_matrix->assemble();

    //out_file << "Fermi Matrix:\n";
    //for(unsigned int i=0; i<temp_retarded->get_num_cols(); ++i)
    //{
    //  for(unsigned int j=0; j<temp_retarded->get_num_rows(); ++j)
    //  {
    //    out_file << fermi_matrix->get(i, j) << "\t";
    //  }
    //  out_file << "\n";
    //}
    //out_file << "\n";

    //fermi_matrix->save_to_matlab_file("fermi_matrix.m");

    // Multiplies the two matrices into result
    delete result;
    result = NULL;
    PetscMatrixParallelComplex::mult(*gamma, *fermi_matrix, &result);
    delete fermi_matrix;
  }
  else if(lesser_type==NemoPhys::Boson_lesser_self)
  {
    // Multiplies each matrix element by fermi function with the corresponding chemical potential using matrix-matrix multiplication

    // Constructs a diagonal matrix of fermi function values to be muliply the matrix
    PetscMatrixParallelComplex* boson_matrix = new PetscMatrixParallelComplex (*temp_retarded);

    std::map<unsigned short, unsigned short>::const_iterator dof_it = dof_to_bp_map.begin();
    for(; dof_it!=dof_to_bp_map.end(); ++dof_it)
    {
      // Retrieves chemical potential from chemical_potential map corresponding to DOFs BP
      double temp_temperature = chemical_potential_map.find(dof_it->second)->second*NemoPhys::kB_nemo;
      //: Why is this negative?
      //fermi_matrix->set(dof_it->first, dof_it->first, -NemoMath::fermi_distribution(dof_it->second,temperature,energy));
      boson_matrix->set(dof_it->first, dof_it->first, -NemoMath::bose_distribution(0.0,temp_temperature,energy));
      //*result *= -NemoMath::fermi_distribution(chemical_potential,temperature,energy);  //result = -Fermi*(PR-PA)
    }

    boson_matrix->assemble();

    //out_file << "bosn Matrix:\n";
    //for(unsigned int i=0; i<temp_retarded->get_num_cols(); ++i)
    //{
    //  for(unsigned int j=0; j<temp_retarded->get_num_rows(); ++j)
    //  {
    //    out_file << boson_matrix->get(i, j) << "\t";
    //  }
    //  out_file << "\n";
    //}
    //out_file << "\n";

    //fermi_matrix->save_to_matlab_file("fermi_matrix.m");

    // Multiplies the two matrices into result
    delete result;
    result = NULL;
    PetscMatrixParallelComplex::mult(*gamma, *boson_matrix, &result);
    delete boson_matrix;
  }

  //DEBUG: Prints out matrices
  //out_file << "Gammma:\n";
  //for(unsigned int i=0; i<temp_retarded->get_num_cols(); ++i)
  //{
  //  for(unsigned int j=0; j<temp_retarded->get_num_rows(); ++j)
  //  {
  //    out_file << gamma->get(i, j) << "\t";
  //  }
  //  out_file << "\n";
  //}
  //out_file << "\n";
  //out_file << "Lesser Self-Energy:\n";
  //for(unsigned int i=0; i<temp_retarded->get_num_cols(); ++i)
  //{
  //  for(unsigned int j=0; j<temp_retarded->get_num_rows(); ++j)
  //  {
  //    out_file << result->get(i, j) << "\t";
  //  }
  //  out_file << "\n";
  //}
  //out_file << "\n";

  //debugging:
  //lesser_it->second->imaginary_part();
  //std::string momentum_name;
  //std::vector<NemoMeshPoint> temp_momentum=momentum_point;
  //std::vector<NemoMeshPoint> * tp=&temp_momentum;
  //translate_momentum_vector(tp,momentum_name);
  //result->save_to_matlab_file(output_Propagator->get_name()+momentum_name+"eq_lesser.m");

  set_job_done_momentum_map(&(output_Propagator->get_name()),&momentum_point, true);
  if(retarded_Propagator==NULL)
    delete temp_retarded;
  if(advanced_Propagator==NULL)
    delete temp_advanced;

  delete gamma;
  //out_file.close();
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::save_local_sigma(const std::vector < std::set<communication_pair> >* input_comm_table,
                                   const std::vector < std::map<int, std::set<std::vector<NemoMeshPoint> > > >* input_local_rank_momentum_map,
                                   std::vector<const Propagator*>& list_of_propagators_to_integrate, std::vector<Simulation*> list_of_propagator_solvers,
                                   const std::string& scattering_type)
{
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::save_local_sigma ";
  NemoUtils::tic(tic_toc_prefix);
  std::string prefix="Self_energy(\""+this->get_name()+"\")::save_local_sigma(): ";

  //1. consistency checks and setup
  NEMO_ASSERT(list_of_propagators_to_integrate.size()==list_of_propagator_solvers.size(),prefix+
              "inconsistent size of input\n");

  bool is_diagonal_only=true;
  PropagationOptionsInterface* opt_interface = dynamic_cast<PropagationOptionsInterface*>(this);
  if(options.get_option("store_offdiagonals",0)>0 || opt_interface->get_compute_blockdiagonal_self_energy())
    is_diagonal_only=false;

  std::set<std::string> exclude_integration;
  //find the energy mesh name
  if(scattering_type!="deformation_potential")
  {
    std::map<std::string, NemoPhys::Momentum_type>::const_iterator momentum_name_it=momentum_mesh_types.begin();
    std::string energy_name=std::string("");
    for (; momentum_name_it!=momentum_mesh_types.end()&&energy_name==std::string(""); ++momentum_name_it)
      if(momentum_name_it->second==NemoPhys::Energy)
        energy_name=momentum_name_it->first;
    exclude_integration.insert(energy_name);
  }


  //3. prepare the information about the sparsity pattern for 4.
  //3.1 find start and end rows from the Hamilton_Constructor's DOFmap
  const DOFmapInterface& temp_dof_map=Hamilton_Constructor->get_const_dof_map(get_const_simulation_domain());
  std::vector<int> local_rows;
  temp_dof_map.get_local_row_indexes(&local_rows);

  //3.2 call the fill_set_of_row_col_indices (to set up set_of_row_col_indices)
  if(set_of_row_col_indices.size()==0)
    fill_set_of_row_col_indices(local_rows[0],local_rows[local_rows.size()-1]+1);

  //4. loop over the communication iterations
  NEMO_ASSERT(input_comm_table->size()==input_local_rank_momentum_map->size(),prefix+"received input with inconsistent sizes\n");
  int my_local_rank;
  MPI_Comm_rank(holder.one_partition_total_communicator, &my_local_rank);
  for(unsigned int i=0; i<input_comm_table->size(); i++)
  {
    //4.1 find the local momenta to integrate
    std::map<int,std::set<std::vector<NemoMeshPoint> > >::const_iterator MPI_cit=(*input_local_rank_momentum_map)[i].find(my_local_rank);
    if(MPI_cit!=(*input_local_rank_momentum_map)[i].end())
    {
      //4.2 perform the local integral
      const std::set<std::vector<NemoMeshPoint> >* local_integration_range=&(MPI_cit->second);
      NEMO_ASSERT(local_integration_range!=NULL,prefix+"received NULL for the local_integration_range\n");
      if(local_integration_range->size()>0)
      {
        std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*> weight_function;
        const std::map<std::pair<int,int>, std::map<std::vector<NemoMeshPoint>, double>*>* pointer_to_weight_function=NULL; //should be NULL, except for k-dependent scattering potentials
        //2. determine momentum dependent prefactors (required only for non-deformation potential scattering (neither acoustic nor optical))
        if(scattering_type=="polar_optical_Froehlich")
        {
          //TODO: fill the weight_function with values
          pointer_to_weight_function=&weight_function;
        }
        else if(scattering_type=="roughness")
        {
          //TODO: fill the weight_function with values
          pointer_to_weight_function=&weight_function;
        }

        for(unsigned int ii=0; ii<list_of_propagators_to_integrate.size(); ++ii)
        {
          std::vector<std::complex<double> > result;
          std::vector<std::complex<double> >* pointer_to_result=&result;
          std::vector <int> group;

          NemoPhys::Propagator_type propagator_type = get_Propagator_type(list_of_propagators_to_integrate[ii]->get_name());
          PropagationUtilities::integrate_submatrix(this, set_of_row_col_indices,local_integration_range,
                              list_of_propagator_solvers[ii], propagator_type, MPI_COMM_SELF, 0,group,
                              pointer_to_result, exclude_integration, is_diagonal_only, pointer_to_weight_function);

          //4.3 save the result in local_sigma_maps
          std::map<std::string,std::vector<std::map<std::set<std::vector<NemoMeshPoint> >,std::vector<std::complex<double> > > > >::iterator result_it=
            local_sigma_maps.find(list_of_propagators_to_integrate[ii]->get_name());
          if(result_it!=local_sigma_maps.end())
          {
            std::map<std::set<std::vector<NemoMeshPoint> >,std::vector<std::complex<double> > >::iterator temp_it=result_it->second[i].find(*local_integration_range);
            if(temp_it!=result_it->second[i].end())
            {
              NEMO_ASSERT(temp_it->second.size()==pointer_to_result->size(),prefix+"inconsistent result vector sizes\n");
              for(unsigned int j=0; j<temp_it->second.size(); ++j)
                temp_it->second[j]=(*pointer_to_result)[j];
            }
            else
              result_it->second[i][*local_integration_range]=*pointer_to_result;
          }
          else
          {
            //create an entry into the map with a 0-vector
            std::vector<std::complex<double> > temp_0_vector(pointer_to_result->size(),std::complex<double>(0.0,0.0));
            std::map<std::set<std::vector<NemoMeshPoint> >,std::vector<std::complex<double> > > temp_map;
            temp_map[*local_integration_range]=temp_0_vector; //*pointer_to_result;
            std::vector<std::map<std::set<std::vector<NemoMeshPoint> >,std::vector<std::complex<double> > > > temp_vector(input_comm_table->size(),temp_map);
            local_sigma_maps[list_of_propagators_to_integrate[ii]->get_name()]=temp_vector;
            //set the one entry of local_sigma_maps which had been solved so far
            (local_sigma_maps[list_of_propagators_to_integrate[ii]->get_name()][i])[*local_integration_range]=*pointer_to_result;
          }
        }
      }
    }
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::fill_set_of_row_col_indices(const int start_local_row, const int end_local_row)
{
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::fill_set_of_row_col_indices ";
  NemoUtils::tic(tic_toc_prefix);
  //1 get the number of offdiagonals to integrate (may be orbital resolved or (to be implemented) not)
  int off_diagonals = options.get_option("store_offdiagonals",0);
  bool store_atom_blockdiagonal = options.get_option("store_atom_blockdiagonal",false);
  //and determine whether these offdiagonals are given as DOFs or as atoms
  bool atomic_resolved_off_diagonals = false;
  int number_of_orbitals = 1;
  std::string prefix="Self_energy(\""+this->get_name()+"\")::fill_set_of_row_col_indices(): ";
  PropagationOptionsBase* prop_opts = dynamic_cast<PropagationOptionsBase*>(this);
  NEMO_ASSERT(prop_opts, prefix + ": " + get_name() + " cannot be cast to type PropagationOptionsBase");
  if (prop_opts->get_compute_blockdiagonal_self_energy())
  {
    // Clean up any existing row and column indices
    set_of_row_col_indices.clear();

    // Use Hamilton_Constructor, cast to EOMMatrixInterface, and call get_ordered_Block_keys to obtain the diagonal blocks which are the indices of the Hamiltonian
    EOMMatrixInterface* eom_interface = dynamic_cast<EOMMatrixInterface*>(Hamilton_Constructor);
    NEMO_ASSERT(eom_interface, prefix + ": " + Hamilton_Constructor->get_name() + " cannot be cast to type EOMMatrixInterface");
    std::vector<std::pair<std::vector<int>,std::vector<int> > > diagonal_indices;
    std::vector<std::pair<std::vector<int>,std::vector<int> > > offdiagonal_indices;
    eom_interface->get_ordered_Block_keys(diagonal_indices, offdiagonal_indices);

    int num_blocks = diagonal_indices.size();
    vector<int> slab_start_indices(num_blocks);
    vector<int> slab_end_indices(num_blocks);
    //int counter = 0;

    // Iterate through slabs
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
    {
      // Iterate through rows
      int nonzeros = diagonal_indices[block_idx].first.size();
      for(int i = 0; i < nonzeros; ++i)
      {
        // Iterate through columns
        for(int j = 0; j < nonzeros; ++j)
        {
          std::pair<int,int> temp_pair;
          temp_pair.first = diagonal_indices[block_idx].first[i];
          temp_pair.second = diagonal_indices[block_idx].second[j];
          set_of_row_col_indices.insert(temp_pair);
        }
      }
    }
  }
  else
  {
    if(options.check_option("store_atomic_offdiagonals"))
    {
      atomic_resolved_off_diagonals = true;
      off_diagonals = options.get_option("store_atomic_offdiagonals",0);
      //find the number of orbitals per atom
      double temp_number_of_orbitals;
      Hamilton_Constructor->get_data("number_of_orbitals",temp_number_of_orbitals);
      number_of_orbitals=int(temp_number_of_orbitals);
    }
    if(!store_atom_blockdiagonal)
    {
      //2 construct the set of local row and column indices to be taken into account
      if(!atomic_resolved_off_diagonals)
      {
        for(int i=start_local_row; i<end_local_row; i++)
        {
          std::pair<int,int> temp_pair;
          temp_pair.first=i;
          for(int j=std::max(start_local_row,i-off_diagonals); j<=min(end_local_row-1,i+off_diagonals); j++)
          {
            //msg<<prefix<<"asking for ("<<i<<", "<<j<<"\n";
            temp_pair.second=j;
            set_of_row_col_indices.insert(temp_pair);
          }
        }
      }
      else
      {
        for(int i=start_local_row; i<end_local_row; i++)
        {
          std::pair<int,int> temp_pair;
          temp_pair.first=i;
          for(int j=std::max(start_local_row,i-off_diagonals*number_of_orbitals); j<=std::min(i+off_diagonals*number_of_orbitals,end_local_row); j++)
          {
            if(std::abs(int(i-j))%number_of_orbitals==0)
            {
              temp_pair.second=j;
              set_of_row_col_indices.insert(temp_pair);
            }
          }
        }
      }
    }
    else
    {
      //loop through all atoms and couple orbitals
      //2.1 loop over all atoms
      const AtomisticDomain* domain  = dynamic_cast<const AtomisticDomain*> (this->get_simulation_domain());
      const AtomicStructure&  atoms  = domain->get_atoms();
      ConstActiveAtomIterator it     = atoms.active_atoms_begin();
      ConstActiveAtomIterator end    = atoms.active_atoms_end();
      const DOFmapInterface&           dof_map = Hamilton_Constructor->get_dof_map(get_const_simulation_domain());
      //int num_rows = dof_map.get_global_dof_number();
      //int counter = 0;
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
          std::pair<int,int> temp_pair(c_it->second,c_it->second);
          set_of_row_col_indices.insert(temp_pair);

          map<short, unsigned int>::const_iterator c_it2=atom_dofs->begin();
          //int counter2 = counter+1;
          for(; c_it2!=atom_dofs->end(); c_it2++)
          {
            std::pair<int,int> temp_pair(c_it->second,c_it2->second);
            set_of_row_col_indices.insert(temp_pair);
          }
        }

      }
    }
  }
  NEMO_ASSERT(set_of_row_col_indices.size()>0,prefix+"empty set_of_row_col_indices\n");
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::check_and_correct_coupling_consistency(std::set<communication_pair>* in_and_out_comm_table)
{
  std::string tic_toc_prefix = "Self_energy(\""+tic_toc_name+"\")::fill_set_of_row_col_indices ";
  NemoUtils::tic(tic_toc_prefix);

  //0. translate the in_and_out_comm_table into a map to fasten the later searches
  std::map<std::vector<NemoMeshPoint>,std::set<std::vector<NemoMeshPoint> > > search_list;
  std::set<communication_pair>::iterator temp_it=in_and_out_comm_table->begin();
  for(; temp_it!=in_and_out_comm_table->end(); ++temp_it)
    search_list[(*temp_it).first]=(*temp_it).second;

  //1. loop over all entries in search_list
  std::map<std::vector<NemoMeshPoint>,std::set<std::vector<NemoMeshPoint> > >::iterator A_it=search_list.begin();
  for(; A_it!=search_list.end(); ++A_it)
  {
    //typedef std::pair<std::vector<NemoMeshPoint>, std::set<std::vector<NemoMeshPoint> > > communication_pair;
    //2. loop over all second entries of the specific communication_pair
    std::vector<NemoMeshPoint> first_momentum=A_it->first;
    std::set<std::vector<NemoMeshPoint> > target_states=A_it->second;
    std::set<std::vector<NemoMeshPoint> >::iterator B_it=target_states.begin();
    for(; B_it!=target_states.end(); ++B_it)
    {
      //3. check for each second that that entry in the communication_pair that has this second as first, the current first is within second
      std::map<std::vector<NemoMeshPoint>,std::set<std::vector<NemoMeshPoint> > >::iterator second_as_first_iterator=search_list.find(*B_it);
      if(second_as_first_iterator!=search_list.end())
      {
        std::set<std::vector<NemoMeshPoint> >::iterator second_temp_it=second_as_first_iterator->second.find(first_momentum);
        //3.1 if not, add this first to it
        if(second_temp_it==second_as_first_iterator->second.end())
          second_as_first_iterator->second.insert(first_momentum);
      }
      else
      {
        //3.1 if not, add this first to it
        std::set<std::vector<NemoMeshPoint> > temp_set;
        temp_set.insert(first_momentum);
        search_list[*B_it]=temp_set;
      }
    }
  }
  //4. translate the search_list back into the set (refill the in_and_out_comm_table
  in_and_out_comm_table->clear();
  A_it=search_list.begin();
  for(; A_it!=search_list.end(); ++A_it)
  {
    communication_pair temp_pair(A_it->first,A_it->second);
    in_and_out_comm_table->insert(temp_pair);
  }
  NemoUtils::toc(tic_toc_prefix);
}

void Self_energy::do_solve_for_scBorn_scattering_self_energies(void)
{
  //check that the Propagation is initialized...
  if(!Propagation_is_initialized)
    initialize_Propagation(get_name());
  //check that the Propagator is initialized
  if(!is_Propagator_initialized(name_of_writeable_Propagator))
  {
    std::string prefix="Self_energy(\""+this->get_name()+"\")::do_solve: ";
    initialize_Propagators(name_of_writeable_Propagator);
    NEMO_ASSERT(is_Propagator_initialized(name_of_writeable_Propagator),
                prefix+"Propagator \""+name_of_writeable_Propagator+"\" is not initialized after initialize_Propagators is called\n");
    get_data(name_of_writeable_Propagator,writeable_Propagator);
  }
  do_full_MPI_solve(writeable_Propagator);
}

void Self_energy::set_input_options_map()
{
  Propagation::set_input_options_map();

  set_input_option_map("do_output",InputOptions::NonReq_Def("false","output all propagator matrix into matlab"));
  set_input_option_map("clean_up",InputOptions::NonReq_Def("false",
                       "clean up propagators, set job to be not ready"));
  set_input_option_map("do_output_scattering_rate",InputOptions::NonReq_Def("false","calculate and output scattering rate"));
  set_input_option_map("number_of_k_points_for_scattering_rate",InputOptions::NonReq_Def("30",
                       "number of k points for scattering rate calculations"));
  set_input_option_map("constant_eta",InputOptions::NonReq_Def("0.0","artificial dephasing usually used in direct_iterations method"));
  set_input_option_map("iterative_damping",InputOptions::NonReq_Def("0.5","exponential damping ratio for constant_eta"));
  set_input_option_map("retarded_lead_method",
                       InputOptions::Req_Def("self-energy solver options: direct_iterations/transfer_matrix/transfer_matrix_avoid_surface_greens_function\
     we usually use transfer_matrix_avoid_surface_greens_function for QTBM ref:M.Luisier, PRB 74, 205323 (2006)/NEMO5_BC.ppt"));
  set_input_option_map("linear_system_solver",InputOptions::NonReq_Def("mumps","linear solver options: petsc/mumps\
                       ref: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/index.html"));
  set_input_option_map("constant_lead_eta",InputOptions::NonReq_Def("1e-12","eta added to diagonal of lead Hamiltonian used in transfer_matrix"));
  set_input_option_map("LRA",InputOptions::NonReq_Def("false","use LRA only for transfer_matrix ref: NEMO5_BC.ppt"));
  set_input_option_map("ratio_of_eigenvalues_to_be_solved",InputOptions::NonReq_Def("1.0",
                       "percentage of number of eigenvalues (0~1.0) to be solved in transfer_matrix method"));
  set_input_option_map("wire_nonSO_basis",InputOptions::NonReq_Def("false",
                       "if true, solve double matrix instead of complex matrix, only worked for wire and no spin tb_basis"));
  set_input_option_map("output_Ek",InputOptions::NonReq_Def("false","output Ek for the lead Hamiltonian from transfer_matix method"));
  set_input_option_map("debug_output_job_list",InputOptions::NonReq_Def("false","output all temporary matrix for debug purpose"));
  set_input_option_map("stationary_Propagation",InputOptions::NonReq_Def("true","time dependent NEGF or not, so far only stationary work"));
  set_input_option_map("equilibrium_model",InputOptions::NonReq_Def("true","lesser self-energy calculation"));
  set_input_option_map("temperature",InputOptions::NonReq_Def("300","temperature for lead"));
  set_input_option_map("chemical_potential",InputOptions::NonReq_Def("0.0","chemical potential for lead"));
  set_input_option_map("scattering_type",
                       InputOptions::Req_Def("scattering model options: deformation_potential/LO_phonon_scattering/lambda_G_scattering"));
  set_input_option_map("scattering_energy_interval",InputOptions::NonReq_Def("0.0","energy interval for scattering"));
  set_input_option_map("scattering_momentum_interval",InputOptions::NonReq_Def("1e20","momentum interval for scattering"));
  set_input_option_map("deformation_potential",InputOptions::NonReq_Def("1.0","Potential for the scattering of electrons on lattice deforming phonons in eV"));
  set_input_option_map("store_offdiagonals",InputOptions::NonReq_Def("0",""));
  set_input_option_map("store_atomic_offdiagonals",InputOptions::NonReq_Def("0",""));
  set_input_option_map("phonon_temperature",InputOptions::NonReq_Def("300",""));
  set_input_option_map("average_eps0",InputOptions::NonReq_Def("12.0",""));
  set_input_option_map("average_epsinf",InputOptions::NonReq_Def("10.0",""));
  set_input_option_map("phonon_energy",InputOptions::NonReq_Def("0.0",""));
  set_input_option_map("lambda",InputOptions::NonReq_Def("0.0",""));
  set_input_option_map("maximum_Sancho_iterations",InputOptions::NonReq_Def("50",
                       "Maximum number of iterations for the Sancho-Rubio method of solving the contact self-energy."));
  set_input_option_map("bp_current_threshold",InputOptions::NonReq_Def("1e-15","Threshold for Buettiker Probe Newton Raphson to be met."));
  set_input_option_map("bp_full_transmission_calculation",InputOptions::NonReq_Def("false",
                       "True if efficient transmission calculation is done for single nonzero Buettiker probe self-energies."));
  set_input_option_map("bp_material_tags",InputOptions::NonReq_Def("No default","Name of materials which contain relevant Buettiker probes."));

  set_input_option_map("Hamilton_constructor0",InputOptions::NonReq_Def("", "schroedinger solver for 0th block"));
  set_input_option_map("Hamilton_constructor1",InputOptions::NonReq_Def("", "schroedinger solver for 1st block"));
  set_input_option_map("Hamilton_constructor2",InputOptions::NonReq_Def("", "schroedinger solver for 2nd block"));
  set_input_option_map("Hamilton_constructor3",InputOptions::NonReq_Def("", "schroedinger solver for 3rd block"));
  set_input_option_map("Hamilton_constructor4",InputOptions::NonReq_Def("", "schroedinger solver for 4th block"));
  set_input_option_map("lead_domain0",InputOptions::NonReq_Def("", "lead domain of subdomain0"));
  set_input_option_map("lead_domain1",InputOptions::NonReq_Def("", "lead domain of subdomain1"));
  set_input_option_map("lead_domain2",InputOptions::NonReq_Def("", "lead domain of subdomain2"));
  set_input_option_map("lead_domain3",InputOptions::NonReq_Def("", "lead domain of subdomain3"));
  set_input_option_map("lead_domain4",InputOptions::NonReq_Def("", "lead domain of subdomain4"));
  set_input_option_map("MPI_interleaving",InputOptions::NonReq_Def("", ""));
  set_input_option_map("scattering_type",InputOptions::NonReq_Def("", ""));
  set_input_option_map("shift",InputOptions::NonReq_Def("", ""));

  set_input_option_map("output_complexEk",InputOptions::NonReq_Def("false", "bool type, output complex E-k"));
  set_input_option_map("bp_probe_model",InputOptions::NonReq_Def("","Specify the type of Buettiker probe models"));
  set_input_option_map("bp_initial_chemical_potential_model",InputOptions::NonReq_Def("zeros", "choose the model to initialize the chemical potential in each probe"));
  set_input_option_map("bp_chem_potential_linear",InputOptions::NonReq_Def("No default", " setup the chemical potential corners for the Buettiker probe"));
  set_input_option_map("source_location",InputOptions::NonReq_Def("0", "temporary location for the source to calculate the initial chemical potential map"));
  set_input_option_map("drain_location",InputOptions::NonReq_Def("0", "temporary location for the drain to calculate the initial chemical potential map"));
  set_input_option_map("bp_chem_potential_direct",InputOptions::NonReq_Def("", "initializing the chemical potential values directly reads in a vector of values equal to the number of Buettiker probes"));
  set_input_option_map("rgf_override",InputOptions::NonReq_Def("", "Checks for RGF override, this function is under development"));
  set_input_option_map("bp_coupling",InputOptions::NonReq_Def("", "setup the BP coupling value, include real and imaginary part"));
  set_input_option_map("Bands:BandEdge:Ec",InputOptions::NonReq_Def("0", "initialize the band edge for the conduction band"));
  set_input_option_map("rgf_eta",InputOptions::NonReq_Def("", "get the imaginary part of coupling constant for RGF method, this part is under development"));
  set_input_option_map("rgf_band",InputOptions::NonReq_Def("", "get the conduction band edge for RGF method, this part is under development"));
  set_input_option_map("bp_analytical_momenta",InputOptions::NonReq_Def("", "specify the analytical momenta type, include 1D, 2D and nanowire case"));
  set_input_option_map("G_for_transmission",InputOptions::NonReq_Def("", "Get the retarded Green's function of the device (either the only retarded Green's function within this propagation, or read in from inputdeck"));
  set_input_option_map("bp_full_transmission_calculation",InputOptions::NonReq_Def("false", "output the type for the transmission calculation"));
  set_input_option_map("bp_trans_map_storage",InputOptions::NonReq_Def("sparse", " setup the matrix type for the transmission map, include sparse and dense"));
  set_input_option_map("hole_distribution",InputOptions::NonReq_Def("false", "specify the fermi distribution type, include electron and hole"));
}

void Self_energy::set_description()
{
  description = "Self_energy Solver";
}
