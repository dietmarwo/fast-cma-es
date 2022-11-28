
'''
Optimizing the efficiency of a power plant. 

See
https://tespy.readthedocs.io/en/main/tutorials_examples.html#thermal-power-plant-efficiency-optimization

Added the application of fcmaes optimizers, both using parallel function evaluation and parallel retry. 

Note: tespy doesn't support Python 3.9, tested on anaconda with python 3.8 on linux. 

See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/PowerPlant.adoc for a detailed description.

'''

from tespy.networks import Network
from tespy.components import (
    Turbine, Splitter, Merge, Condenser, Pump, Sink, Source,
    HeatExchangerSimple, Desuperheater, CycleCloser
)
from tespy.connections import Connection, Bus
from tespy.tools import logger
import logging
import numpy as np

logger.define_logging(screen_level=logging.ERROR)

class PowerPlant():

    def __init__(self):
        self.nw = Network(
            fluids=['BICUBIC::water'],
            p_unit='bar', T_unit='C', h_unit='kJ / kg',
            iterinfo=False)#, memorise_fluid_properties=False)
        # components
        # main cycle
        eco = HeatExchangerSimple('economizer')
        eva = HeatExchangerSimple('evaporator')
        sup = HeatExchangerSimple('superheater')
        cc = CycleCloser('cycle closer')
        hpt = Turbine('high pressure turbine')
        sp1 = Splitter('splitter 1', num_out=2)
        mpt = Turbine('mid pressure turbine')
        sp2 = Splitter('splitter 2', num_out=2)
        lpt = Turbine('low pressure turbine')
        con = Condenser('condenser')
        pu1 = Pump('feed water pump')
        fwh1 = Condenser('feed water preheater 1')
        fwh2 = Condenser('feed water preheater 2')
        dsh = Desuperheater('desuperheater')
        me2 = Merge('merge2', num_in=2)
        pu2 = Pump('feed water pump 2')
        pu3 = Pump('feed water pump 3')
        me = Merge('merge', num_in=2)

        # cooling water
        cwi = Source('cooling water source')
        cwo = Sink('cooling water sink')

        # connections
        # main cycle
        cc_hpt = Connection(cc, 'out1', hpt, 'in1', label='feed steam')
        hpt_sp1 = Connection(hpt, 'out1', sp1, 'in1', label='extraction1')
        sp1_mpt = Connection(sp1, 'out1', mpt, 'in1', state='g')
        mpt_sp2 = Connection(mpt, 'out1', sp2, 'in1', label='extraction2')
        sp2_lpt = Connection(sp2, 'out1', lpt, 'in1')
        lpt_con = Connection(lpt, 'out1', con, 'in1')
        con_pu1 = Connection(con, 'out1', pu1, 'in1')
        pu1_fwh1 = Connection(pu1, 'out1', fwh1, 'in2')
        fwh1_me = Connection(fwh1, 'out2', me, 'in1', state='l')
        me_fwh2 = Connection(me, 'out1', fwh2, 'in2', state='l')
        fwh2_dsh = Connection(fwh2, 'out2', dsh, 'in2', state='l')
        dsh_me2 = Connection(dsh, 'out2', me2, 'in1')
        me2_eco = Connection(me2, 'out1', eco, 'in1', state='l')
        eco_eva = Connection(eco, 'out1', eva, 'in1')
        eva_sup = Connection(eva, 'out1', sup, 'in1')
        sup_cc = Connection(sup, 'out1', cc, 'in1')

        self.nw.add_conns(cc_hpt, hpt_sp1, sp1_mpt, mpt_sp2, sp2_lpt,
                          lpt_con, con_pu1, pu1_fwh1, fwh1_me, me_fwh2,
                          fwh2_dsh, dsh_me2, me2_eco, eco_eva, eva_sup, sup_cc)

        # cooling water
        cwi_con = Connection(cwi, 'out1', con, 'in2')
        con_cwo = Connection(con, 'out2', cwo, 'in1')

        self.nw.add_conns(cwi_con, con_cwo)

        # preheating
        sp1_dsh = Connection(sp1, 'out2', dsh, 'in1')
        dsh_fwh2 = Connection(dsh, 'out1', fwh2, 'in1')
        fwh2_pu2 = Connection(fwh2, 'out1', pu2, 'in1')
        pu2_me2 = Connection(pu2, 'out1', me2, 'in2')

        sp2_fwh1 = Connection(sp2, 'out2', fwh1, 'in1')
        fwh1_pu3 = Connection(fwh1, 'out1', pu3, 'in1')
        pu3_me = Connection(pu3, 'out1', me, 'in2')

        self.nw.add_conns(sp1_dsh, dsh_fwh2, fwh2_pu2, pu2_me2,
                          sp2_fwh1, fwh1_pu3, pu3_me)

        # busses
        # power bus
        self.power = Bus('power')
        self.power.add_comps(
            {'comp': hpt, 'char': -1}, {'comp': mpt, 'char': -1},
            {'comp': lpt, 'char': -1}, {'comp': pu1, 'char': -1},
            {'comp': pu2, 'char': -1}, {'comp': pu3, 'char': -1})

        # heating bus
        self.heat = Bus('heat')
        self.heat.add_comps(
            {'comp': eco, 'char': 1}, {'comp': eva, 'char': 1},
            {'comp': sup, 'char': 1})

        self.nw.add_busses(self.power, self.heat)

        # parametrization
        # components
        hpt.set_attr(eta_s=0.9)
        mpt.set_attr(eta_s=0.9)
        lpt.set_attr(eta_s=0.9)

        pu1.set_attr(eta_s=0.8)
        pu2.set_attr(eta_s=0.8)
        pu3.set_attr(eta_s=0.8)

        eco.set_attr(pr=0.99)
        eva.set_attr(pr=0.99)
        sup.set_attr(pr=0.99)

        con.set_attr(pr1=1, pr2=0.99, ttd_u=5)
        fwh1.set_attr(pr1=1, pr2=0.99, ttd_u=5)
        fwh2.set_attr(pr1=1, pr2=0.99, ttd_u=5)
        dsh.set_attr(pr1=0.99, pr2=0.99)

        # connections
        eco_eva.set_attr(x=0)
        eva_sup.set_attr(x=1)

        cc_hpt.set_attr(m=200, T=650, p=100, fluid={'water': 1})
        hpt_sp1.set_attr(p=20)
        mpt_sp2.set_attr(p=3)
        lpt_con.set_attr(p=0.05)

        cwi_con.set_attr(T=20, p=10, fluid={'water': 1})

    def calculate_efficiency(self, x):
        # set extraction pressure
        self.nw.get_conn('extraction1').set_attr(p=x[0])
        self.nw.get_conn('extraction2').set_attr(p=x[1])

        self.nw.solve('design')

        # components are saved in a DataFrame, column 'object' holds the
        # component instances
        for cp in self.nw.comps['object']:
            if isinstance(cp, Condenser) or isinstance(cp, Desuperheater):
                if cp.Q.val > 0:
                    return np.nan
            elif isinstance(cp, Pump):
                if cp.P.val < 0:
                    return np.nan
            elif isinstance(cp, Turbine):
                if cp.P.val > 0:
                    return np.nan

        if self.nw.res[-1] > 1e-3 or self.nw.lin_dep:
            return np.nan
        else:
            return self.nw.busses['power'].P.val / self.nw.busses['heat'].P.val

    def calculate_qd(self, x):
        y = self.calculate_efficiency(x)
        desc = [self.nw.busses['power'].P.val, self.nw.busses['heat'].P.val]
        return y, desc
        
def optimize_fcmaes():
    
    from fcmaes.optimizer import Bite_cpp, De_cpp, Cma_cpp, Crfmnes_cpp, de_cma, wrapper
    from fcmaes import retry, mode, modecpp, de, bitecpp, decpp, crfmnes, crfmnescpp, cmaes
    from scipy.optimize import Bounds
    import threading, math, threadpoolctl
    
    class fcmaes_problem():
        
        def __init__(self):
            self.dim = 2
            self.nobj = 1
            self.ncon = 1
            self.bounds = Bounds([1]*self.dim, [40]*self.dim)          
            self.local = threading.local()
           
        def get_model(self):
            if not hasattr(self.local, 'model'):
                self.create_model()
            return self.local.model
        
        def create_model(self):
            self.local.model = PowerPlant()
        
        def efficiency(self, x):   
            try:
                with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                    eff = self.get_model().calculate_efficiency(x)    
                if not np.isfinite(eff): # model gets corrupted in case of an error
                    self.create_model() # we need to recreate the model
                    return 0
                return eff
            except Exception as ex:
                return 0  
  
        def fitness(self, x):
            y = -self.efficiency(x)
            c = -x[0] + x[1]
            return [y, c]
    
        def fitness_so(self, x):
            if x[1] > x[0]: # handle constraint
                return 1000 + x[1] - x[0]
            return -self.efficiency(x)
            
    problem = fcmaes_problem()
    
    # Parallel retry of different single-objective optimizers

    # ret = retry.minimize(wrapper(problem.fitness_so), problem.bounds,
    #                       num_retries = 32, optimizer=Bite_cpp(20000))            
    #
    # ret = retry.minimize(wrapper(problem.fitness_so), problem.bounds,
    #                       num_retries = 32, optimizer=De_cpp(20000))     
    #
    # ret = retry.minimize(wrapper(problem.fitness_so), problem.bounds,
    #                       num_retries = 32, optimizer=Cma_cpp(20000))       
    #
    # ret = retry.minimize(wrapper(problem.fitness_so), problem.bounds,
    #                       num_retries = 32, optimizer=Crfmnes_cpp(20000))          
   
    # Multi objective optimization parallel retry:   
 
    # x, y = modecpp.retry(mode.wrapper(problem.fitness, problem.nobj), problem.nobj, 
    #              problem.ncon, problem.bounds, 
    #              popsize = 32, max_evaluations = 1000000, 
    #              nsga_update=True, num_retries = 32,
    #              workers=32)
    #
    # # Differential Evolution using parallel function evaluation:
    #
    ret = de.minimize(wrapper(problem.fitness_so), problem.dim, problem.bounds, max_evaluations = 20000, workers=32)   
    
    # Multi objective optimization using parallel function evaluation:         

    # x, y = mode.minimize(mode.wrapper(problem.fitness, problem.nobj), problem.nobj, 
    #                            problem.ncon, problem.bounds, 
    #                            popsize = 32, max_evaluations = 100000, nsga_update=True, 
    #                            workers=32)

    # The C++ version of this algorithm only works single threaded with tespy, but modecpp.retry works multi threaded 
        
    # x, y = modecpp.minimize(mode.wrapper(problem.fitness, problem.nobj), problem.nobj, 
    #                            problem.ncon, problem.bounds, 
    #                            popsize = 32, max_evaluations = 100000, nsga_update=True, 
    #                            workers=1)
       
    # some single threaded single objective optimizers
          
    #ret = decpp.minimize(wrapper(problem.fitness_so), problem.dim, problem.bounds, max_evaluations = 20000)            
    
    #ret = cmaes.minimize(wrapper(problem.fitness_so), problem.bounds, max_evaluations = 20000)            
    
    #ret = bitecpp.minimize(wrapper(problem.fitness_so), problem.bounds, max_evaluations = 20000)            
    
    #ret = de_cma(20000).minimize(wrapper(problem.fitness_so), problem.bounds)  
      

def optimize_qd():
    
    from fcmaes import diversifier, mapelites
    from scipy.optimize import Bounds
    import threadpoolctl, threading
    
    class qd_problem():
        
        def __init__(self):
            self.dim = 2
            self.qd_dim = 2
            self.bounds = Bounds([1]*self.dim, [40]*self.dim)          
            self.desc_bounds = Bounds([2.2E8, 5E8], [2.8E8, 6.3E8])          
            self.local = threading.local()
        
        def get_model(self):
            if not hasattr(self.local, 'model'):
                self.create_model()
            return self.local.model
        
        def create_model(self):
            self.local.model = PowerPlant()
        
        def efficiency(self, x):   
            try:
                with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                    eff, desc = self.get_model().calculate_qd(x)    
                if not np.isfinite(eff): # model gets corrupted in case of an error
                    self.create_model() # we need to recreate the model
                    return 0, self.desc_bounds.lb
                #print (eff, desc)
                return eff, desc
            except Exception as ex:
                return 0, self.desc_bounds.lb  
  
        def qd_fitness(self, x):
            y, desc = self.efficiency(x)
            return 1-y, desc

    problem = qd_problem()
    name = 'powerplant2'
    opt_params0 = {'solver':'elites', 'popsize':1000, 'use':2}
    opt_params1 = {'solver':'CMA_CPP', 'max_evals':2000, 'popsize':16, 'stall_criterion':3}
    archive = diversifier.minimize(
         mapelites.wrapper(problem.qd_fitness, 2, interval=1000), problem.bounds, problem.desc_bounds, 
         workers = 32, opt_params=[opt_params0, opt_params1], retries=640, 
         niche_num = 4000, samples_per_niche = 20)
    
    print('final archive:', archive.info())
    archive.save(name)
    plot_archive(archive)

from elitescass2 import plot3d

def plot_archive(archive):
    si = archive.argsort()
    ysp = []
    descriptions = archive.get_ds()[si]
    ys = archive.get_ys()[si]
    for i in range(len(si)):
        if ys[i] < 1: # throw out invalid
            desc = descriptions[i]
            ysp.append([desc[0], desc[1], ys[i]])

    ysp = np.array(ysp)
    print(len(ysp))
    print(ysp)
    plot3d(ysp, "powerplant2", 'power', 'heat', 'power / heat')
    
if __name__ == '__main__':
    #optimize_fcmaes()
    optimize_qd()
    
