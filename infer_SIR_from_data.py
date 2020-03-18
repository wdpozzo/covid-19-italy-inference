#!/usr/bin/env python
import numpy as np
from math import log
import cpnest.model
import cpnest
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numba import jit

__author__ = "Walter Del Pozzo"
__email__  = "walter.delpozzo@unipi.it"

@jit
def dN_dt(N, t, E, p, initial_population):
    """
    basic logistic differential model
    with populationcap
    """
    return E*p*(1.0-N/initial_population)*N

def N_t(t, x):
    N0 = x['N0']
    E  = x['E']
    p  = x['p']
    initial_population = x['population']
    return np.squeeze(odeint(dN_dt,N0,t,args=(E,p,initial_population,)))

@jit
def dS_dt(S, I, beta, population):
    """
    basic logistic differential model
    with populationcap
    """
    return -beta*S*I/population

@jit
def dI_dt(S, I, beta, gamma, population):
    """
    basic logistic differential model
    with populationcap
    """
    return beta*S*I/population-gamma*I

@jit
def dR_dt(I, gamma):
    """
    basic logistic differential model
    with populationcap
    """
    return gamma*I
    
@jit
def dSIR_dt(SIR, t, beta, gamma):
    S, I, R = SIR
    population = S + I + R
    return [dS_dt(S, I, beta, population), dI_dt(S, I, beta, gamma, population), dR_dt(I, gamma)]
    
def SIR(t, x):
    beta   = x['beta']
    gamma  = x['gamma']
    I0     = x['I0']
    R0     = x['R0']
    initial_population = x['population']
#    mr     = 0.0#x['mortality_rate']*x['fraction_of_positive']*x['fraction_of_hospitalised']
    return odeint(dSIR_dt,[initial_population-I0-R0,I0,R0],t,args=(beta, gamma))

@jit
def poisson_log_likelihood(k,M):
    """
    """
    return -M+k*log(M)-log_stirling_approx(k)
    
@jit
def binomial_log_likelihood(n,k,p):
    """
    """
    return k*log(p)+(n-k)*log(1.0-p)+log_stirling_approx(n)-log_stirling_approx(k)-log_stirling_approx(n-k)

@jit
def log_stirling_approx(n):
    return n*log(n)-n

class diffusion_model(cpnest.model.Model):
    names = []
    bounds = []
    
    def __init__(self, data, time, growth_model = None):
        super(diffusion_model,self).__init__()
        self.infected = data['totale_casi']
        self.positive = data['totale_attualmente_positivi']
        self.recovered = data['dimessi_guariti']
        self.hospitalised = data['totale_ospedalizzati']
        self.dead        = data['deceduti']
        self.errors_infected = np.sqrt(self.infected)
        self.errors_recovered = np.sqrt(self.recovered)
        self.errors_hospitalised = np.sqrt(self.hospitalised)
        self.errors_dead        = np.sqrt(self.dead)
        self.tests  = data['tamponi']
        self.time   = time
        self.growth_model = growth_model
        
        if growth_model == "logistic":
            # these are the intrinsic population parametes affected by the virus
            self.names = ['N0','E', 'p']
            self.bounds = [[1.0,5000.0],[0.0,3.0],[0.0,1.0]]
            # these are the sampling parameters, we are assuming a binomial likelihood for the number of positive vs number of tests
            # this is inaccurate since tests are "clustered" in infection spots
            self.names.append('fraction_of_positive')
            self.bounds.append([0.01,1.0])
            self.log_likelihood = self.log_likelihood_logistic
        
        if growth_model == "SIR":
            # these are the intrinsic population parameters affected by the virus
            # I0: initial number of infected
            # R0: initial number or recevered
            # beta: mean contact rate   (in 1/days)
            # gamma: mean recovery rate (in 1/days)
            # population: total number of affected individuals or susceptible individuals at time 0
            
            self.names = ['I0','R0','beta','gamma','population']
            self.bounds = [[np.maximum(self.infected[0]-3*self.errors_infected[0],1.0),np.maximum(self.infected[0]+3*self.errors_infected[0],1e5)],
                           [np.maximum(self.recovered[0]-3*self.errors_recovered[0],0.0),np.maximum(self.recovered[0]+3*self.errors_recovered[0],1e5)],
                           [1e-2,5.0],
                           [1e-2,5.0],
                           [1e4,6e7]]
            
            self.names.append('fraction_of_positive')
            self.bounds.append([0.01,1.0])
            self.names.append('fraction_of_hospitalised')
            self.bounds.append([0.01,1.0])
            self.log_likelihood = self.log_likelihood_SIR
        else:
            print("please specify a growth model between logistic and SIR")
            exit()
        
    def log_prior(self, x):
        """
        log prior function, uniform in all variables
        """
        logP = super(diffusion_model,self).log_prior(x)
        return logP
            
    def log_likelihood_logistic(self, x):
        f = x['fraction_of_positive']
        m = f*self.tests
        v = np.sqrt(m*(1.0-f))
        logL = -0.5*np.sum(((m-self.infected)/v)**2)
        normalised_residuals = (self.infected - f*N_t(self.time,x))/self.errors
        return -0.5*np.sum(normalised_residuals**2)+logL

    def log_likelihood_SIR(self, x):
        """
        log likelihood for the SIR model
        We correct the observed number of infected individuals for the detection efficiency
        related to the number of tests that a country has performed as a function of time
        we then also correct the number of recovered individuals for the
        estimated number of ospitalisations
        
        f: fraction of positive tests over the number of tests
        g: fraction of ospitalisations over the number of infected
        
        N_infected = f*Ntot_infected
        N_ospitalised = g*N_infected
        N_recovered = f*g*N_tot_recovered
        N_dead = k*N_ospitalised
        
        """
        frac = 10.0/301338.0 # fraction of average surface covered by an individual compared to the surface of Italy
        logL = 0.0
        N = len(self.time)
        # solution is a tuple (S,I,R)
        solution = SIR(self.time, x)
        # compute the average length of positive streaks
        T = 1./x['gamma']
        idx = int(np.floor(3*T))
        correlation_length = np.ones(N)
        for i in range(N):
            if i-idx < 0: correlation_length[i] += x['beta']*np.sum(solution[0:i,0])
            else:  correlation_length[i] += np.sum(solution[i-idx:i,0])
            if correlation_length[i] > self.tests[i]: correlation_length[i] = self.tests[i]
#
        correlation_length *= frac*x['beta']#*solution[:,0]
        correlation_length = 1+np.ceil(correlation_length)
#
        f = x['fraction_of_positive']
        g = x['fraction_of_hospitalised']
#
        # downsample the tests for the estimated cluster size
        effective_number_of_tests = 1+(self.tests/correlation_length).astype(np.int)
        effective_number_of_infected = 1+(self.infected/correlation_length).astype(np.int)
        logL += np.sum([binomial_log_likelihood(effective_number_of_tests[i],effective_number_of_infected[i],f) for i in range(N)])
        # now estimate the fraction of people that recover without going to the hospital
        g = x['fraction_of_hospitalised']
        logL += np.sum([binomial_log_likelihood(self.infected[i],self.hospitalised[i],g) for i in range(N)])
        
        logL += np.sum([poisson_log_likelihood(self.infected[i],f*solution[i,1]) for i in range(N)])
        logL += np.sum([poisson_log_likelihood(self.recovered[i],f*g*solution[i,2]) for i in range(N)])

        return logL
        
def compute_confidence_region(time, model, ax, color_mean, color_out):
    mll,mm,mhh = np.percentile(model, [5,50,95], axis = 0)
    ax.fill_between(time, mll, mhh, facecolor = color_out, zorder = 0)
    ax.plot(time, mm, color = color_mean)
    return ax


if __name__ == "__main__":

    import os
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-d', '--data',        default=None, type='string', metavar='data', help='data to analyse')
    parser.add_option('-r', '--region',      default=None, type='string', metavar='region', help='regional data')
    parser.add_option('-o', '--output',      default=None, type='string', metavar='DIR', help='directory for output')
    parser.add_option('-m', '--model',       default='SIR', type='string', metavar='model', help='model for the analysis (default SIR). Supports SIR, SIRN and logistic')
    parser.add_option('-t', '--threads',     default=4, type='int', metavar='threads', help='Number of threads (default = 1/core)')
    parser.add_option('-s', '--seed',        default=0, type='int', metavar='seed', help='Random seed initialisation')
    parser.add_option('--nlive',             default=1000, type='int', metavar='nlive', help='Number of live points')
    parser.add_option('--poolsize',          default=100, type='int', metavar='poolsize', help='Poolsize for the samplers')
    parser.add_option('--maxmcmc',           default=1000, type='int', metavar='maxmcmc', help='Maximum number of mcmc steps')
    parser.add_option('--postprocess',       default=0, type='int', metavar='postprocess', help='Run only the postprocessing')
    (opts,args)=parser.parse_args()
    
    import pandas as pd
    data = pd.read_csv(opts.data, sep=',')
    if opts.region is None:
        data = data.to_records()
    else:
        data = data[data['denominazione_regione'].str.contains(opts.region)].to_records()

    time = 1.0+np.linspace(0.0,data.shape[0],data.shape[0])
    M = diffusion_model(data, time, growth_model=opts.model)
    
    if 1:
        work=cpnest.CPNest(M,
                           verbose  = 2,
                           poolsize = opts.poolsize,
                           nthreads = opts.threads,
                           nlive    = opts.nlive,
                           maxmcmc  = opts.maxmcmc,
                           output   = opts.output,
                           resume   = 1)
        work.run()
        print('Model evidence {0}'.format(work.NS.logZ))
        x = work.get_posterior_samples(filename='posterior.dat')
    else:
        x = np.genfromtxt(os.path.join(opts.output,'posterior.dat'), names = True)

    import matplotlib
    matplotlib.use("MACOSX")
    import matplotlib.dates as mdates
    import matplotlib.ticker as ticker
    import datetime as DT
    import corner
    

    observation_dates = mdates.num2date(mdates.drange(DT.datetime(2020, 2, 24),
                                                      DT.datetime(2020, 3, 13),
                                                      DT.timedelta(days=1)))
    
    prediction_dates = mdates.num2date(mdates.drange(DT.datetime(2020, 2, 24),
                                                     DT.datetime(2020, 5, 30),
                                                     DT.timedelta(days=1)))
    if M.growth_model ==  "logistic":
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        time_simulation = 1.0+np.linspace(0.0,100,100)
        observed_models = []
        total_models    = []
        for xi in x:
            yi = xi['fraction_of_positive']*N_t(time_simulation,xi)
            observed_models.append(yi)
            total_models.append(yi/xi['fraction_of_positive'])
        observed_models = np.array(observed_models)
        total_models = np.array(total_models)
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        mll,ml,mm,mh,mhh = np.percentile(observed_models, [5,14,50,68,95], axis = 0)
        ax.fill_between(time_simulation, mll, mhh, facecolor = 'turquoise')
        ax.fill_between(time_simulation, ml, mh, facecolor = 'aquamarine')
        ax.plot(time_simulation, mm, color = 'k')
        ax.errorbar(time, data['totale_casi'], yerr=np.sqrt(data['totale_casi']), fmt='.', color='black',
                    ecolor='black', elinewidth=3, capsize=0)
        ax.set_xlim(-0.5,time[-1]+0.5)
        ax.set_xlabel('days')
        ax.set_ylabel('observed number of infections')
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.set_yscale('log')
        plt.savefig(os.path.join(opts.output,'logistic_model_observation.pdf'),bbox_inches='tight')
        
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        mll,ml,mm,mh,mhh = np.percentile(total_models, [5,14,50,68,95], axis = 0)
        ax.fill_between(time_simulation, mll, mhh, facecolor = 'turquoise')
        ax.fill_between(time_simulation, ml, mh, facecolor = 'aquamarine')
        ax.plot(time_simulation, mm, color = 'k')
        ax.errorbar(time, data['totale_casi'], yerr=np.sqrt(data['totale_casi']), fmt='.', color='black',
                    ecolor='black', elinewidth=3, capsize=0)
        ax.set_xlim(-0.5,time_simulation[-1]+0.5)
        ax.set_xlabel('days')
        ax.set_ylabel('total number of infections')
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.set_yscale('log')
        plt.savefig(os.path.join(opts.output,'logistic_model_total.pdf'),bbox_inches='tight')
        
        p = np.column_stack((x['p'],x['N0'],x['fraction_of_positive'],x['E']))
        
        corner.corner(p, labels = ['p','N0','f','E'])
        plt.savefig(os.path.join(output,'corner.pdf'),bbox_inches='tight')
        print("Logistic growth model: prediction for tomorrow (median, 5th percentile, 90th percentile):  {0} in {1} {2}".format(ml[len(time)+1],mll[len(time)+1],mhh[len(time)+1]))

    if M.growth_model ==  "SIR":
        time_simulation = 1.0+np.linspace(0.0,60,60)
        observed_S = []
        observed_I = []
        observed_R = []
        deaths     = []
        for xi in x:
            solution = SIR(time_simulation,xi)
            Si = xi['fraction_of_positive']*solution[:,0]
            Ii = xi['fraction_of_positive']*solution[:,1]
            Ri = xi['fraction_of_positive']*xi['fraction_of_hospitalised']*solution[:,2]
            observed_S.append(Si)
            observed_I.append(Ii)
            observed_R.append(Ri)
#            deaths.append(xi['mortality_rate']*xi['fraction_of_hospitalised']*Ii)

        observed_S = np.array(observed_S)
        observed_I = np.array(observed_I)
        observed_R = np.array(observed_R)
#        deaths     = np.array(deaths)
        imax = np.argmax(observed_I, axis = 1)
        tmin,tm,tmax = np.percentile([time_simulation[i] for i in imax],[5,50,95])
        print("SIR growth model: prediction for peak infection:  {0} in {1} {2}".format(tm,tmin,tmax))
        Imax = np.max(observed_I, axis = 1)
        tmin,tm,tmax = np.percentile(Imax,[5,50,95])
        print("SIR growth model: prediction for peak number f infection:  {0} in {1} {2}".format(tm,tmin,tmax))
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
#        ax = compute_confidence_region(time_simulation, observed_S, ax, "blue", "cyan", "turquoise")
        ax = compute_confidence_region(time_simulation, observed_I, ax, "red", "magenta")
        ax = compute_confidence_region(time_simulation, observed_R, ax, "green", "aquamarine")
#        ax = compute_confidence_region(time_simulation, deaths, ax, "black","grey")
#        ax.plot(time_simulation,np.mean(observed_S,axis=0),'b',label='susceptible')
        ax.plot(time_simulation,np.median(observed_I,axis=0),'r',label='infected')
        ax.plot(time_simulation,np.median(observed_R,axis=0),'g',label='recovered')
#        ax.plot(time_simulation,np.median(deaths,axis=0),'k',label='deceased')

        ax.errorbar(time, data['dimessi_guariti'], yerr=np.sqrt(data['dimessi_guariti']), fmt='.', color='green',
                    ecolor='green', elinewidth=3, capsize=0)
        
        ax.errorbar(time, data['totale_casi'], yerr=np.sqrt(data['totale_casi']), fmt='.', color='red',
                    ecolor='red', elinewidth=3, capsize=0)
        
#        ax.errorbar(time, data['deceduti'], yerr=np.sqrt(data['deceduti']), fmt='.', color='black',
#                    ecolor='black', elinewidth=3, capsize=0)
                    
#        ax.set_xlim(0.5,time[-1]+1.5)
        ax.set_xlabel('days from 2020-02-24')
        ax.set_ylabel('number')
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.set_yscale('log')
        ax.grid(linestyle = 'dotted', color = 'grey',  alpha = 0.5, which='both')
#        ax.set_xscale('log')
        plt.legend(loc='lower right', fancybox=True)
        plt.savefig(os.path.join(opts.output,'SIR_model_observation.pdf'),bbox_inches='tight')
        
        l,m,u = np.percentile(observed_I,[5,50,95],axis=0)
        print("SIR growth model: prediction for tomorrow infections (median, 5th percentile, 95th percentile):  {0} in {1} {2}".format(m[len(time)+1],l[len(time)+1],u[len(time)+1]))
#
#        l,m,u = np.percentile(deaths,[5,50,95],axis=0)
#        print("SIR growth model: prediction for tomorrow deaths (median, 5th percentile, 90th percentile):  {0} in {1} {2}".format(m[len(time)+1],l[len(time)+1],u[len(time)+1]))
#
        time_simulation = np.linspace(0.0,365,365)
        observed_S = []
        observed_I = []
        observed_R = []
        deaths     = []
        population = []
        for xi in x:
            solution = SIR(time_simulation,xi)
            Si = solution[:,0]
            Ii = solution[:,1]
            Ri = solution[:,2]
            observed_S.append(Si)
            observed_I.append(Ii)
            observed_R.append(Ri)
            population.append(Si+Ii+Ri)
#            deaths.append(Ii*xi['mortality_rate']*xi['fraction_of_hospitalised'])

        observed_S = np.array(observed_S)
        observed_I = np.array(observed_I)
        observed_R = np.array(observed_R)
        deaths     = np.array(deaths)
        population = np.array(population)
        
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
#        ax = compute_confidence_region(time_simulation, deaths, ax, "black","grey")
        ax = compute_confidence_region(time_simulation, population, ax, "black","grey")
        ax = compute_confidence_region(time_simulation, observed_R, ax, "green", "aquamarine")
        ax = compute_confidence_region(time_simulation, observed_I, ax, "red", "magenta")
        ax = compute_confidence_region(time_simulation, observed_S, ax, "blue", "cyan")
        
#        ax.axhline(initial_population, linestyle='dashed', color='k')
        ax.plot(time_simulation,np.median(observed_S,axis=0),'b',label='susceptible')
        ax.plot(time_simulation,np.median(observed_I,axis=0),'r',label='infected')
        ax.plot(time_simulation,np.median(observed_R,axis=0),'g',label='recovered')
#        ax.plot(time_simulation,np.median(deaths,axis=0),'k',label='deceased')
        ax.plot(time_simulation,np.median(population,axis=0),'k',linestyle='dashed',label='population')
#        ax.axhline(initial_population, linestyle='dotted',color='k')
#        ax.set_xlim(-0.5,time[-1]+0.5)
        ax.set_xlabel('days from 2020-02-24')
        ax.set_ylabel('number')
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
#        ax.set_yscale('log')
#        ax.set_xscale('log')
        ax.grid(linestyle = 'dotted', color = 'grey',  alpha = 0.5, which='both')
        plt.legend(loc='center right', fancybox=True)
        plt.savefig(os.path.join(opts.output,'SIR_model_prediction.pdf'),bbox_inches='tight')
        
        
        p = np.column_stack((x['I0'],x['R0'],1./x['beta'],1./x['gamma'],x['fraction_of_positive'],x['fraction_of_hospitalised'],x['population'],x['beta']/x['gamma']))
        
        labels = [r'initial  number of infections',r'initial  number of recoveries',r'average infection time',r'average recovery time',r'fraction of infected',r'fraction of hospitalisation',r'total number of affected',r'infection strength']
        
        for i in range(p.shape[1]):
            l,m,h = np.percentile(p[:,i],[5,50,95])
            print(labels[i],"$%.2f_{-%.2f}^{+%.2f}$"%(m,m-l,h-m))
        
        corner.corner(p, labels = labels)
        plt.savefig(os.path.join(opts.output,'corner.pdf'),bbox_inches='tight')
