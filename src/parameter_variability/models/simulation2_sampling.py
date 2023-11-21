"""
Toy Data Sampler

Using a SBML Model, declare a random distribution to generate solutions to the model as simulations.



"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import roadrunner
import os
import json
from config import parse_args

np.random.seed(2001)


class Sampler(object):
    def __init__(self, loc, scale, name, n, steps, model_path, save_dir, plot):
        self.loc = loc
        self.scale = scale
        self.name = name
        self.n = n
        self.steps = steps
        self.model_path = model_path
        print('-------------------- Sampler Initialized --------------------')
        self.model = roadrunner.RoadRunner(self.model_path)

        self.true_distribution = None
        self.thetas = None
        self.errors_distribution = None

        self.df_res = None

        self.define_distribution()
        self.draw_theta()
        self.sample_data()
        self.save_data(save_dir)
        if plot:
            self.plot()

    def define_distribution(self, true_dsn=stats.lognorm, errors_dsn=stats.halfnorm):
        true_dsn = true_dsn(loc=self.loc, s=self.scale)
        errors_dsn = errors_dsn()
        self.true_distribution = true_dsn
        self.errors_distribution = errors_dsn

        print('\nDistribution defined')

    def draw_theta(self):
        thetas = self.true_distribution.rvs(size=self.n)
        self.thetas = thetas
        print(f'\nThetas drawn: {self.thetas}')

    def sample_data(self):
        self.model.resetAll()
        sims = []
        num_thetas = self.thetas.shape[0]

        for i, theta in enumerate(self.thetas):
            self.model.setValue(self.name, theta)
            step_correction = self.steps + 1

            errors = self.errors_distribution.rvs(size=step_correction)
            sim = self.model.simulate(start=0, end=10, steps=self.steps)

            # Adding the errors: y_i = yhat_i + errors_i
            sim[:, 1:] = sim[:, 1:] + errors.reshape((step_correction, 1))
            df = pd.DataFrame(sim, columns=sim.colnames)
            df['patient_id'] = i
            sims.append(df)

            self.model.resetAll()

        if len(sims) > 1:
            df_res = pd.concat(sims, axis=1)

        else:
            df_res = sims[0]

        self.df_res = df_res

    def save_data(self, save_dir):
        res = self.df_res.to_dict(orient='split')
        res['thetas'] = self.thetas.tolist()
        res['n'] = self.n

        with open(save_dir+'/simulation2_samples.json', 'w') as file:
            file.truncate()
            json.dump(res, file, indent=4, sort_keys=True)

        print(f'\nResults saved in {save_dir}'
              f'\n-------------------- Simulation Finished! --------------------')
        print()

    # TODO: add plotting method to plot and summarize the data
    def plot(self):
        fig, ax = plt.subplots()
        self.df_res.plot(x='time', y=['[y_gut]', '[y_cent]', '[y_peri]'], ax=ax, style='.-')
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Concentration [mM]')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    args = parse_args()

    Sampler(loc=args.mean,
            scale=args.variance,
            name=args.parameter,
            n=args.n,
            steps=args.steps,
            model_path=args.model_path,
            save_dir=args.save_dir,
            plot=args.plot_data)
