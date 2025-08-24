# main_.py
# This is the main script to run the power system simulation.
# Author: Power System Engineer

from simulation_core import IEEE39DynamicSim
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    logging.info("Initializing power system simulation...")
    sim = IEEE39DynamicSim(dt=0.02, t_end=30.0)
    logs = sim.run()
    sim.plot()

if __name__ == "__main__":
    main()
