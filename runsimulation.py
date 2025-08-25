import logging
from simulation import IEEE39QSTS

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation.log')
    ]
)

def get_user_inputs():
    """Interactive input function for load and demand parameters"""
    print("\n" + "="*50)
    print("IEEE-39 Power System Simulation Setup")
    print("="*50)
    
    try:
        # Load scaling factor
        load_factor = float(input("Enter load scaling factor (1.0 = base load, 1.1 = 10% increase): ") or "1.0")
        
        # Load step disturbance
        step_enabled = input("Enable load step disturbance? (y/n): ").lower() == 'y'
        step_time = 5.0
        step_magnitude = 1.0
        
        if step_enabled:
            step_time = float(input("Step disturbance time (seconds, default=5.0): ") or "5.0")
            step_magnitude = float(input("Step magnitude factor (1.1 = 10% increase): ") or "1.1")
        
        # Simulation parameters
        dt = float(input("Time step (seconds, default=0.05): ") or "0.05")
        t_end = float(input("Simulation end time (seconds, default=30.0): ") or "30.0")
        beta = float(input("AGC beta parameter (default=20.0): ") or "20.0")
        
        # Convergence parameters
        max_iterations = int(input("Maximum AGC/AVR iterations per step (default=10): ") or "10")
        freq_tolerance = float(input("Frequency convergence tolerance (Hz, default=0.001): ") or "0.001")
        
        return {
            'load_factor': load_factor,
            'step_enabled': step_enabled,
            'step_time': step_time,
            'step_magnitude': step_magnitude,
            'dt': dt,
            't_end': t_end,
            'beta': beta,
            'max_iterations': max_iterations,
            'freq_tolerance': freq_tolerance
        }
        
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
        logging.info("Using default values...")
        return {
            'load_factor': 1.0,
            'step_enabled': True,
            'step_time': 5.0,
            'step_magnitude': 1.1,
            'dt': 0.05,
            't_end': 30.0,
            'beta': 20.0,
            'max_iterations': 10,
            'freq_tolerance': 0.001
        }

if __name__ == "__main__":
    # Get user inputs
    params = get_user_inputs()
    
    logging.info("="*60)
    logging.info("STARTING IEEE-39 POWER SYSTEM SIMULATION")
    logging.info("="*60)
    logging.info(f"Configuration:")
    logging.info(f"  Load scaling factor: {params['load_factor']}")
    logging.info(f"  Step disturbance: {'Enabled' if params['step_enabled'] else 'Disabled'}")
    if params['step_enabled']:
        logging.info(f"    Step time: {params['step_time']}s")
        logging.info(f"    Step magnitude: {params['step_magnitude']}")
    logging.info(f"  Time step: {params['dt']}s")
    logging.info(f"  End time: {params['t_end']}s")
    logging.info(f"  AGC beta: {params['beta']}")
    logging.info(f"  Max iterations: {params['max_iterations']}")
    logging.info(f"  Frequency tolerance: {params['freq_tolerance']} Hz")
    logging.info("="*60)
    
    # Create and configure simulation
    sim = IEEE39QSTS(
        dt=params['dt'], 
        t_end=params['t_end'], 
        beta=params['beta']
    )
    
    # Apply initial load scaling
    sim.load_factor = params['load_factor']
    logging.info(f"Applied initial load scaling: {params['load_factor']}")
    
    # Configure step disturbance
    if params['step_enabled']:
        sim.step_disturbance = {
            'enabled': True,
            'time': params['step_time'],
            'magnitude': params['step_magnitude']
        }
        logging.info(f"Step disturbance configured: {params['step_magnitude']}x at t={params['step_time']}s")
    
    # Configure convergence parameters
    sim.max_control_iterations = params['max_iterations']
    sim.freq_tolerance = params['freq_tolerance']
    
    logging.info("Starting simulation...")
    try:
        logs = sim.run()
        
        logging.info("="*60)
        logging.info("SIMULATION COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        logging.info("Final Results:")
        logging.info(f"  Final frequency: {sim.logs['freq_Hz'][-1]:.4f} Hz")
        logging.info(f"  Final generation: {sim.logs['sumGen_MW'][-1]:.2f} MW")
        logging.info(f"  Final load: {sim.logs['sumLoad_MW'][-1]:.2f} MW")
        logging.info(f"  Final load factor: {sim.logs['load_factor'][-1]:.4f}")
        
        # Control system analysis
        freq_deviation = abs(sim.logs['freq_Hz'][-1] - 60.0)
        logging.info(f"  Frequency deviation: {freq_deviation:.4f} Hz")
        logging.info(f"  Power balance: {sim.logs['sumGen_MW'][-1] - sim.logs['sumLoad_MW'][-1]:.2f} MW")
        
        if 'control_iterations' in sim.logs:
            avg_iterations = sum(sim.logs['control_iterations']) / len(sim.logs['control_iterations'])
            max_iterations = max(sim.logs['control_iterations'])
            logging.info(f"  Average control iterations per step: {avg_iterations:.1f}")
            logging.info(f"  Maximum control iterations: {max_iterations}")
        
        logging.info("="*60)
        
        # Generate plots
        logging.info("Generating plots...")
        sim.plot()
        
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        logging.error("Check simulation.log for detailed error information")
        raise
    
    logging.info("Simulation and analysis complete.")