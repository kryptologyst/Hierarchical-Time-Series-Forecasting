#!/usr/bin/env python3
"""
Command-line interface for hierarchical time series forecasting.

This script provides a simple CLI for running forecasting experiments
from the command line.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append('src')

from src.main import HierarchicalForecastingPipeline
from src.utils import setup_logging, save_results


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Hierarchical Time Series Forecasting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python cli.py

  # Run with custom configuration
  python cli.py --config config/custom_config.yaml

  # Run and save results
  python cli.py --output results/experiment_results.json

  # Run with specific log level
  python cli.py --log-level DEBUG

  # Run Streamlit interface
  python cli.py --streamlit
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for results (JSON format)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: logs/cli.log)'
    )
    
    parser.add_argument(
        '--streamlit',
        action='store_true',
        help='Launch Streamlit web interface instead of running pipeline'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for Streamlit interface (default: 8501)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or 'logs/cli.log'
    setup_logging(level=args.log_level, log_file=log_file)
    
    if args.streamlit:
        # Launch Streamlit interface
        import subprocess
        import os
        
        print(f"Launching Streamlit interface on port {args.port}...")
        print("Open your browser to http://localhost:8501")
        
        try:
            subprocess.run([
                'streamlit', 'run', 'streamlit_app.py',
                '--server.port', str(args.port)
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to launch Streamlit: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("Streamlit not found. Please install it with: pip install streamlit")
            sys.exit(1)
    
    else:
        # Run forecasting pipeline
        try:
            print("Starting Hierarchical Time Series Forecasting Pipeline...")
            print(f"Configuration: {args.config}")
            print(f"Log level: {args.log_level}")
            
            # Check if config file exists
            if not Path(args.config).exists():
                print(f"Warning: Configuration file {args.config} not found. Using defaults.")
            
            # Initialize and run pipeline
            pipeline = HierarchicalForecastingPipeline(config_path=args.config)
            results = pipeline.run_complete_pipeline()
            
            # Save results if output path specified
            if args.output:
                print(f"Saving results to {args.output}...")
                save_results(results, args.output, format='json')
                print(f"Results saved successfully!")
            
            print("\nPipeline completed successfully!")
            
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
