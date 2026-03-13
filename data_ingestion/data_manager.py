import sys
import os
import argparse

# Add the root directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.shared.dataset_handler import generate_validation_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Automated Data Generation Tool for VRP-RL')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes (customers + depot)')
    parser.add_argument('--num_instances', type=int, default=1000, help='Number of dataset instances')
    parser.add_argument('--num_vehicles', type=int, default=1, help='Number of vehicles')
    parser.add_argument('--capacity', type=float, default=50.0, help='Vehicle capacity')
    parser.add_argument('--weather_dim', type=int, default=3, help='Weather dimensions')
    return parser.parse_args()

def main():
    args = parse_args()
    save_path = os.path.join(
        os.path.dirname(__file__), 
        f'validation_dataset_n{args.num_nodes}.pkl'
    )
    
    print("=====================================")
    print("VRP-RL Data Automation Script")
    print("=====================================")
    print(f"Nodes: {args.num_nodes}")
    print(f"Instances: {args.num_instances}")
    print(f"Saving to: {save_path}")
    
    generate_validation_dataset(
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        num_vehicles=args.num_vehicles,
        capacity=args.capacity,
        weather_dim=args.weather_dim,
        save_path=save_path,
        device='cpu' # Generate on CPU for sharing
    )

if __name__ == '__main__':
    main()
