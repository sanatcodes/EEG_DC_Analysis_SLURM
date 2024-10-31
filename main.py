import wandb
from config.sweep_config import SWEEP_CONFIG
from dataloader import TopomapDataset
from utils.experiment import run_clustering_analysis


def cluster(config=None):
    with wandb.init(config=config) as run:
        # Access all hyperparameter values through wandb.config
        config = wandb.config
        
        # Run the clustering analysis
        results = run_clustering_analysis(config)

        wandb.log(results)
        # Additional processing of results if needed
        wandb.finish()

def main():
    """Main function that handles both sweep initialization and agent execution"""
    # Login to wandb
    wandb.login()
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project="MS-deep-clustering-analysis"
    )
    
    print(f"Initialized sweep with ID: {sweep_id}")
    
    # Start the sweep agent
    wandb.agent(
        sweep_id,
        function=cluster,
        count=20  
    )

if __name__ == "__main__":
    main()