SWEEP_CONFIG = {
    "program": "experiment.py",
    "method": "grid",
    "metric": {
        "name": "best_kmeans_silhouette",
        "goal": "maximize"
    },
    "parameters": {
        "data_path": {
            "value": "/home/CAMPUS/d18129674/eeg_microstate_deepclustering_experiment/dataset"
        },
        "model_path": {
            "value": "/home/CAMPUS/d18129674/eeg_microstate_deepclustering_experiment/model_training_SLURM/artifacts/obj/md5/9a/5026db2161df3761cf884fc01cd090"
        },
        "batch_size": {
            "value": 40
        },
        "num_workers": {
            "value": 4
        },
        "latent_dim": {
            "value": 4
        },
        "dropout_conv": {
            "value": 0.23037448546520012
        },
        "dropout_fc": {
            "value": 0.42793365261644367
        },
        "leaky_relu_slope": {
            "value": 0.01
        },
        "learning_rate": {
            "value": 0.00692440691884721
        },
        "num_epochs": {
            "value": 100
        },
        "optimizer": {
            "value": "adam"
        },
        "scheduler_factor": {
            "value": 0.1
        },
        "scheduler_patience": {
            "value": 10
        },
        "weight_decay": {
            "value": 0.00000612680524937952
        },
        "kmeans_k_values": {
            "values": [[2, 3, 4], [4, 5, 6], [6, 9, 12]]
        },
        "hdbscan_min_sizes": {
            "values": [[3, 5, 10], [5, 10, 15], [10, 15, 20]]
        },
        "dbscan_min_samples": {
            "values": [[3, 5, 10], [5, 10, 15], [10, 15, 20]]
        },
        "dbscan_eps_values": {
            "values": [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]]
        }
    }
}