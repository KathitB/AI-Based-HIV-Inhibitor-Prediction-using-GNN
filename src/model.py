import deepchem as dc

def build_gnn_model(mode='classification',
                    n_tasks=1,
                    model_dir='model_checkpoints',
                    learning_rate=0.001,
                    batch_size=32,
                    dropout=0.2):
    """
    
    Parameters of the model:
    - mode: 'classification' or 'regression'
    - n_tasks: number of output tasks (1 for binary classification)
    - model_dir: where to save model checkpoints
    - learning_rate: optimizer learning rate
    - batch_size: training batch size
    - dropout: dropout rate for regularization

    """

    model = dc.models.GraphConvModel(
        n_tasks=n_tasks,
        mode=mode,
        model_dir=model_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout=dropout
    )
    
    return model
