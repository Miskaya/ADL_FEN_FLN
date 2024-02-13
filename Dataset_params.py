dataset_params = {
    'WISDM': {
        'd_model': 192,
        'nhead': 4,
        'dim_feedforward': 960,
        'num_encoder_layers': 2,
        'fen_lr': 0.001,
        'fln_lr': 0.001,
        'epoch': 20
    },
    'WARD': {
        'd_model': 1600,
        'nhead': 8,
        'dim_feedforward': 4800,
        'num_encoder_layers': 6,
        'fen_lr': 0.001,
        'fln_lr': 0.0001,
        'epoch': 30
    },
    'USCHAD': {
        'd_model': 384,
        'nhead': 16,
        'dim_feedforward': 2304,
        'num_encoder_layers': 6,
        'fen_lr': 0.001,
        'fln_lr': 0.0001,
        'epoch': 100
    },    
    'UniMiBSHAR': {
        'd_model': 192,
        'nhead': 2,
        'dim_feedforward': 576,
        'num_encoder_layers': 8,
        'fen_lr': 0.0001,
        'fln_lr': 0.0001,
        'epoch': 250
    },
    'Realworld': {
        'd_model': 384,
        'nhead': 4,
        'dim_feedforward': 2304,
        'num_encoder_layers': 2,
        'fen_lr': 0.0001,
        'fln_lr': 0.0001,
        'epoch': 30
    },
    'Realdisp': {
        'd_model': 3456,
        'nhead': 16,
        'dim_feedforward': 13824,
        'num_encoder_layers': 2,
        'fen_lr': 0.001,
        'fln_lr': 0.0001,
        'epoch': 30
    },    
    'PAMAP2': {
        'd_model': 1152,
        'nhead': 4,
        'dim_feedforward': 2304,
        'num_encoder_layers': 6,
        'fen_lr': 0.001,
        'fln_lr': 0.0001,
        'epoch': 30
    },
    'Motionsense': {
        'd_model': 384,
        'nhead': 4,
        'dim_feedforward': 1920,
        'num_encoder_layers': 1,
        'fen_lr': 0.001,
        'fln_lr': 0.001,
        'epoch': 30
    },
    'Mhealth': {
        'd_model': 960,
        'nhead': 4,
        'dim_feedforward': 5760,
        'num_encoder_layers': 3,
        'fen_lr': 0.0001,
        'fln_lr': 0.0001,
        'epoch': 20
    },    
    'HHAR': {
        'd_model': 384,
        'nhead': 4,
        'dim_feedforward': 1536,
        'num_encoder_layers': 8,
        'fen_lr': 0.001,
        'fln_lr': 0.0001,
        'epoch': 50
    },
    'DSADS': {
        'd_model': 1920,
        'nhead': 2,
        'dim_feedforward': 7680,
        'num_encoder_layers': 1,
        'fen_lr': 0.0001,
        'fln_lr': 0.0001,
        'epoch': 30
    },
}