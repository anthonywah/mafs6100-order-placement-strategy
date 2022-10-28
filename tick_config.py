TICK_CONFIG_LIST = [
    {
        'start': 0.01,
        'tick': {
            'stock': 0.01,
            'etf': 0.01
        },
    },
    {
        'start': 5,
        'tick': {
            'stock': 0.01,
            'etf': 0.01
        },
    },
    {
        'start': 10,
        'tick': {
            'stock': 0.05,
            'etf': 0.01
        },
    },
    {
        'start': 50,
        'tick': {
            'stock': 0.1,
            'etf': 0.05
        },
    },
    {
        'start': 100,
        'tick': {
            'stock': 0.5,
            'etf': 0.05
        },
    },
    {
        'start': 150,
        'tick': {
            'stock': 0.5,
            'etf': 0.05
        },
    },
    {
        'start': 500,
        'tick': {
            'stock': 1.,
            'etf': 0.05
        },
    },
    {
        'start': 1000,
        'tick': {
            'stock': 0.1,
            'etf': 0.05
        },
    }
]

SEC_TYPE_DICT = {
    '0050': 'etf',
    '2330': 'stock'
}