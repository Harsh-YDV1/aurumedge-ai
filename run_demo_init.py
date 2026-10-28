from src.demo_data import generate_synthetic_gold_dataset

if __name__ == "__main__":
    df = generate_synthetic_gold_dataset()
    print(f"Success: Generated {len(df)} rows across symbols: {df['symbol'].unique().tolist()}")
