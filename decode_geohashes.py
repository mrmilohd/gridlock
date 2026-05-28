import pandas as pd
import pygeohash as pgh

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv('train_cleaned.csv')
    
    # Extract unique geohashes to optimize processing speed
    unique_geohashes = df['geohash'].unique()
    print(f"Found {len(unique_geohashes)} unique geohashes. Decoding...")
    
    # Decode utilizing pygeohash
    records = []
    for gh in unique_geohashes:
        lat, lon, lat_err, lon_err = pgh.decode_exactly(gh)
        records.append({
            'geohash': gh,
            'latitude': lat,
            'longitude': lon,
            'latitude_err': lat_err,
            'longitude_err': lon_err
        })
        
    # Create an independent mapping structural dataframe
    mapping_df = pd.DataFrame(records)
    
    # Save the isolated mapping (useful as a lookup table)
    mapping_csv = 'geohash_mapping.csv'
    mapping_df.to_csv(mapping_csv, index=False)
    print(f"Saved unique geohash mapping to -> {mapping_csv}")
    
    # Merge the lat/lon coordinates back into the full dataset
    print("Merging coordinates back into the main dataset...")
    df_merged = df.merge(mapping_df, on='geohash', how='left')
    
    # Save the expanded dataset
    final_csv = 'train_expanded_geohash.csv'
    df_merged.to_csv(final_csv, index=False)
    print(f"Saved fully decoded dataset to -> {final_csv}")
    print("Decoding complete!")

if __name__ == "__main__":
    main()
