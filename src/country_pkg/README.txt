USE BELOW FOR CREATING COUNTRY CLASSES

def generate_countries_from_multiple_csvs(
    country_csv_path,
    cr_box_csv_path=None,
    essential_workers_csv_path=None,
    baghouse_csv_path=None
):
    # ---------------- Main country CSV ----------------
    df = pd.read_csv(country_csv_path, encoding='cp1252')
    required_cols = ['ISO-3', 'Country Name']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must have a column named '{col}'")
    
    # ---------------- CR Box CSV ----------------
    cr_box_df = None
    if cr_box_csv_path:
        cr_box_df = pd.read_csv(cr_box_csv_path, encoding='cp1252')
        if "Country" not in cr_box_df.columns:
            raise ValueError("CR Box CSV must have a 'Country' column")
        cr_box_df["Country"] = cr_box_df["Country"].apply(Country._cc.convert, to="name_short")
    
    # ---------------- Essential / Indoor worker CSV ----------------
    essential_workers_df = None
    if essential_workers_csv_path:
        essential_workers_df = pd.read_csv(essential_workers_csv_path, encoding='cp1252')
        required_essential_cols = ['Country Name', 'Country Code']
        for col in required_essential_cols:
            if col not in essential_workers_df.columns:
                raise ValueError(f"Essential Workers CSV must have a column named '{col}'")
    
    # ---------------- Baghouse Airflow CSV ----------------
    baghouse_df = None
    if baghouse_csv_path:
        baghouse_df = pd.read_csv(baghouse_csv_path, encoding='cp1252')
        required_baghouse_cols = ['Country', 'Operating MW']
        for col in required_baghouse_cols:
            if col not in baghouse_df.columns:
                raise ValueError(f"Baghouse CSV must have a column named '{col}'")
        # Standardize country names
        baghouse_df["Country"] = baghouse_df["Country"].apply(Country._cc.convert, to="name_short")
    
    countries = {}
    
    for _, row in df.iterrows():
        iso_code = row['ISO-3']
        country_name = row['Country Name']
        
        # Create country object
        c = Country(name=iso_code)
        c.properties['ISO-3'] = iso_code
        
        # ---------------- Merge CR Box properties ----------------
        if cr_box_df is not None:
            standardized_name = Country._cc.convert(country_name, to="name_short")
            cr_row = cr_box_df[cr_box_df["Country"] == standardized_name]
            if not cr_row.empty:
                for col in cr_row.columns:
                    if col != "Country":
                        c.properties[col] = cr_row.iloc[0][col]
            else:
                for col in cr_box_df.columns:
                    if col != "Country":
                        c.properties[col] = 0
        
        # ---------------- Merge Essential / Indoor worker properties ----------------
        if essential_workers_df is not None:
            essential_workers_row = essential_workers_df[essential_workers_df["Country Code"] == iso_code]
            if not essential_workers_row.empty:
                for col in essential_workers_row.columns:
                    if col not in ["Country Code", "Country Name"]:
                        c.properties[col] = essential_workers_row.iloc[0][col]
        
        # ---------------- Merge Baghouse properties ----------------
        if baghouse_df is not None:
            standardized_name = Country._cc.convert(country_name, to="name_short")
            baghouse_row = baghouse_df[baghouse_df["Country"] == standardized_name]
            if not baghouse_row.empty:
                c.properties["Baghouse Operating MW"] = baghouse_row.iloc[0]["Operating MW"]
            else:
                c.properties["Baghouse Operating MW"] = 0
        
        countries[iso_code] = c
    
    return countries
