import pandas as pd
import plotly.express as px
import country_converter as coco
import plotly.graph_objects as go


def create_un_region_choropleth(df, region_col, value_col, title="UN Regions Choropleth Map", 
                               color_scale="RdYlGn", projection_type='equirectangular', width=1000, height=600):
    
    # Initialize country converter
    cc = coco.CountryConverter()
    
    # Create a list to store country-value pairs
    country_values = []
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        region_name = row[region_col]
        value = row[value_col]
            
        # Get countries in this UN region
        countries_in_region = cc.data[cc.data['UNregion'] == region_name]['ISO3'].tolist()
        
        # Add each country with the region's value
        for country_iso3 in countries_in_region:
            country_values.append({
                'country_code': country_iso3,
                value_col: value,
                'region': region_name
            })
    
    choropleth_df = pd.DataFrame(country_values)
    
    # Create the choropleth map
    fig = px.choropleth(
        choropleth_df,
        locations='country_code',
        color=value_col,
        hover_name='country_code',
        hover_data={'region': True, value_col: True},
        color_continuous_scale=color_scale,
        title=title,
        labels={value_col: value_col, 'country_code': 'Country'}
    )
    
    # Update layout - SINGLE consolidated update
    fig.update_layout(
        width=width,
        height=height,
        title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showcountries=False,  # Remove country borders
            showsubunits=False,   # Remove state/province borders
            projection_type= projection_type
        )
    )
    
    # Remove borders between colored regions
    fig.update_traces(marker_line_width=0)
    
    return fig



def create_un_region_choropleth_deco(df, region_col, value_col, 
                               title="UN Regions Choropleth Map", 
                               color_scale="RdYlGn", 
                               projection_type='natural earth',
                               width=1200, 
                               height=700,
                               font_family="Arial, sans-serif",
                               bg_color="white",
                               ocean_color="#e8f4f8",
                               land_color="#f5f5f5",
                               show_legend_title=True,
                               reverse_scale=False):
    """
    Create a professional choropleth map from UN region data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing UN region names and corresponding values
    region_col : str
        Column name containing UN region names
    value_col : str
        Column name containing the values to plot
    title : str
        Title for the map
    color_scale : str
        Color scale for the choropleth (e.g., 'Viridis', 'Blues', 'RdYlGn')
    projection_type : str
        Map projection type (default: 'natural earth')
    width : int
        Width of the map (default: 1200)
    height : int
        Height of the map (default: 700)
    font_family : str
        Font family for all text (default: 'Arial, sans-serif')
    bg_color : str
        Background color (default: 'white')
    ocean_color : str
        Ocean color (default: '#e8f4f8')
    land_color : str
        Color for unpopulated land (default: '#f5f5f5')
    show_legend_title : bool
        Whether to show the legend title (default: True)
    reverse_scale : bool
        Reverse the color scale (default: False)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The professionally styled choropleth map figure
    """
    
    # Initialize country converter
    cc = coco.CountryConverter()
    
    # Create a list to store country-value pairs
    country_values = []
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        region_name = row[region_col]
        value = row[value_col]
        
        # Skip if missing data
        if pd.isna(region_name) or pd.isna(value):
            continue
            
        # Get countries in this UN region
        countries_in_region = cc.data[cc.data['UNregion'] == region_name]['ISO3'].tolist()
        
        # Add each country with the region's value
        for country_iso3 in countries_in_region:
            country_values.append({
                'country_code': country_iso3,
                value_col: value,
                'region': region_name
            })
    
    choropleth_df = pd.DataFrame(country_values)
    
    # Create the choropleth map
    fig = px.choropleth(
        choropleth_df,
        locations='country_code',
        color=value_col,
        hover_name='region',
        hover_data={
            'region': False,  # Don't show twice
            'country_code': True,
            value_col: ':.2f'  # Format to 2 decimal places
        },
        color_continuous_scale=color_scale,
        title=title,
        labels={value_col: value_col.replace('_', ' ').title(), 'country_code': 'Country'}
    )
    
    # Reverse color scale if requested
    if reverse_scale:
        fig.update_traces(reversescale=True)
    
    # Professional layout styling
    fig.update_layout(
        # Dimensions
        width=width,
        height=height,
        
        # Title styling
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top',
            'font': {
                'size': 24,
                'family': font_family,
                'color': '#2c3e50',
                'weight': 600
            }
        },
        
        # Font styling
        font=dict(
            family=font_family,
            size=12,
            color='#34495e'
        ),
        
        # Background
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        
        # Margins for better spacing
        margin=dict(l=20, r=20, t=80, b=20),
        
        # Color bar styling
        coloraxis_colorbar=dict(
            title=dict(
                text=value_col.replace('_', ' ').title() if show_legend_title else "",
                font=dict(size=14, family=font_family, color='#2c3e50')
            ),
            thickness=20,
            len=0.7,
            x=1.02,
            tickfont=dict(size=11, family=font_family),
            tickformat='.2f',
            outlinewidth=0
        ),
        
        # Geographic styling
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#95a5a6',
            coastlinewidth=0.5,
            showcountries=False,
            showsubunits=False,
            projection_type=projection_type,
            bgcolor=bg_color,
            showocean=True,
            oceancolor=ocean_color,
            showlakes=True,
            lakecolor=ocean_color,
            showland=True,
            landcolor=land_color
        )
    )
    
    # Clean borders between regions
    fig.update_traces(
        marker_line_width=0,
        marker_line_color='rgba(255,255,255,0.3)'
    )
    
    return fig 