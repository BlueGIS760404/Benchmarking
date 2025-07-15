import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import time
import psutil
import os
from shapely.geometry import box
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Function to get memory usage
def get_memory_usage():
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2

# Function to perform spatial join
def benchmark_spatial_join(point_file, polygon_file):
    """
    Benchmark a spatial join between point and polygon shapefiles.
    Args:
        point_file (str): Path to point shapefile.
        polygon_file (str): Path to polygon shapefile.
    Returns:
        dict: Results including time and memory usage.
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Load data
    point_gdf = gpd.read_file(point_file)
    polygon_gdf = gpd.read_file(polygon_file)
    
    # Perform spatial join
    joined_gdf = gpd.sjoin(point_gdf, polygon_gdf, how="inner", predicate="within")
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    return {
        "operation": "spatial_join",
        "time_seconds": end_time - start_time,
        "memory_mb": end_memory - start_memory,
        "output_rows": len(joined_gdf)
    }

# Function to perform raster-vector alignment
def benchmark_raster_vector_alignment(raster_file, polygon_file):
    """
    Benchmark extracting raster values within polygon boundaries.
    Args:
        raster_file (str): Path to GeoTIFF raster file.
        polygon_file (str): Path to polygon shapefile.
    Returns:
        dict: Results including time, memory usage, and pixel count.
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Load data
    polygon_gdf = gpd.read_file(polygon_file)
    pixel_count = 0
    
    with rasterio.open(raster_file) as src:
        raster_crs = src.crs
        # Reproject polygon to match raster CRS if needed
        if polygon_gdf.crs != raster_crs:
            polygon_gdf = polygon_gdf.to_crs(raster_crs)
        
        # Extract raster values for each polygon
        for _, row in polygon_gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = mask(src, geom, crop=True, nodata=0)
                pixel_count += np.count_nonzero(out_image)
            except ValueError:
                continue
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    return {
        "operation": "raster_vector_alignment",
        "time_seconds": end_time - start_time,
        "memory_mb": end_memory - start_memory,
        "pixel_count": pixel_count
    }

# Function to benchmark reprojection
def benchmark_reprojection(vector_file, target_crs="EPSG:4326"):
    """
    Benchmark reprojecting a vector dataset to a target CRS.
    Args:
        vector_file (str): Path to vector shapefile.
        target_crs (str): Target CRS (e.g., EPSG code).
    Returns:
        dict: Results including time and memory usage.
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Load vector data
    gdf = gpd.read_file(vector_file)
    
    # Reproject to target CRS
    reprojected_gdf = gdf.to_crs(target_crs)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    return {
        "operation": "reprojection",
        "time_seconds": end_time - start_time,
        "memory_mb": end_memory - start_memory,
        "output_crs": target_crs
    }

def main():
    # Example paths (update with your dataset paths)
    raster_file = "raster.tif"
    polygon_file = "polygon.shp"
    point_file = "point.shp"
    target_crs = "EPSG:4326"  # WGS84
    
    # Benchmark spatial join
    print("Running spatial join benchmark...")
    spatial_join_results = benchmark_spatial_join(point_file, polygon_file)
    
    # Benchmark raster-vector alignment
    print("Running raster-vector alignment benchmark...")
    raster_vector_results = benchmark_raster_vector_alignment(raster_file, polygon_file)
    
    # Benchmark reprojection
    print("Running reprojection benchmark...")
    reprojection_results = benchmark_reprojection(polygon_file, target_crs)
    
    # Print results
    print("\nBenchmark Results:")
    print("\n1. Spatial Join:")
    print(f"Time: {spatial_join_results['time_seconds']:.2f} seconds")
    print(f"Memory Usage: {spatial_join_results['memory_mb']:.2f} MB")
    print(f"Output Rows: {spatial_join_results['output_rows']}")
    
    print("\n2. Raster-Vector Alignment:")
    print(f"Time: {raster_vector_results['time_seconds']:.2f} seconds")
    print(f"Memory Usage: {raster_vector_results['memory_mb']:.2f} MB")
    print(f"Pixel Count: {raster_vector_results['pixel_count']}")
    
    print("\n3. Reprojection:")
    print(f"Time: {reprojection_results['time_seconds']:.2f} seconds")
    print(f"Memory Usage: {reprojection_results['memory_mb']:.2f} MB")
    print(f"Output CRS: {reprojection_results['output_crs']}")

if __name__ == "__main__":
    main()
