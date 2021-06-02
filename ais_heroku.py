# https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku
# https://hidden-harbor-66669.herokuapp.com/ | https://git.heroku.com/hidden-harbor-66669.git

# Main
import os
import sys
import math
import itertools
import numpy as np
import pandas as pd
# from streamlit_helpers import *
import warnings

# Geometry
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
import geopandas as gpd

# Mapping & Colors
# from ipyleaflet import Map, basemaps, basemap_to_tiles, Polygon, Polyline, Circle, Marker, Popup, LayerGroup, LayersControl, LegendControl
import streamlit as st
from streamlit_folium import folium_static
import folium
from folium.vector_layers import Polygon, PolyLine, CircleMarker  # path_options
from folium.map import LayerControl  # FeatureGroup, FitBounds, Layer, Marker, Tooltip
from folium.raster_layers import TileLayer
# from folium.plugins import MarkerCluster
from palettable.cartocolors.qualitative import Safe_10, Pastel_10, Antique_10, Prism_10, Bold_10
from palettable.colorbrewer.qualitative import Dark2_8
from palettable.cmocean.sequential import Algae_5, Amp_5, Dense_5_r, Matter_5_r, Solar_5, Speed_5, Tempo_5_r, Turbid_5
# cmocean: {'Algae_5': 'green', 'Amp_5': 'salmon-red', 'Dense_5': 'berry', 'Matter_5': 'earth-berry', 
# 'Solar_5': 'brown-yellow', 'Speed_5': 'bold green', 'Tempo_5': 'sea-blues', 'Turbid_5': 'brown'}
from palettable.colorbrewer.sequential import Blues_5, Oranges_5, YlOrBr_5, Purples_5_r, PuRd_5, Reds_5_r, YlGn_5, OrRd_5_r, GnBu_5, Greens_5, Greys_5, YlOrRd_5_r
from palettable.scientific.sequential import Bamako_5, Batlow_5, LaJolla_5, LaPaz_5, Nuuk_5

# Program Settings
CODE_DIR = './'
DATA_DIR = CODE_DIR + 'data/'
IMAGE_DIR = CODE_DIR + 'images/'
SHAPEFILE_DIR = CODE_DIR + 'shapefiles/'
os.chdir(CODE_DIR)

# AIS for the whole US, filtered to a Bounding Box including Puget Sound and the waters upto the 49th parallel.
pugetCsv = DATA_DIR + 'AIS_PUGET_2020_01_06.csv'
bufferWidth = 0.03  # Euclidean buffer - equivalent to 2 nautical miles around a shipping lane
warnings.filterwarnings("ignore")
# pd.options.mode.chained_assignment = None

# Helper function to flatten nested lists
flatten = itertools.chain.from_iterable
#-------------------------------------------------------------
# Seattle
laneLabel = "sl43"
start = pd.Timestamp('2020-01-06T13:00:00', tz='UTC')
end = pd.Timestamp('2020-01-06T15:00:00', tz='UTC')
tracksDf = pd.DataFrame()
#-------------------------------------------------------------
# Read data into a pandas DataFrame (It has both motion data and metadata)
col_names = ['mmsi', 'base_dt', 'lat', 'lon', 'sog', 'cog', 'heading', 'vessel_name', 'imo', 
             'call_sign', 'vessel_type', 'status', 'length', 'width', 'draft', 'cargo', 'transceiver_class', 'vessel_category']
types = ['object', 'object', 'float64', 'float64', 'float64', 'float64', 'float64', 'object', 'object', 
         'object', 'Int64', 'object', 'float64', 'float64', 'float64', 'Int64', 'object', 'object']
col_dtypes = dict(zip(col_names, types))

df = pd.read_csv(pugetCsv, header=0, names=col_names, dtype=col_dtypes)

# When parsing the datetime string, pandas assumes it is in local time. Make sure the timezone is specified as UTC.
df['base_dt'] = pd.to_datetime(df['base_dt'], utc=True, infer_datetime_format=True)
df['local_dt'] = df['base_dt'].dt.tz_convert('US/Pacific')
# For leaflet, folium & Raster Tiles, coordinates need to be in (lat, lon) order
df['coordinates'] = [list(tup) for tup in zip(df.lat.round(5), df.lon.round(5) )]
#------------------------------------------------------------------
@st.cache
def split_df(df):
    """
    Splits the AIS dataframe into two. 
    MotionDf contains the dynamic columns. 
    MetadataDf cotains the static metadata columns
    """
    def firstNonNull(s):
        """
        Aggregation Function used to get metadata from a grouped DataFrame
        Returns the first non-null element in a series.
        """
        idx = s.first_valid_index()
        first_valid_value = s.loc[idx] if idx is not None else None
        return first_valid_value

    motionDf = df[['mmsi', 'vessel_category', 'base_dt', 'local_dt', 'coordinates', 'sog', 'cog', 'heading', 'status', 'draft', 'lat', 'lon']].sort_values('local_dt', ascending=True)
    staticGby = df[['mmsi', 'vessel_category', 'vessel_name', 'length', 'width', 'transceiver_class', 'imo', 'call_sign', 'vessel_type', 'cargo']].groupby(by='mmsi', axis=0)
    metadataDf = staticGby.agg(func=firstNonNull).reset_index()  # 'mmsi' is now a column, not the index
    metadataDf['vessel_name'] = metadataDf['vessel_name'].apply(lambda vn: "<NA>" if vn is None else vn)
    return (motionDf, metadataDf)

motion_puget_pdf, metadata_puget_pdf = split_df(df)
#-------------------------------------------------------------------
# Lane Labels & Relationships: Lane pairs (in opposite directions), and the TSS between them.
# The Quadrant direction of a lane is used to calculate the Lane Bearing

# Lane Pairs
d = {'sl77':'sl78', 'sl43':'sl47', 'sl42':'sl52', 'sl44':'sl52', 'sl40':'sl41', 'sl45':'sl46', 'sl50':'sl51', 'sl69':'sl71', 'sl63':'sl65', 'sl66':'sl74', 
             'sl59':'sl60', 'sl64':'sl67', 'sl53':'sl54', 'sl55':'sl54', 'sl72':'sl73', 'sl49':'sl56', 'sl68':'sl70', 'sl61':'sl62', 'sl57':'sl58', 'sl48':'sl52'}

lane_singles = ['sl00', 'sl01', 'sl02']

# Traffic Separation Schemes, and the shipping lanes on either side
t = {'ts08':['sl77', 'sl78'], 'ts06':['sl43', 'sl47'], 'ts07':['sl42', 'sl52'], 'ts09':['sl44', 'sl52'], 'ts05':['sl40', 'sl41'], 'ts04':['sl45', 'sl46'], 
     'ts03':['sl50', 'sl51'], 'ts22':['sl66', 'sl74'], 'ts21':['sl65', 'sl63'], 'ts14':['sl69', 'sl71'], 'ts20':['sl59', 'sl60'], 'ts15':['sl64', 'sl67'], 
     'ts12':['sl54', 'sl55'], 'ts17':['sl53', 'sl54'], 'ts11':['sl49', 'sl56'], 'ts19':['sl72', 'sl73'], 'ts18':['sl68', 'sl70'], 'ts16':['sl61', 'sl62'], 
     'ts13':['sl48', 'sl52'], 'ts10':['sl57', 'sl58']}

# This is used to calculate the lane bearing
quadrant = {'sl77':2, 'sl78':4, 'sl43':2, 'sl47':4, 'sl42':1, 'sl52':4, 'sl44':2, 'sl40':2, 'sl41':4, 'sl45':4,
            'sl46':2, 'sl50':2, 'sl51':4, 'sl69':1, 'sl71':3, 'sl63':2, 'sl65':4, 'sl66':4, 'sl74':2, 'sl59':2, 
            'sl60':1, 'sl64':1, 'sl67':3, 'sl53':2, 'sl54':4, 'sl55':4, 'sl72':1, 'sl73':3, 'sl49':4, 'sl56':2, 
            'sl68':4, 'sl70':2, 'sl61':3, 'sl62':1, 'sl57':2, 'sl58':4, 'sl48':2, 'sl75':1, 'sl76':3, 'sl00':2, 'sl01':1, 'sl02':3}

# For each lane (key), get the lane on the opposite side (value)
lane_pairs = d.copy()  # dictionary
for k,v in d.items():
    if v not in lane_pairs.keys():
        lane_pairs.update({v:k})
    else:
        if type(lane_pairs[v]) is str:
            lane_pairs.update({v: ([lane_pairs[v]] + [k])})
        elif type(lane_pairs[v]) is list:
            lane_pairs.update({v: (lane_pairs[v] + [k])})
            
# For each Traffic Separation Scheme (key), get the Shipping Lanes (value_list) on both sides.
tss_pairs = dict()  # empty dictionary
for k,v in t.items():
    for i in range(len(v)):
        if v[i] not in tss_pairs.keys():
            tss_pairs.update({v[i]: k})
        else:
            if type(tss_pairs[v[i]]) is str:
                tss_pairs.update({v[i]: ([tss_pairs[v[i]]] + [k])})
            elif type(tss_pairs[v[i]]) is list:
                tss_pairs.update({v[i]: (tss_pairs[v[i]] + [k])})
                
# Standard formula for calculating the bearing between two points.
# Bearing is measured with True North = 0
def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    return brng

def lane_bearing(laneLabel):
    """
    Calculates the bearing of a shipping lane (instead of the Euclidean angle)
    https://stackoverflow.com/questions/54873868/python-calculate-bearing-between-two-lat-long
    """
    bounds = puget_shiplanes_gdf.loc[puget_shiplanes_gdf['label']==laneLabel, 'bounds'].iloc[0]
    if quadrant[laneLabel] == 1:
        bearing = get_bearing(bounds[1], bounds[0], bounds[3], bounds[2])
    elif quadrant[laneLabel] == 2:
        bearing = get_bearing(bounds[1], bounds[2], bounds[3], bounds[0])
    elif quadrant[laneLabel] == 3:
        bearing = get_bearing(bounds[3], bounds[2], bounds[1], bounds[0])
    elif quadrant[laneLabel] == 4:
        bearing = get_bearing(bounds[3], bounds[0], bounds[1], bounds[2])

    return bearing
#-----------------------------------------------------------------------
# Lane Geometry from the shapefile

puget_shiplanes_gdf = gpd.read_file(SHAPEFILE_DIR + 'shiplanes_puget.shp')

# puget_shiplanes_gdf.crs is "EPSG:4326" (WGS 84), but folium uses default="EPSG:3857" (pseudo-Mercator)
# puget_shiplanes_gdf['geometry'] = puget_shiplanes_gdf['geometry'].to_crs("EPSG:3857")

# For each lane polygon, get the centroid. Then assign a label that uniquely identifies the polygon.
puget_shiplanes_gdf['centroid'] = puget_shiplanes_gdf['geometry'].centroid
puget_shiplanes_gdf['bounds'] = puget_shiplanes_gdf['geometry'].apply(lambda geom: geom.bounds)  # (minx, miny, maxx, maxy) tuple
puget_shiplanes_gdf['bufferBbox'] = puget_shiplanes_gdf['geometry'].apply(lambda geom: geom.buffer(distance=bufferWidth).bounds) # (minx, miny, maxx, maxy) tuple

numLabels = puget_shiplanes_gdf.shape[0]
centroid_num = [str(i).zfill(2) for i in list(range(numLabels))]
labels = [('pa'+centroid_num[i]) if puget_shiplanes_gdf['OBJL'].iloc[i] == '96' \
          else ('ts'+centroid_num[i]) if puget_shiplanes_gdf['OBJL'].iloc[i] == '150' \
          else ('sl'+centroid_num[i]) for i in range(numLabels)]  # '152', '148'
puget_shiplanes_gdf['label'] = labels

puget_shiplanes_gdf['bearing'] = puget_shiplanes_gdf['label'].apply(lambda l: lane_bearing(l) if l[:2]=='sl' else np.nan)

# Union polygon of all shipping lanes, traffic separation schemes & precautionary areas - using shapely.ops.unary_union
polygons = puget_shiplanes_gdf['geometry']
unionOfLanes_gs =  gpd.GeoSeries(unary_union(polygons))
unionOfLanesPolygon = unionOfLanes_gs.iloc[0]
#------------------------------------------------------------------------
# Mapping & Colors

colors = Safe_10.hex_colors + Pastel_10.hex_colors + Antique_10.hex_colors + Prism_10.hex_colors + Bold_10.hex_colors
darkColors = Dark2_8.hex_colors

# Track Color for each Vessel Category
vcat_colors = {'Passenger': colors[33], 'Fishing': colors[31], 'TowFrontSide': colors[36], 'TowLargeAstern': colors[37], 'Cargo': colors[38], 'Tanker': colors[35], 
               'WingInGround': colors[30], 'Sailing': colors[32], 'PleasureCraft': colors[34], 'Other': colors[20], 'Unknown': colors[21], 'Dredging': colors[22], 
               'Diving': colors[23], 'Military': colors[24], 'HighSpeedCraft': colors[25], 'PilotVessel': colors[26], 'SearchAndRescue': colors[27], 'Tug': colors[28], 
               'PortTender': colors[29], 'AntiPollution': colors[0], 'LawEnforcement': colors[4], 'LocalVessel': colors[1], 'MedicalTransport': colors[9], 
               'SpecialCraft': colors[2], 'PublicVessel': colors[5], 'TankBarge': colors[11], 'Research': colors[19], 'Reserved': colors[12], '107': colors[13], 
               '207': colors[14], '255': colors[15]}

vcat_palette = {'Passenger': Algae_5, 'Fishing': Blues_5, 'TowFrontSide': Oranges_5, 'TowLargeAstern': Solar_5, 'Cargo': Amp_5, 'Tanker': YlOrBr_5, 
               'WingInGround': Bamako_5, 'Sailing': Tempo_5_r, 'PleasureCraft': YlGn_5, 'Other': PuRd_5, 'Unknown': Speed_5, 'Dredging': LaJolla_5, 
               'Diving': Matter_5_r, 'Military': Purples_5_r, 'HighSpeedCraft': LaPaz_5, 'PilotVessel': Reds_5_r, 'SearchAndRescue': Dense_5_r, 'Tug': Turbid_5, 
               'PortTender': OrRd_5_r, 'AntiPollution': Greens_5, 'LawEnforcement': Batlow_5, 'LocalVessel': GnBu_5, 'MedicalTransport': Reds_5_r, 
               'SpecialCraft': Nuuk_5, 'PublicVessel': Greens_5, 'TankBarge': LaJolla_5, 'Research': Tempo_5_r, 'Reserved': Nuuk_5, '107': Greys_5, 
               '207': Greys_5, '255': Greys_5}

# Helper function
def track_color(mmsi):
    """
    Set the Track Color for a vessel
    """
    trackColor = vcat_colors[metadata_puget_pdf.loc[metadata_puget_pdf.mmsi==str(mmsi), 'vessel_category'].iloc[0]]
    return trackColor


def track24h_lf(mmsi, color, motionDf=motion_puget_pdf):
    """
    Returns a leaflet/folium PolyLine from the track coordinates in (lat, lon) order
    """
    track = motionDf.loc[motionDf['mmsi']==mmsi, 'coordinates'].tolist()
    return PolyLine(locations=track, weight=5, color=color, fill=False, fill_opacity=0.5)


def add_categorical_legend(folium_map, title, colors, labels):
    """
    https://stackoverflow.com/questions/65042654/how-to-add-categorical-legend-to-python-folium-map
    """
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map
#--------------------------------------------------------------------------
def vessel_info(mmsis):
    """
    Get the metadata of one or more vessels from the metadata dataframe.
    'mmsis' can be either 
    1. a single mmsi of type str or int
    2. a list of mmsis each of type int
    3. a pandas Series of mmsi
    4. a pandas Dataframe containing an mmsi column, or
    """
    if type(mmsis) in [int, str]:
        mmsis = [str(mmsis)]
    elif type(mmsis) is pd.core.series.Series:
        mmsis = [str(m) for m in mmsis.unique()]
    elif type(mmsis) is pd.core.frame.DataFrame:
        mmsis = [str(m) for m in df.mmsi.unique()]
    return metadata_puget_pdf.loc[metadata_puget_pdf.mmsi.isin(mmsis), :]


def ais_in_bufferbbox(laneLabel, start, end, lanesGdf=puget_shiplanes_gdf, motionDf=motion_puget_pdf):
    """
    Returns the AIS of vessels within the Bounding Box of the Buffer of the Lane Geometry, in the time between start and end.
    'start', 'end': pd.Timestamp in UTC time zone
    The vessels must have a course (cog) parallel to the lane direction, and must be moving, (sog > 2.0)
    """
    lanePolygon = lanesGdf.loc[lanesGdf['label']==laneLabel, 'geometry'].iloc[0]
    (minLon, minLat, maxLon, maxLat) = lanesGdf.loc[lanesGdf['label']==laneLabel, 'bufferBbox'].iloc[0]
    aisInBufferBbox = motionDf.loc[(motionDf.base_dt >= start) & (motionDf.base_dt < end) & (motionDf.sog >= 2.0) & 
                                   (motionDf.lon >= minLon) & (motionDf.lat >= minLat) & (motionDf.lon < maxLon) & (motionDf.lat < maxLat)]
    # Create a shapely.geometry Point for each vessel location within the Bounding Box of the shipping lane
    if aisInBufferBbox.shape[0] > 0:
        aisInBufferBbox['point'] = aisInBufferBbox.apply(lambda row: Point(row.lon, row.lat), axis=1)
        return aisInBufferBbox
    else:  # return an empty DataFrame (to avoid returning a None)
        col_names = motionDf.columns.tolist() + ['point']
        return pd.DataFrame(columns = col_names)


def tracks_in_buffer(laneLabel, start, end, lanesGdf=puget_shiplanes_gdf, motionDf=motion_puget_pdf):
    """
    Returns a DataFrame of the tracks in the Lane Buffer Bounding Box
    """
    # Aggregation Function: Convert a series of Points to a LineString
    def linestring(df):
        """
        Creates a shapely LineString from a column of shapely Points in a DataFrame
        """
        if df.shape[0] > 1:
            return LineString(list(df['point'].values))
        else:
            return df['point'].iloc[0]
        
    aisInBufferBboxDf = ais_in_bufferbbox(laneLabel, start, end, lanesGdf, motionDf)
    groupdf = aisInBufferBboxDf.groupby(by='mmsi', axis=0)
    tracksDf = groupdf.apply(func=linestring).to_frame(name="trackInBbox")
    buffer = lanesGdf.loc[lanesGdf['label']==laneLabel, 'geometry'].iloc[0].buffer(distance=bufferWidth)
    tracksDf['track'] = tracksDf['trackInBbox'].apply(lambda tbox: buffer.intersection(tbox))
    tracksDf = tracksDf[tracksDf['track'].apply(lambda t: t.is_empty==False)]  # using geom.is_empty() to filter
    return tracksDf[['track']]
#-------------------------------------------------------------
# Folium
def map_tracks_in_buffer(laneLabel=laneLabel, lanesGdf=puget_shiplanes_gdf, tracksDf=tracksDf):
    """
    Create a map of the tracks
    """
    lane_centroid = lanesGdf.loc[lanesGdf['label'] == laneLabel, 'centroid'].iloc[0]
    
    f = folium.Figure(width=700, height=900)
    m = folium.Map(
        location=(lane_centroid.y, lane_centroid.x),
        width=600,
        height=800,
        crs='EPSG3857',
        zoom_start=11,
        tiles=None  # When set to None, it chooses the first Tilelayer created
        )

    TileLayer( \
        name='ESRI World Imagery', \
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', \
        attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community' \
        ).add_to(m)

    TileLayer( \
        name='ESRI NatGeo World Map', \
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', \
        attr='Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC' \
        ).add_to(m)
    # TileLayer('Stamen Watercolor', name='Stamen Watercolor').add_to(m)
    LayerControl(position='topleft', collapsed=True).add_to(m)

    shiplane = lanesGdf.loc[lanesGdf['label'] == laneLabel, 'geometry'].iloc[0]
    buffer = shiplane.buffer(distance=0.03)
    bb = buffer.bounds
    lane_boundary = list(shiplane.exterior.coords)
    buffer_boundary = list(buffer.exterior.coords)

    lanePolygon = Polygon(locations=[(p[1], p[0]) for p in lane_boundary], color='black', weight=1, fill_color=colors[45], fill_opacity=1.0)
    bufferPolygon = Polygon(locations=[(p[1], p[0]) for p in buffer_boundary], color='black', weight=1, fill_color=colors[13], fill_opacity=0.2)
    bufferBboxLine = PolyLine(locations=[(bb[1], bb[0]), (bb[3], bb[0]), (bb[3], bb[2]), (bb[1], bb[2]), (bb[1], bb[0])], color='red', weight=2, fill=False)
    lanePolygon.add_to(m)
    bufferPolygon.add_to(m)
    bufferBboxLine.add_to(m)

    if laneLabel not in lane_singles:
        if type(lane_pairs[laneLabel]) is list:  # _sg means shapely.geometry Polygon, _lf means ipyleaflet
            otherLanePolygons_sg = lanesGdf.loc[lanesGdf['label'].isin(lane_pairs[laneLabel]), 'geometry'].to_list()
            tssPolygons_sg = lanesGdf.loc[lanesGdf['label'].isin(tss_pairs[laneLabel]), 'geometry'].to_list()
            otherLanePolygons_lf = [Polygon(locations=[(p[1], p[0]) for p in list(poly.exterior.coords)], color='black', weight=1, fill_color=colors[18], fill_opacity=0.4) for poly in otherLanePolygons_sg]
            tssPolygons_lf = [Polygon(locations=[(p[1], p[0]) for p in list(poly.exterior.coords)], color='black', weight=1, fill_color=colors[10], fill_opacity=0.7) for poly in tssPolygons_sg]
            for poly in (otherLanePolygons_lf + tssPolygons_lf):
                poly.add_to(m)

        elif type(lane_pairs[laneLabel]) is str:
            otherLanePolygon_sg = lanesGdf.loc[lanesGdf['label'] == lane_pairs[laneLabel], 'geometry'].iloc[0]
            tssPolygon_sg = lanesGdf.loc[lanesGdf['label'] == tss_pairs[laneLabel], 'geometry'].iloc[0]
            otherLanePolygon_lf = Polygon(locations=[(p[1], p[0]) for p in otherLanePolygon_sg.exterior.coords], color='black', weight=1, fill_color=colors[18], fill_opacity=0.4)
            tssPolygon_lf = Polygon(locations=[(p[1], p[0]) for p in tssPolygon_sg.exterior.coords], color='black', weight=1, fill_color=colors[10], fill_opacity=0.7)
            otherLanePolygon_lf.add_to(m)
            tssPolygon_lf.add_to(m)

    tracks = [PolyLine(locations=[(p[1], p[0]) for p in list(tracksDf.loc[mmsi, 'track'].coords)], color=track_color(mmsi), fill_color=track_color(mmsi), \
                       fill_opacity=0.0) for mmsi in tracksDf.index if tracksDf.loc[mmsi, 'track'].geom_type == "LineString"]
    # Flatten MultiLineStrings to a LineString using itertools.chain.from_iterable
    tracks_mls = list(flatten([[PolyLine(locations=[(p[1], p[0]) for p in list(ls.coords)], color=track_color(mmsi), fill_color=track_color(mmsi), fill_opacity=0.0) \
                       for ls in tracksDf.loc[mmsi, 'track']] for mmsi in tracksDf.index if tracksDf.loc[mmsi, 'track'].geom_type == "MultiLineString"]))
    for polyline in (tracks + tracks_mls):
        polyline.add_to(m)

    # vmeta = vessel_info([str(m) for m in tracksDf.index])
    trackStarts = [CircleMarker(\
        location=(tracksDf.loc[mmsi, 'track'].coords[0][1], tracksDf.loc[mmsi, 'track'].coords[0][0]), \
        tooltip=vmeta.loc[vmeta.mmsi==str(mmsi), 'vessel_name'].iloc[0], \
        radius=3, stroke=False, fill_color=colors[48], fill_opacity=1.0) \
        for mmsi in tracksDf.index if tracksDf.loc[mmsi, 'track'].geom_type == "LineString"]
    trackStarts_mls = [CircleMarker(\
        location=(tracksDf.loc[mmsi, 'track'][0].coords[0][1], tracksDf.loc[mmsi, 'track'][0].coords[0][0]), \
        tooltip=vmeta.loc[vmeta.mmsi==str(mmsi), 'vessel_name'].iloc[0], \
        radius=3, stroke=False, fill_color=colors[48], fill_opacity=1.0) \
        for mmsi in tracksDf.index if tracksDf.loc[mmsi, 'track'].geom_type == "MultiLineString"]
    for cm in (trackStarts + trackStarts_mls):
        cm.add_to(m)
    
    # Create the legend using the "add_categorical_legend" function defined earlier
    legend_keys = vmeta['vessel_category'].unique().tolist()
    legend_colors = [vcat_colors[lKey] for lKey in legend_keys]
    g = add_categorical_legend(m.add_to(f), 'Vessel Category', colors = legend_colors, labels = legend_keys).add_to(f)
    
    return g
#---------------------------------------------------------------------------
def inlane_or_crossing(mmsi, laneLabel, tracksDf, lanesGdf):
    """
    Determines if the vessel track corresponding to mmsi is in the lane, crossing the lane, or neither
    """
    lanePolygon = lanesGdf.loc[lanesGdf['label']==laneLabel, 'geometry'].iloc[0]
    track = tracksDf.loc[mmsi, 'track']
    if track.geom_type == "Point":
        if lanePolygon.covers(track):
            return ("inlane", 1)
        else:
            return ("neither", 1)
    elif track.geom_type == "LineString":
        intx = lanePolygon.intersection(track)
        if intx.geom_type == "MultiLineString":
            segments = [list(segment.coords) for segment in intx]
            segmentBearings = [get_bearing(l[0][1], l[0][0], l[-1][1], l[-1][0]) for l in segments]  # get_bearing(lat1, long1, lat2, long2)
            if (max(segmentBearings) - min(segmentBearings) < 90):
                return ("inlane", 1)
            else:
                return ("crossing", len(intx))
        elif intx.geom_type == "LineString":
            intxCoords = intx.coords
            if len(intxCoords) == 0:
                return ("neither", 1)
            elif len(intxCoords) == 1:
                return ("crossing", 1)
            else:
                vesselCourse = get_bearing(intxCoords[0][1], intxCoords[0][0], intxCoords[-1][1], intxCoords[-1][0])
                laneBearing = lane_bearing(laneLabel)
                if ((np.abs(vesselCourse - laneBearing) < 30) or 
                ((laneLabel in lane_singles) and (laneBearing >= 0) and (np.abs(vesselCourse - (laneBearing - 180)) < 30)) or 
                ((laneLabel in lane_singles) and (laneBearing < 0) and (np.abs(vesselCourse - (laneBearing + 180)) < 30))):
                    return ("inlane", 1)
                else:
                    return ("crossing", 1)
    elif track.geom_type == "MultiLineString":
        multiIntx = lanePolygon.intersection(track)
        if multiIntx.geom_type == "LineString":
            if len(multiIntx.coords) == 0:
                return ("neither", 1)
            else:
                return ("crossing", 1)
        elif multiIntx.geom_type == "MultiLineString":
            num_crossings = np.sum([1 for intx in multiIntx if len(intx.coords) > 0])
            if num_crossings == 0:
                return ("neither", 1)
            else:
                return ("crossing", num_crossings)
#--------------------------------------------------------------------------
def map_shiplanes(lanesGdf):
    """
    Draws a map of the shipping lanes
    lanesGdf: is the GeoDataFrame with rows corresponding to the shipping lanes, precautionary areas, and separation schemes
    m is an ipyleaflet Map object, with Center coordinates and Zoom level specified
    """
    f = folium.Figure(width=700, height=900)
    m = folium.Map(
        location=(48.0, -122.65),
        width=600,
        height=800,
        crs='EPSG3857',
        zoom_start=10,
        tiles=None  # When set to None, it chooses the first Tilelayer created
        )

    TileLayer( \
        name='ESRI World Imagery', \
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', \
        attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community' \
        ).add_to(m)

    TileLayer( \
        name='ESRI NatGeo World Map', \
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', \
        attr='Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC' \
        ).add_to(m)
    
    LayerControl(position='topleft', collapsed=True).add_to(m)
    
    # Precautionary Areas
    precAreas_gs = lanesGdf[lanesGdf.OBJL == '96'].geometry  
    num_precAreas = len(precAreas_gs)
    precAreas = [None] * num_precAreas
    for i in range(num_precAreas):
        precAreas[i] = [(((precAreas_gs.iloc[i]).exterior.coords)[k][1], ((precAreas_gs.iloc[i]).exterior.coords)[k][0]) for k in range(len((precAreas_gs.iloc[i]).exterior.coords))]
    
    # Traffic Separation Schemes
    sepSchemes_gs = lanesGdf[lanesGdf.OBJL == '150'].geometry
    num_sepSchemes = len(sepSchemes_gs)
    sepSchemes = [None] * num_sepSchemes
    for i in range(num_sepSchemes):
        sepSchemes[i] = [(((sepSchemes_gs.iloc[i]).exterior.coords)[k][1], ((sepSchemes_gs.iloc[i]).exterior.coords)[k][0]) for k in range(len((sepSchemes_gs.iloc[i]).exterior.coords))]
    
    # Shipping Lanes
    shiplanes_gs = lanesGdf[(lanesGdf.OBJL == '148') | (lanesGdf.OBJL == '152')].geometry
    num_shiplanes = len(shiplanes_gs)
    shiplanes = [None] * num_shiplanes
    for i in range(num_shiplanes):
        shiplanes[i] = [(((shiplanes_gs.iloc[i]).exterior.coords)[k][1], ((shiplanes_gs.iloc[i]).exterior.coords)[k][0]) for k in range(len((shiplanes_gs.iloc[i]).exterior.coords))]
        
    precAreas_polygons_list = [Polygon(locations=precAreas[i], color='red', weight=5, fill_color='red') for i in range(num_precAreas)]
    sepSchemes_polygons_list = [Polygon(locations=sepSchemes[i], color='black', weight=1, fill_color='blue', fill_opacity=0.7) for i in range(num_sepSchemes)]
    shiplanes_polygons_list = [Polygon(locations=shiplanes[i], color='black', weight=1, fill_color=darkColors[4], fill_opacity=1.0) for i in range(num_shiplanes)]  # fill_color=colors[i % 8]

    for polygon in (precAreas_polygons_list + sepSchemes_polygons_list + shiplanes_polygons_list):
      polygon.add_to(m)
    g = m.add_to(f)
    return g
#---------------------------------------------------------------------------
def map_tracks_by_vcat(vcat, vcat2mmsiSer, lanesGdf=puget_shiplanes_gdf, motionDf=motion_puget_pdf, metaDf=metadata_puget_pdf):
    """
    Maps the vessel tracks for every mmsi in a vessel category (vcat).
    The vessel send at least 25 AIS signals, while it is in motion (sog > 2.0)
    """
    f = folium.Figure(width=700, height=900)
    m = map_shiplanes(lanesGdf)
    mmsiSer = vcat2mmsiSer[vcat]
    palette = (vcat_palette[vcat]).hex_colors
    allTracks = [track24h_lf(mmsiSer.index[i], palette[i%5], motionDf) for i in range(len(mmsiSer))]
    trackStarts = [CircleMarker( \
        location=(allTracks[j].locations[0][0], allTracks[j].locations[0][1]), \
        tooltip=((metaDf.loc[metaDf.mmsi==str(mmsiSer.index[j]), 'vessel_name'].iloc[0]) + ", " \
            + (motionDf.loc[((motionDf.mmsi==str(mmsiSer.index[j])) & (motionDf.sog > 2.0)), 'local_dt'].iloc[0]).strftime("%H:%M")), \
        radius=3, stroke=False, fill_color=colors[48], fill_opacity=1.0) for j in range(len(allTracks))]
    for mapElement in (allTracks + trackStarts):
        mapElement.add_to(m)
    g = m.add_to(f)
    return g
#---------------------------------------------------------------------------

if __name__ == "__main__":
    # st.selectbox, st.form & st.form_submit_button, st.sidebar.selectbox, st.image
    # st.title, st.header, st.subheader
    # streamlit-folium
    # This Component uses components.html to take the HTML generated from a folium.Map or 
    # folium.Figure object render() method, and display it as-is within Streamlit.
    vcats = np.sort(motion_puget_pdf.loc[(motion_puget_pdf.sog > 2.0), 'vessel_category'].unique())
    vcat2TrackLengths = {vcat:motion_puget_pdf.loc[(motion_puget_pdf.sog > 2.0) & (motion_puget_pdf['vessel_category']==vcat), 'mmsi'].value_counts() for vcat in vcats}
    vcat2mmsis = {vcat:vcat2TrackLengths[vcat][vcat2TrackLengths[vcat] >= 25] for vcat in vcat2TrackLengths.keys()}
    vcat2mmsiSer = {k: v for k, v in vcat2mmsis.items() if (len(v) > 0)}
    options = ["Lanes only. No tracks"] + list(vcat2mmsiSer.keys())
    quickMaps = ('Admiralty Inlet', 'Dungeness Spit', 'Rosario Strait', 'Seattle', 'South Sound')

    # form = st.sidebar.form("my_form")
    # mapChoice = form.selectbox("Traffic by Shipping Lane", quickMaps)
    # tracksChoice = form.selectbox("Vessel Tracks by Category", tuple(options))
    # st.set_page_config(page_title="Collision Avoidance", layout="wide")
    st.title("Collision Avoidance in the Greater Puget Sound")
    st.subheader("by Rock Pereira")
    with st.sidebar.form(key="my_form"):
        mapChoice = st.selectbox("Traffic by Shipping Lane", quickMaps)
        tracksChoice = st.selectbox("Vessel Tracks by Category", tuple(options))
        submit_button = st.form_submit_button(label='Submit')

    st.sidebar.markdown(body="""
    *Description:*

    AIS data is used to study traffic in the Greater Puget Sound.
    The data represents vessel motion on one day - Jan 06, 2020, from 12:00 am to 11:59 pm, UTC.

    The buffer size is 2 nautical miles. 
    
    The pink dot is the start of the trip or segment. The tooltip on the dot uses local time (Pacific).

    To change the base layer, click on the expander icon (topleft)
    """)

    col1, col2, col3 = st.beta_columns([4,1,4])

    with col1:
        if mapChoice == 'Admiralty Inlet':
            laneLabel = "sl58"
            start = pd.Timestamp('2020-01-06T18:00:00', tz='UTC')
            end = pd.Timestamp('2020-01-06T20:00:00', tz='UTC')
            st.header("Admiralty Inlet, 10-12 pm")
        elif mapChoice == 'Dungeness Spit':
            laneLabel = "sl72"
            start = pd.Timestamp('2020-01-06T01:00:00', tz='UTC')
            end = pd.Timestamp('2020-01-06T03:00:00', tz='UTC')
            st.header("Tankers near Dungeness Spit, 5-7 pm")
        elif mapChoice == 'Rosario Strait':
            laneLabel = "sl02"
            start = pd.Timestamp('2020-01-06T15:00:00', tz='UTC')
            end = pd.Timestamp('2020-01-06T17:00:00', tz='UTC')
            st.header("Rosario Strait, 7-9 am")
        elif mapChoice == 'Seattle':
            laneLabel = "sl43"
            start = pd.Timestamp('2020-01-06T13:00:00', tz='UTC')
            end = pd.Timestamp('2020-01-06T15:00:00', tz='UTC')
            st.header("Seattle Cross-Traffic, 5-7 am")
        elif mapChoice == 'South Sound':
            laneLabel = "sl78"
            start = pd.Timestamp('2020-01-06T15:00:00', tz='UTC')
            end = pd.Timestamp('2020-01-06T17:00:00', tz='UTC')
            st.header("Tacoma to Seattle, 7-9 am")
        else:
            mapChoice = ""

    if mapChoice in quickMaps:
        tracksDf = tracks_in_buffer(laneLabel=laneLabel, start=start, end=end, lanesGdf=puget_shiplanes_gdf, motionDf=motion_puget_pdf)
        vmeta = vessel_info([str(m) for m in tracksDf.index])
        mapTracksInBuffer = map_tracks_in_buffer(laneLabel=laneLabel, lanesGdf=puget_shiplanes_gdf, tracksDf = tracksDf)

        tracksDf['vessel_category'] = tracksDf.index.to_series().apply(lambda m: vmeta.loc[vmeta.mmsi==str(m), 'vessel_category'].iloc[0])
        tracksDf['vessel_name'] = tracksDf.index.to_series().apply(lambda m: vmeta.loc[vmeta.mmsi==str(m), 'vessel_name'].iloc[0])
        tracksDf['length'] = tracksDf.index.to_series().apply(lambda m: vmeta.loc[vmeta.mmsi==str(m), 'length'].iloc[0])
        tracksDf['width'] = tracksDf.index.to_series().apply(lambda m: vmeta.loc[vmeta.mmsi==str(m), 'width'].iloc[0])

        l = tracksDf.index.to_series().apply(lambda m: inlane_or_crossing(m, laneLabel, tracksDf, puget_shiplanes_gdf))

        tracksDf['traversal_type'] = l.apply(lambda tup: tup[0])
        tracksDf['traversal_count'] = l.apply(lambda tup: tup[1])
        traversals = tracksDf[tracksDf.columns[1:]].sort_values(by='vessel_category', inplace=False)

        with col1:
            folium_static(mapTracksInBuffer, width=500, height=800)
            st.header("In-Lane and Crossing Traversals")
            st.dataframe(traversals, width=800)

        # doNotRender = """
        with col2:
            st.header("<= legend")
            if mapChoice == 'Admiralty Inlet':
                st.image(IMAGE_DIR + "crossings_legend_admiralty_inlet.png", use_column_width=True)
            elif mapChoice == 'Dungeness Spit':
                st.image(IMAGE_DIR + "crossings_legend_dungeness_spit.png", use_column_width=True)
            elif mapChoice == 'Rosario Strait':
                st.image(IMAGE_DIR + "crossings_legend_rosario_strait.png", use_column_width=True)
            elif mapChoice == 'Seattle':
                st.image(IMAGE_DIR + "crossings_legend_seattle.png", use_column_width=True)
            elif mapChoice == 'South Sound':
                st.image(IMAGE_DIR + "crossings_legend_south_sound.png", use_column_width=True)
            else:
                mapChoice = "" 
        # """

    with col3:
        if tracksChoice == "Lanes only. No tracks":
            st.header("Shipping Lanes")
            mapShiplanes = map_shiplanes(puget_shiplanes_gdf)
            folium_static(mapShiplanes, width=500, height=800)
        elif (tracksChoice in options[1:]):
            st.header("Vessel Tracks: " + tracksChoice)
            mapTracksByVcat = map_tracks_by_vcat(tracksChoice, vcat2mmsiSer, puget_shiplanes_gdf, motion_puget_pdf, metadata_puget_pdf)
            folium_static(mapTracksByVcat, width=500, height=800)

# https://streamlit.io/gallery
# https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py