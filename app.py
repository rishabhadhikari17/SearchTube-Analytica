################################
## Importing Libraries
################################

import isodate
from dateutil import parser
from IPython.display import JSON
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tabulate import tabulate
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.subplots as sp
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

#####################################
## importing libraries for animations
######################################
import json
import requests
from streamlit_lottie import st_lottie


##############################
## functions to load animations
###############################

def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
    

###############################
## Search engine Duration-Seconds Function
################################

# Define a function to convert duration string to seconds
def duration_to_seconds(duration):
    match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration)
    if match is not None:
        hours = int(match.group(1)[:-1]) if match.group(1) else 0
        minutes = int(match.group(2)[:-1]) if match.group(2) else 0
        seconds = int(match.group(3)[:-1]) if match.group(3) else 0
        return hours * 3600 + minutes * 60 + seconds
    else:
        return 0


#######################################
## Set up the YouTube API client
## api_service_name = "youtube"
## api_version = "v3"
#######################################
api_key = 'AIzaSyAHsBjik6jeBDU_n01xM7K9VR8WM9RvcqY'
youtube = build('youtube', 'v3', developerKey=api_key)


###################################
## Data Collection for channel_data
###################################

def get_channel_stats(youtube, channel_ids):
    all_data = []
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=','.join(channel_ids))
    response = request.execute()

    for i in range(len(response['items'])):
        data = dict(channelName=response['items'][i]['snippet']['title'],
                    subscribers=response['items'][i]['statistics']['subscriberCount'],
                    views=response['items'][i]['statistics']['viewCount'],
                    totalVideos=response['items'][i]['statistics']['videoCount'],
                    playlistId=response['items'][i]['contentDetails']['relatedPlaylists']['uploads'])
        all_data.append(data)

    return pd.DataFrame(all_data)


def get_video_ids(youtube, playlist_id):

    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50)
    response = request.execute()

    video_ids = []

    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])

    next_page_token = response.get('nextPageToken')
    more_pages = True

    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token)
            response = request.execute()
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]
                                 ['contentDetails']['videoId'])

            next_page_token = response.get('nextPageToken')

    return video_ids


def get_video_details(youtube, video_ids):

    all_video_info = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute()

        for video in response['items']:
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                             'contentDetails': ['duration'
                                                #'definition', 'caption'
                            ]
                             }
            video_info = {}
            video_info['video_id'] = video['id']

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None

            all_video_info.append(video_info)

    return pd.DataFrame(all_video_info)



#########################
## Functions For Graphs
#########################

# Comparison of Subscriber Count
def comparison_of_subscriber_count(channel_data):
    fig = px.bar(
        data_frame=channel_data.sort_values('subscribers', ascending=False),
        x='channelName',
        y='subscribers',
        labels={'channelName': 'Channel Name', 'subscribers': 'Subscribers'},
        color='channelName'
    )
    fig.update_layout(
        xaxis_tickangle=-90,
        yaxis_tickformat=',.0f',
        yaxis_ticksuffix='K',
        yaxis_title='Subscribers'
    )
    return fig

# pie chart for showing the proportion of total views from the whole lot
def pie_chart_of_views(channel_data):
    fig = go.Figure(data=go.Pie(
        labels=channel_data['channelName'],
        values=channel_data['views'],
        textinfo='label+percent',
        hoverinfo='value',
        title='Proportion of Views for each channel'
    ))

    return fig


# box plot for view analysis
def box_plot_for_view_count_analysis(video_df):
    fig=px.box(video_df,y="viewCount",color="channelTitle")
    return fig


# relationship between view and likecount for different channels
def view_like_correlation(video_df):
    # Convert 'published_date' column to datetime
    video_df['publishedAt'] = pd.to_datetime(video_df['publishedAt'])

    # Create the channel engagement chart
    fig = px.scatter(video_df, x='viewCount', y='likeCount',
                     hover_data=['channelTitle', 'publishedAt'],
                    color=video_df["channelTitle"])

    # Customize the axes labels
    fig.update_layout(xaxis_title='View Count', yaxis_title='Like Count')

    # Display the chart
    return fig


#Channel Engagement according to viewCount over the years
def channel_engagement_viewCount(video_df):
    fig = go.Figure()
    # Iterate over unique Channel Titles
    for channelTitle in video_df['channelTitle'].unique():
        # Filter the dataframe for the specific Channel Title
        channel_data = video_df[video_df['channelTitle'] == channelTitle]
        # Add trace for the specific Channel Title
        fig.add_trace(go.Scatter(
            x=channel_data['publishedAt'],
            y=channel_data['viewCount'],
            mode='lines+markers',
            name=channelTitle,
            marker=dict(
                size=8,
                symbol='circle'
            )
        ))
    return fig


# Engagement Count on the basis of CommentCount
def channel_engagement_commentCount(video_df):
    fig = go.Figure()
    # Iterate over unique Channel Titles
    for channelTitle in video_df['channelTitle'].unique():
        # Filter the dataframe for the specific Channel Title
        channel_data = video_df[video_df['channelTitle'] == channelTitle]
        # Add trace for the specific Channel Title
        fig.add_trace(go.Scatter(
            x=channel_data['publishedAt'],
            y=channel_data['commentCount'],
            mode='lines+markers',
            name=channelTitle,
            marker=dict(
                size=8,
                symbol='circle'
            )
        ))
    return fig


#Engagement chart on the basis of like Count
def channel_engagement_likeCount(video_df):
    fig = go.Figure()
    # Iterate over unique Channel Titles
    for channelTitle in video_df['channelTitle'].unique():
        # Filter the dataframe for the specific Channel Title
        channel_data = video_df[video_df['channelTitle'] == channelTitle]
        # Add trace for the specific Channel Title
        fig.add_trace(go.Scatter(
            x=channel_data['publishedAt'],
            y=channel_data['likeCount'],
            mode='lines+markers',
            name=channelTitle,
            marker=dict(
                size=8,
                symbol='circle'
            )
        ))
    return fig


# Does Description length affects view count?
def description_length_viewCount(video_df):
    fig = px.histogram(video_df, x='descriptionLength', y='viewCount', nbins=30, facet_col='channelTitle',color='channelTitle')
    fig.update_layout(
        xaxis_title="Description Length",
        yaxis_title="View Count"
    )
    return fig


# When my competitors are posting content on YouTube?
def video_publishing_days(video_df):
    fig = px.histogram(video_df, x='pushblishDayName', color='channelTitle', 
                       category_orders={'pushblishDayName': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    fig.update_layout(
        xaxis_title="Day of the Week",
        yaxis_title="Count"
    )
    return fig


# Word Cloud:Description
def worlcloud_description(video_df):
    # Iterate over unique channel titles
    channel_titles = video_df['channelTitle'].unique()
    # Create subplots based on the number of unique channel titles
    fig, axs = plt.subplots(1, len(channel_titles), figsize=(15, 9))
    # Generate word cloud for each channel title
    for i, channel_title in enumerate(channel_titles):
        channel_df = video_df[video_df['channelTitle'] == channel_title]
        wordcloud = WordCloud(width=1000, height=1000, background_color='white', min_font_size=10).generate(' '.join(channel_df['description'].astype(str)))
        axs[i].imshow(wordcloud)
        axs[i].set_title(channel_title)
        axs[i].axis('off')
    return plt
        

# Word Cloud: Tags
def worlcloud_tags(video_df):
    # Iterate over unique channel titles
    channel_titles = video_df['channelTitle'].unique()
    # Create subplots based on the number of unique channel titles
    fig, axs = plt.subplots(1, len(channel_titles), figsize=(15, 9))
    # Generate word cloud for each channel title
    for i, channel_title in enumerate(channel_titles):
        channel_df = video_df[video_df['channelTitle'] == channel_title]
        wordcloud = WordCloud(width=1000, height=1000, background_color='white', min_font_size=10).generate(' '.join(channel_df['tags'].astype(str)))
        axs[i].imshow(wordcloud)
        axs[i].set_title(channel_title)
        axs[i].axis('off')
    return plt
        

# How much proportion of tags my channel is using?
def tags_proportion(video_df):
    fig = go.Figure(data=go.Pie(
        labels=video_df['channelTitle'],
        values=video_df['tagsCount'],
        textinfo='label+percent',
        hoverinfo='value',
    ))
    fig.update_layout(
        height=500,
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig


# is there any relationship between tags count and viewCount?
def tag_view_relation(video_df):
    # Get unique channel titles
    channel_titles = video_df['channelTitle'].unique()
    # Create subplots based on the number of unique channel titles
    fig = sp.make_subplots(rows=len(channel_titles), cols=1, subplot_titles=channel_titles)
    # Variables to track the minimum and maximum values of viewCount
    min_view_count = float('inf')
    max_view_count = float('-inf')
    # Generate scatter plot for each channel title
    for i, channel_title in enumerate(channel_titles):
        channel_df = video_df[video_df['channelTitle'] == channel_title]
        fig.add_trace(go.Scatter(
            x=channel_df['tagsCount'],
            y=channel_df['viewCount'],
            mode='markers',
            name=channel_title,
            marker=dict(
                size=8,
                color=i,  # Assign a unique color to each channel title
                colorscale='Viridis',
                showscale=False
            )
        ), row=i+1, col=1)
        # Update the minimum and maximum viewCount values
        min_view_count = min(min_view_count, channel_df['viewCount'].min())
        max_view_count = max(max_view_count, channel_df['viewCount'].max())
        # Set axis labels for each subplot
        fig.update_xaxes(title_text="Tags Count", row=i+1, col=1)
        fig.update_yaxes(title_text="View Count", range=[min_view_count, max_view_count], row=i+1, col=1)
    # Set layout properties
    fig.update_layout(
        showlegend=False,  # Hide the legend for individual channel titles
        height=700,  # Adjust the height of the plot
        hovermode='closest'  # Show the closest data point when hovering
    )
    # Show the plot
    return fig


# is there any relationship between title length and viewCount?
def titlelength_viewCount_relation(video_df):
    # Get unique channel titles
    channel_titles = video_df['channelTitle'].unique()
    # Create subplots based on the number of unique channel titles
    fig = sp.make_subplots(rows=len(channel_titles), cols=1, subplot_titles=channel_titles)
    # Variables to track the minimum and maximum values of viewCount
    min_view_count = float('inf')
    max_view_count = float('-inf')
    # Generate scatter plot for each channel title
    for i, channel_title in enumerate(channel_titles):
        channel_df = video_df[video_df['channelTitle'] == channel_title]
        fig.add_trace(go.Scatter(
            x=channel_df['titleLength'],
            y=channel_df['viewCount'],
            mode='markers',
            name=channel_title,
            marker=dict(
                size=8,
                color=i,  # Assign a unique color to each channel title
                colorscale='Viridis',
                showscale=False
            )
        ), row=i+1, col=1)
        # Update the minimum and maximum viewCount values
        min_view_count = min(min_view_count, channel_df['viewCount'].min())
        max_view_count = max(max_view_count, channel_df['viewCount'].max())
        # Set axis labels for each subplot
        fig.update_xaxes(title_text="titleLength", row=i+1, col=1)
        fig.update_yaxes(title_text="View Count", range=[min_view_count, max_view_count], row=i+1, col=1)
    # Set layout properties
    fig.update_layout(
        showlegend=False,  # Hide the legend for individual channel titles
        height=700,  # Adjust the height of the plot
        hovermode='closest'  # Show the closest data point when hovering
    )
    # Show the plot
    return fig


# Content Duration Posted
def content_duration(video_df):
    # Get unique channel titles
    channel_titles = video_df['channelTitle'].unique()
    # Create subplots based on the number of unique channel titles
    fig = sp.make_subplots(rows=len(channel_titles), cols=1, subplot_titles=channel_titles)
    # Determine the common x-axis range
    max_duration = video_df['durationSecs'].max()
    min_duration = video_df['durationSecs'].min()
    # Generate histogram for each channel title
    for i, channel_title in enumerate(channel_titles):
        channel_df = video_df[video_df['channelTitle'] == channel_title]
        fig.add_trace(go.Histogram(
            x=channel_df['durationSecs'],
            xbins=dict(start=min_duration, end=max_duration, size=12),
            name=channel_title,
            showlegend=True,  
            histfunc='count'  
        ), row=i+1, col=1)
        # Set axis labels for each subplot
        fig.update_xaxes(title_text="Duration (seconds)", range=[min_duration, max_duration], row=i+1, col=1)
    # Set layout title
    return fig


# Is there any relationship between duration and ViewCount
def duration_view_relation(video_df):
    channel_titles = video_df['channelTitle'].unique()
    fig = sp.make_subplots(rows=len(channel_titles), cols=1, subplot_titles=channel_titles)
    # Determine the common x-axis range
    max_duration = video_df['durationSecs'].max()
    min_duration = video_df['durationSecs'].min()
    for i, channel_title in enumerate(channel_titles):
        channel_df = video_df[video_df['channelTitle'] == channel_title]
        fig.add_trace(go.Scatter(
            x=channel_df['durationSecs'],
            y=channel_df['viewCount'],
            mode='markers',
            name=channel_title
        ), row=i+1, col=1)
        # Set axis labels and range for each subplot
        fig.update_xaxes(title_text="Duration (seconds)", range=[min_duration, max_duration], row=i+1, col=1)
        fig.update_yaxes(title_text="View Count", row=i+1, col=1)
    return fig



##########################
## Streamlit Layout
##########################
st.title("SearchTube Analytica")


##########################
# Web APP Navigation
##########################


selected=option_menu(
    menu_title = None,
    options=['Home','Instructions','Discover','Analysis'],
    menu_icon='cast',
    icons=['house','book','eye','bar-chart'],
    orientation='horizontal',
    styles={
        'container':{'padding':'0!important','background-color':'white'},
        'icon':{'color':'white','font-size':'15px'},
        'nav-link':{
            'font-size':"15px",
            'text-align':'left',
            'margin':'0px',
            '--hover-color':'red',
        },
        'nav-link-selected':{'background-color':'red'}
    },
    )


############################
## Conditions for Navigation
#############################


#############
## Home
############
if selected=='Home':

    header_style = """
    <style>
        .header {
            text-align: justify;
            color:darkblue;
        }
    </style>
    """
    col1,col2=st.columns(2)
    # st.header("Unleash the Potential of Your YouTube Channel. Discover Trends, Analyze Competitors, and Chart Your Journey with Our Cutting-Edge Analysis and Search Engine Web App.")
    with col1:
        home_animation='home_animation.gif'
        st.image(home_animation)

    with col2:
        st.write("            ")
        st.write("            ")
        st.write("            ")
        st.write("            ")
        st.write("            ")
        st.markdown(header_style, unsafe_allow_html=True)
        st.markdown("<h5 class='header'><big>Unleash</big> the Potential of Your YouTube Channel. Discover Trends, Analyze Competitors, and Chart Your Journey with Our Cutting-Edge Analysis and Search Engine Web App.</h5>", unsafe_allow_html=True)
        


###################
## General Instructions
####################

elif selected == 'Instructions':
    # st.header("Mastering the App: Your Expert Instruction Manual")
    header_style = """
    <style>
        .header_instructions{
            text-align: left;
            color:darkblue;
        }
    </style>
    """

    col1,col2=st.columns(2)
    with col1:
        st.markdown(header_style,unsafe_allow_html=True)
        st.markdown("<h3 class='header_instructions'>Mastering the App: Your Expert Instruction Manual</h3>", unsafe_allow_html=True)
        lottie_creator=load_lottiefile("creator_animation.json")
        st_lottie(lottie_creator,speed=1,reverse=False,loop=True)

    instruction_style="""
    <style>
    .instruction_heading{
    text-align:left;
    }
    </style>
    """
    # st.write("General Instructions:")
    with col2:
        st.markdown("<h5 class='instruction_heading'>General Instructions</h5>",unsafe_allow_html=True)
        st.write("1. This web app does not claim to increase the reach of your YouTube Channel.")
        st.write("2. For better visualization of graphs, try to compare 2-3 channels at a time.")
        st.write("3. The YouTube API Key might get exhaust ,when used repeatedly in a single go.")
        st.write("4. The web app might get slow when the channels has numerous resources.")
        st.write("5. Please make sure you have an active internet connection on your system.")

###################
## Search Engine Part
####################

elif selected=='Discover':
    # Set the search query
    query = st.text_input("Enter the query:")

    # Make the search request
    if query:
        try:
            search_response = youtube.search().list(
                q=query,
                type='channel',
                part='id,snippet',
                maxResults=25
            ).execute()
        except HttpError as error:
            st.error(f'An error occurred: {error}')
            search_response = None

        # Extract the channel names, their view counts, and average video duration
        channel_data = []
        if search_response is not None:
            for search_result in search_response.get('items', []):
                channel_id = search_result['id']['channelId']
                channel_name = search_result['snippet']['title']
                channel_response = youtube.channels().list(
                    id=channel_id,
                    part='statistics'
                ).execute()
                view_count = int(channel_response['items'][0]['statistics']['viewCount'])
                subscriber_count = int(channel_response['items'][0]['statistics']['subscriberCount'])
                channel_videos = youtube.search().list(
                    channelId=channel_id,
                    type='video',
                    part='id',
                    maxResults=25
                ).execute()
                video_ids = [item['id']['videoId'] for item in channel_videos['items']]
                video_durations = []
                for video_id in video_ids:
                    video_response = youtube.videos().list(
                        id=video_id,
                        part='contentDetails'
                    ).execute()
                    duration = video_response['items'][0]['contentDetails']['duration']
                    video_durations.append(duration)
                video_durations_in_seconds = [duration_to_seconds(duration) for duration in video_durations]
                if len(video_durations_in_seconds) > 0:
                    average_duration_in_seconds = sum(video_durations_in_seconds) // len(video_durations_in_seconds)
                else:
                    average_duration_in_seconds = 0
                channel_data.append([channel_name, channel_id, view_count, subscriber_count, average_duration_in_seconds])

        # Sort the channels by subscriber count in descending order
        sorted_channels = sorted(channel_data, key=lambda x: x[3], reverse=True)

        # Create a DataFrame for the channel data
        df = pd.DataFrame(sorted_channels, columns=['Channel Name', 'Channel ID', 'View Count', 'Subscriber Count', 'Average Duration (Seconds)'])

        # Display the DataFrame
        if len(df) > 0:
            st.header("YouTube Channel Analysis")
            st.dataframe(df)
        else:
            st.warning("No channels found.")


####################
## Analysis Part
####################

elif selected == "Analysis":
    st.header("Chart Your YouTube Journey ")
    ### Inputting the no. of channels to be compared and the respective channels IDS
    choice = st.number_input("Enter number of channels you want to compare ",step=1)
    channel_id = []
    for i in range(1,choice+1):
        channel = st.text_input(f"Enter the Channel id {i}")
        channel_id.append(channel)

    ### If Analyze button is clicked
    if st.button("Analyze"):

    ### Changing the datatypes of channel_data to numeric
        channel_data = get_channel_stats(youtube, channel_id)
        numeric_cols = ['subscribers', 'views', 'totalVideos']
        channel_data[numeric_cols] = channel_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    ### Creating video_df for Graphs and overall analysis
        video_df = pd.DataFrame()
        for c in channel_data['channelName'].unique():
            print("Getting video information from channel: " + c)
            playlist_id = channel_data.loc[channel_data['channelName']
                                        == c, 'playlistId'].iloc[0]
            video_ids = get_video_ids(youtube, playlist_id)

            # get video data
            video_data = get_video_details(youtube, video_ids)
            # append video data together and comment data together
            video_df = pd.concat([video_df, pd.DataFrame(video_data)], axis=0)


        # converting columns into numeric datatype
        cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
        video_df[cols] = video_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        # creating the Day of the week column for each video
        video_df['publishedAt'] = video_df['publishedAt'].apply(
            lambda x: parser.parse(str(x)))
        video_df['pushblishDayName'] = video_df['publishedAt'].apply(
            lambda x: x.strftime("%A"))
        # converting  duration of the videos to seconds (for each video)
        video_df['durationSecs'] = video_df['duration'].apply(
            lambda x: isodate.parse_duration(x))
        video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')
        # Add number of tags
        video_df['tagsCount'] = video_df['tags'].apply(
            lambda x: 0 if x is None else len(x))
        # Comments and likes per 1000 view ratio
        video_df['likeRatio'] = video_df['likeCount'] / video_df['viewCount'] * 1000
        video_df['commentRatio'] = video_df['commentCount'] / \
            video_df['viewCount'] * 1000
        # Title character length
        video_df['titleLength'] = video_df['title'].apply(lambda x: len(x))
        # Getting description length
        video_df["descriptionLength"] = video_df["description"].apply(
            lambda x: 0 if x is None else len(x))
        


        ### Calling all the graphs functions created above
        
        st.header('Channel Wise Subscriber Count')
        st.plotly_chart(comparison_of_subscriber_count(channel_data))

        st.header("Proportion Of Views For Each Channel")
        st.plotly_chart(pie_chart_of_views(channel_data))

        st.header("Box Plot Analysis For View Count")
        st.plotly_chart(box_plot_for_view_count_analysis(video_df))

        st.header("Relationship Between View And Like Count")
        st.plotly_chart(view_like_correlation(video_df))

        st.header("Channel Engagement : View Count")
        st.plotly_chart(channel_engagement_viewCount(video_df))

        st.header("Channel Engagement : Comment Count")
        st.plotly_chart(channel_engagement_commentCount(video_df))

        st.header("Channel Engagement : Like Count")
        st.plotly_chart(channel_engagement_likeCount(video_df))

        st.header("Does Description Length Matters For View Count")
        st.plotly_chart(description_length_viewCount(video_df))

        st.header("When Others Are Posting Content?")
        st.plotly_chart(video_publishing_days(video_df))

        st.header("Word Cloud : Description")
        st.pyplot(worlcloud_description(video_df))

        st.header("Word Cloud : Tags")
        st.pyplot(worlcloud_tags(video_df))

        st.header("How much Proportion of Tags the Channel Is Using?")
        st.plotly_chart(tags_proportion(video_df))

        st.header("Is Thier Relationship Between Tags Count And View Count?")
        st.plotly_chart(tag_view_relation(video_df))

        st.header("Is Thier Relationship Between Title Length And View Count?")
        st.plotly_chart(titlelength_viewCount_relation(video_df))

        st.header("How Much Duration Of Content The Channel Is Posting?")
        st.plotly_chart(content_duration(video_df))

        st.header("Is Their Any Realtion Between Duration And View Count?")
        st.plotly_chart(duration_view_relation(video_df))