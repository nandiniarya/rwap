import streamlit as st
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
import altair as a

# Setting the Page configuration
st.set_page_config(page_title="FIFA Player Stats Dashboard", page_icon=":bar_chart:", layout="wide")

# Setting the dashboard title
st.title(":bar_chart: FIFA Player Stats Dashboard")

# Make the title appear above
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

data_url="final.csv"
df = pd.read_csv(data_url)

# Display the DataFrame
st.subheader("Data Preview:")
st.dataframe(df)






import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Sample DataFrame for demonstration (use your df here)
# df = pd.read_csv('your_data.csv')  # Ensure to use your actual data

# Define wage ranges
wage_ranges = [
    (0, 100000), (100001, 200000), (200001, 300000), (300001, 400000),
    (400001, 500000), (500001, 600000), (600001, 700000), (700001, 800000),
    (800001, 900000), (900001, 1000000)
]
wage_range_labels = [f"{low:,} - {high:,}" for low, high in wage_ranges]

# Sidebar for FIFA version selection
st.sidebar.header("Filters")
fifa_version = st.sidebar.selectbox(
    "Select FIFA Version",
    sorted(df['fifa_version'].unique())
)

# Filtered DataFrame based on FIFA version
df_fifa = df[df['fifa_version'] == fifa_version] if fifa_version else df

# Wage range selection
wage_range_index = st.sidebar.selectbox(
    "Select Wage Range",
    list(range(len(wage_range_labels))),
    format_func=lambda x: wage_range_labels[x]
)

low, high = wage_ranges[wage_range_index]
filtered_df_wage = df_fifa[(df_fifa['wage_eur'] >= low) & (df_fifa['wage_eur'] <= high)]

if not filtered_df_wage.empty:
    fig = px.treemap(
        filtered_df_wage,
        path=['player_positions_1'],
        values='wage_eur',
        title='Player Positions Treemap'
    )

    # Use Plotly's FigureWidget for interactive clicks
    fig_widget = go.FigureWidget(fig)
    scatter = fig_widget.data[0]

    st.subheader("Player Stats by Wage Range")
    st.plotly_chart(fig)

    # Interactive handling within Streamlit
    clicked_data = st.session_state.get('clicked_data')

    if st.session_state.get('clicked') and clicked_data:
        selected_position = clicked_data
        filtered_details = filtered_df_wage[filtered_df_wage['player_positions_1'] == selected_position]
        if not filtered_details.empty:
            st.write("Player Details:")
            st.dataframe(filtered_details[['short_name', 'player_positions_1', 'nationality_name', 'club_name', 'wage_eur', 'value_eur', 'overall', 'potential', 'age']])

# Third Visualization: Player Tags Bar Chart
# ... (rest of your code) ...



import streamlit as st
import pandas as pd
import plotly.express as px

# Load your data into df (replace with your actual data loading code)
# df = pd.read_csv('your_data.csv')  # Example data loading line

# Streamlit App Layout
st.title("FIFA Player Stats Dashboard")

# Get unique options for dropdowns
fifa_versions = sorted(df['fifa_version'].unique())
player_positions = sorted(df['player_positions_1'].dropna().unique()) if not df.empty else []
league_names = sorted(df['league_name'].dropna().unique()) if not df.empty else []
club_names = sorted(df['club_name'].dropna().unique()) if not df.empty else []

# Function to plot the chart
def plot_chart(data):
    fig = px.bar(data, x='short_name', y=['overall', 'potential', 'value_eur', 'age'],
                 barmode='group', labels={'value_eur': 'Value (€)'})
    st.plotly_chart(fig)

# Display a default chart with full data
st.subheader("Player Stats - Full Data")
plot_chart(df)

# Add a horizontal line
st.markdown("---")

# Add slicers to filter data
st.sidebar.header("Filters")

# Dropdowns for selecting FIFA version, player positions, league name, and club name
fifa_version = st.sidebar.selectbox("Select FIFA Version", options=[''] + fifa_versions)
player_position = st.sidebar.selectbox("Select Player Position", options=[''] + player_positions if fifa_version else [], index=0 if player_positions else -1)
league_name = st.sidebar.selectbox("Select League Name", options=[''] + league_names if player_position else [], index=0 if league_names else -1)
club_name = st.sidebar.selectbox("Select Club Name", options=[''] + club_names if league_name and league_name != 'N/A' else [], index=0 if club_names else -1)

# Filter data based on selected options
filtered_df = df[
    (df['fifa_version'] == fifa_version) &
    (df['player_positions_1'] == player_position) &
    (df['league_name'] == league_name) &
    (df['club_name'] == club_name)
]

# Display the filtered chart
st.subheader("Filtered Player Stats")
if not filtered_df.empty:
    plot_chart(filtered_df)
else:
    st.write("No data available for the selected filters.")




import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import altair as alt

# Assuming df is your DataFrame
df_clean = df[['value_eur', 'overall', 'potential', 'age', 'wage_eur', 'international_reputation', 'player_positions_1', 'skill_moves']].dropna()

# Convert categorical variable to one-hot encoded variables
df_clean = pd.get_dummies(df_clean, columns=['player_positions_1'], drop_first=True)

# Define the regression model
formula = 'value_eur ~ overall + potential + age + wage_eur + international_reputation + skill_moves + ' + \
          ' + '.join(df_clean.columns[df_clean.columns.str.startswith('player_positions_1')])
model = smf.ols(formula=formula, data=df_clean).fit()

# Streamlit app
st.title('Player Value Prediction')

# Input fields for the player's details
overall = st.number_input('Overall', min_value=0, max_value=100, value=75)
potential = st.number_input('Potential', min_value=0, max_value=100, value=80)
age = st.number_input('Age', min_value=16, max_value=50, value=25)
wage_eur = st.number_input('Wage (EUR)', min_value=0, value=50000)
international_reputation = st.selectbox('International Reputation', options=[1, 2, 3, 4, 5])
skill_moves = st.selectbox('Skill Moves', options=[1, 2, 3, 4, 5])

# Dropdown for player position (one-hot encoded)
player_position = st.selectbox('Player Position', options=df['player_positions_1'].unique())

# Create a DataFrame for prediction based on the input values
input_data = pd.DataFrame({
    'overall': [overall],
    'potential': [potential],
    'age': [age],
    'wage_eur': [wage_eur],
    'international_reputation': [international_reputation],
    'skill_moves': [skill_moves],
    **{f'player_positions_1_{pos}': [1 if player_position == pos else 0] for pos in df['player_positions_1'].unique()}
})

# Ensure all columns are present in input_data
for col in model.model.exog_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Button to trigger prediction
if st.button('Predict Value (EUR)'):
    try:
        predicted_value = model.predict(input_data)[0]
        st.write(f"Predicted Value (EUR): {predicted_value:,.2f}")
        
        # Prepare data for chart
        df_chart = pd.DataFrame({
            'Type': [ 'Predicted Value'],
            'Value (EUR)': [predicted_value]
        })

        # Create a bar chart to compare actual vs. predicted value
        chart = alt.Chart(df_chart).mark_bar().encode(
            x='Type',
            y='Value (EUR)',
            color='Type'
        ).properties(
            width=alt.Step(80)  # Controls the width of the bars
        )

        st.altair_chart(chart)
    except Exception as e:
        st.error(f"Error during prediction: {e}")


        st.altair_chart(chart)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
