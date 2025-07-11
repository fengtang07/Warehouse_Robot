import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pickle
import json
import time
import os
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Warehouse Training Visualization",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Warehouse Training Visualization")

# Define helper functions at the top
def get_milestone_episodes():
    """Find key milestone episodes for any training session"""
    episode_files = [f for f in os.listdir("training_data") if f.startswith("episode_") and f.endswith(".pkl")]
    
    # Collect episode data
    episode_data = []
    for file in episode_files:
        try:
            file_num = int(file.split("_")[1].split(".")[0])
            with open(f"training_data/{file}", "rb") as f:
                data = pickle.load(f)
            episode_data.append({
                'episode': data['episode'],
                'success': data.get('success', False),
                'reward': data.get('final_reward', 0)
            })
        except:
            continue
    
    # Sort by episode number
    episode_data.sort(key=lambda x: x['episode'])
    
    # Find milestones
    milestones = {
        'first_attempt': 1,  # Always episode 1
        'first_success': None,
        'highest_reward': None,
        'latest_success': None
    }
    
    successful_episodes = [ep for ep in episode_data if ep['success']]
    
    if successful_episodes:
        # First successful episode
        milestones['first_success'] = successful_episodes[0]['episode']
        
        # Highest reward episode (among successful ones)
        highest_reward_ep = max(successful_episodes, key=lambda x: x['reward'])
        milestones['highest_reward'] = highest_reward_ep['episode']
        
        # Latest successful episode
        milestones['latest_success'] = successful_episodes[-1]['episode']
    
    return milestones

def get_successful_episodes():
    """Get list of all successful episodes (for timeline)"""
    successful_episodes = []
    episode_files = [f for f in os.listdir("training_data") if f.startswith("episode_") and f.endswith(".pkl")]
    
    for file in episode_files:
        try:
            file_num = int(file.split("_")[1].split(".")[0])
            with open(f"training_data/{file}", "rb") as f:
                data = pickle.load(f)
            if data.get('success', False):
                successful_episodes.append(data['episode'])
        except:
            continue
    
    return sorted(successful_episodes)

# Initialize session state for unique keys and animation control
if 'refresh_time' not in st.session_state:
    st.session_state.refresh_time = int(time.time())
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False

# Check if training data exists
if not os.path.exists("training_data"):
    st.warning("No training data found. Please run the training script first.")
    st.stop()

# Load progress data
try:
    with open("training_data/progress.json", "r") as f:
        progress = json.load(f)
except FileNotFoundError:
    st.warning("No progress data found. Training may not have started yet.")
    progress = {"current_episode": 0, "total_episodes": 0, "latest_reward": 0, "success_rate": 0}

# Sidebar controls
st.sidebar.header("🎛️ Visualization Controls")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)
    # Add auto-refresh placeholder
    placeholder = st.empty()

# Episode selection - FIXED to only show current training session episodes
episode_files = [f for f in os.listdir("training_data") if f.startswith("episode_") and f.endswith(".pkl")]
episode_numbers = [int(f.split("_")[1].split(".")[0]) for f in episode_files]

# FIXED: Use session range from progress.json for accurate filtering
current_episode = progress.get('current_episode', 0)
session_start = progress.get('session_start_episode', 1)
session_episodes = progress.get('total_episodes', 0)

if current_episode > 0:
    # Filter episodes to only show those from current training session
    episode_numbers = [ep for ep in episode_numbers if session_start <= ep <= current_episode]

episode_numbers.sort()

if episode_numbers:
    # Get dynamic milestone episodes for this training session FIRST
    milestones = get_milestone_episodes()
    
    # Show most recent episodes for better usability (max 300 episodes)
    if len(episode_numbers) > 300:
        # Show first 50, middle 100, last 150 episodes for better navigation
        middle_start = len(episode_numbers) // 2 - 50
        middle_end = len(episode_numbers) // 2 + 50
        recent_episodes = (episode_numbers[:50] + 
                          episode_numbers[middle_start:middle_end] + 
                          episode_numbers[-150:])
        
        # ALWAYS include milestone episodes (critical fix!)
        milestone_numbers = [ep for ep in milestones.values() if ep and ep in episode_numbers]
        recent_episodes.extend(milestone_numbers)
        
        episode_numbers = sorted(list(set(recent_episodes)))
    
    # Add milestone episodes for easy access
    milestone_episodes = []
    
    # Always include episode 1
    if 1 in episode_numbers:
        milestone_episodes.append(1)
    
    # Add dynamic milestones if they exist and are in current session
    for milestone_type, ep_num in milestones.items():
        if ep_num and ep_num in episode_numbers and ep_num not in milestone_episodes:
            milestone_episodes.append(ep_num)
    
    # Combine milestones with regular episodes, removing duplicates
    milestone_episodes.sort()
    remaining_episodes = [ep for ep in episode_numbers if ep not in milestone_episodes]
    
    # Create organized display options with key episodes at the top
    display_options = ["Latest"]
    
    # Add milestone episodes with descriptive labels
    if milestone_episodes:
        for ep in milestone_episodes:
            if ep == milestones['first_attempt']:
                display_options.append(f"Episode {ep} - Very First Attempt")
            elif ep == milestones['first_success']:
                display_options.append(f"Episode {ep} - 🎉 FIRST SUCCESS!")
            elif ep == milestones['highest_reward']:
                display_options.append(f"Episode {ep} - 🏆 HIGHEST REWARD!")
            elif ep == milestones['latest_success']:
                display_options.append(f"Episode {ep} - 🎯 LATEST SUCCESS!")
            else:
                display_options.append(f"Episode {ep}")
    
    # Add remaining episodes (without special labels)
    if remaining_episodes:
        display_options.extend([f"Episode {num}" for num in remaining_episodes])
    
    selectable_options = display_options
    
    # Add interactive milestone buttons
    st.sidebar.markdown("### 🎯 **Key Milestones**")
    
    # Initialize session state for selected episode if not exists
    if 'selected_episode_number' not in st.session_state:
        st.session_state.selected_episode_number = None
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    if 'persistent_episode_selection' not in st.session_state:
        st.session_state.persistent_episode_selection = None
    
    # Single row layout for buttons in sidebar with persistent selection
    if st.sidebar.button("🤖 First Attempt", key="btn_first", help=f"Episode {milestones['first_attempt']} - Very first attempt", use_container_width=True):
        st.session_state.persistent_episode_selection = milestones['first_attempt']
        st.session_state.selected_episode_number = milestones['first_attempt']
        st.session_state.button_clicked = True
    
    if milestones['first_success'] and st.sidebar.button("🎉 First Success", key="btn_success", help=f"Episode {milestones['first_success']} - First successful delivery", use_container_width=True):
        st.session_state.persistent_episode_selection = milestones['first_success']
        st.session_state.selected_episode_number = milestones['first_success']
        st.session_state.button_clicked = True
    
    if milestones['highest_reward'] and st.sidebar.button("🏆 Best Reward", key="btn_highest", help=f"Episode {milestones['highest_reward']} - Highest reward achieved", use_container_width=True):
        st.session_state.persistent_episode_selection = milestones['highest_reward']
        st.session_state.selected_episode_number = milestones['highest_reward']
        st.session_state.button_clicked = True
    
    if milestones['latest_success'] and st.sidebar.button("🎯 Latest Success", key="btn_latest", help=f"Episode {milestones['latest_success']} - Most recent success", use_container_width=True):
        st.session_state.persistent_episode_selection = milestones['latest_success']
        st.session_state.selected_episode_number = milestones['latest_success']
        st.session_state.button_clicked = True
    
    # Show detected milestones summary with availability check and debugging
    st.sidebar.markdown("**Detected Episodes:**")
    milestone_summary = []
    if milestones['first_attempt']:
        available = "✅" if milestones['first_attempt'] in episode_numbers else "❌"
        milestone_summary.append(f"First: {milestones['first_attempt']}{available}")
    if milestones['first_success']:
        available = "✅" if milestones['first_success'] in episode_numbers else "❌"
        milestone_summary.append(f"Success: {milestones['first_success']}{available}")
    if milestones['highest_reward']:
        available = "✅" if milestones['highest_reward'] in episode_numbers else "❌"
        milestone_summary.append(f"Best: {milestones['highest_reward']}{available}")
    if milestones['latest_success']:
        available = "✅" if milestones['latest_success'] in episode_numbers else "❌"
        milestone_summary.append(f"Latest: {milestones['latest_success']}{available}")
    
    if milestone_summary:
        st.sidebar.markdown(f"*{' | '.join(milestone_summary)}*")
    
    # Handle button selections and persistent episode choice
    default_index = 0
    
    # Check if we have a persistent episode selection to maintain
    if st.session_state.get('persistent_episode_selection'):
        target_episode = st.session_state.persistent_episode_selection
        
        # Find the display option that matches the persistent episode
        for option in selectable_options:
            if option != "Latest":
                try:
                    # Handle both "Episode X" and "Episode X - Description" formats
                    if " - " in option:
                        episode_part = option.split(" - ")[0]  # Get "Episode X" part
                    else:
                        episode_part = option  # Already just "Episode X"
                    
                    ep_num = int(episode_part.split()[1])  # Extract number
                    if ep_num == target_episode:
                        try:
                            default_index = selectable_options.index(option)
                            break
                        except ValueError:
                            default_index = 0
                except (IndexError, ValueError):
                    continue
    
    # Clear the temporary selection state (but keep persistent)
    if st.session_state.get('button_clicked'):
        st.session_state.selected_episode_number = None
        st.session_state.button_clicked = False

    selected_display = st.sidebar.selectbox(
        "Select Episode", 
        selectable_options,
        index=default_index,
        help="Choose any episode to see the robot's behavior at that stage!",
        key="episode_selectbox"
    )
    
    # Check if user manually changed the dropdown (override button selection)
    if st.session_state.get('persistent_episode_selection'):
        # Extract episode number from current selection
        current_episode = None
        if selected_display == "Latest":
            current_episode = "Latest"
        else:
            try:
                episode_part = selected_display.split(" - ")[0]
                current_episode = int(episode_part.split()[1])
            except (IndexError, ValueError):
                current_episode = "Latest"
        
        # If user manually selected something different, clear persistent selection
        if current_episode != st.session_state.persistent_episode_selection and current_episode != "Latest":
            st.session_state.persistent_episode_selection = None
    
    # Convert display selection back to file number for loading
    if selected_display == "Latest":
        selected_episode = "Latest"
    elif selected_display.startswith("───"):
        # Skip separator lines - this shouldn't happen but just in case
        selected_episode = "Latest"
    else:
        # Extract episode number from formats like "Episode 59 - 🎉 FIRST SUCCESS!"
        try:
            episode_part = selected_display.split(" - ")[0]  # Get "Episode 59" part
            selected_episode = int(episode_part.split()[1])  # Extract number
        except (IndexError, ValueError):
            # Fallback to original method
            selected_episode = int(selected_display.split()[1])
else:
    st.sidebar.warning("No episode data found")
    st.stop()

# Speed control for animation
animation_speed = st.sidebar.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)

# Display current training progress
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Episode", progress["current_episode"])
with col2:
    # FIXED: Show both session episodes and current episode number clearly
    session_episodes = progress.get("total_episodes", 0)
    session_start = progress.get("session_start_episode", 1)
    st.metric("Session Episodes", f"{session_episodes} (#{session_start}-{progress['current_episode']})")
with col3:
    st.metric("Latest Reward", f"{progress['latest_reward']:.1f}")
with col4:
    st.metric("Success Rate", f"{progress['success_rate']:.1%}")

# Learning Timeline Section
st.markdown("---")
st.markdown("## 🎯 **Learning Timeline Reference**")

# Get successful episodes for timeline display

# Get actual successful episodes from current training
successful_episodes = get_successful_episodes()

# FIXED: Use session range for accurate filtering  
current_episode = progress.get('current_episode', 0)
session_start = progress.get('session_start_episode', 1)
session_episodes = progress.get('total_episodes', 0)

if current_episode > 0:
    # Filter out episodes beyond current training session range
    successful_episodes = [ep for ep in successful_episodes if session_start <= ep <= current_episode]

# Create FOCUSED timeline with key milestones only
if len(successful_episodes) == 0:
    # CLEAN START - No successes yet in current training
    if session_episodes < 10:
        timeline_text = "🚀 **Training Starting...** \n\nKey milestone episodes will appear here as training progresses."
    elif session_episodes < 100:
        timeline_text = f"📚 **Early Training** ({session_episodes} episodes completed)\n\nKey milestone episodes will appear here when achieved."
    else:
        timeline_text = f"📚 **Learning in Progress** ({session_episodes} episodes completed)\n\nNo successful deliveries yet in this training session - keep training!"
else:
    # MILESTONE-FOCUSED SUCCESS LIST - Show only key episodes
    timeline_text = f"🎯 **Key Training Milestones** ({len(successful_episodes)} total successes):\n\n"
    
    # Get episode rewards for finding highest reward
    episode_rewards = {}
    episode_files = [f for f in os.listdir("training_data") if f.startswith("episode_") and f.endswith(".pkl")]
    for file in episode_files:
        try:
            file_num = int(file.split("_")[1].split(".")[0])
            with open(f"training_data/{file}", "rb") as f:
                data = pickle.load(f)
            if data.get('success', False) and data['episode'] in successful_episodes:
                episode_rewards[data['episode']] = data.get('final_reward', 0)
        except:
            continue
    
    # Find key episodes
    first_success = successful_episodes[0] if successful_episodes else None
    last_success = successful_episodes[-1] if successful_episodes else None
    highest_reward_episode = max(episode_rewards.keys(), key=lambda x: episode_rewards[x]) if episode_rewards else None
    
    # Show key milestones
    if first_success:
        timeline_text += f"🎉 **Episode {first_success}** - First successful delivery breakthrough!\n"
    
    if highest_reward_episode and highest_reward_episode != first_success:
        reward_value = episode_rewards[highest_reward_episode]
        timeline_text += f"🏆 **Episode {highest_reward_episode}** - Highest reward achieved ({reward_value:.1f})\n"
    
    if last_success and last_success != first_success and last_success != highest_reward_episode:
        timeline_text += f"🎯 **Episode {last_success}** - Most recent successful delivery\n"
    
    timeline_text += f"\n📊 **Current Training Progress:**\n"
    timeline_text += f"- Session Episodes: {session_episodes} (#{session_start}-{current_episode})\n"
    timeline_text += f"- Total Successes: {len(successful_episodes)}\n"
    timeline_text += f"- Success Rate: {len(successful_episodes)/max(session_episodes, 1)*100:.1f}%\n"
    timeline_text += f"\n💡 *Click any milestone episode number above in the dropdown to view its animation!*"

st.markdown(timeline_text)
st.markdown("*Note: Only episodes from current training session (#{}-{}) are shown here.*".format(session_start, current_episode) if current_episode > 0 else "*Note: Timeline shows current training session progress.*")
st.markdown("---")

# Function to load episode data
@st.cache_data
def load_episode_data(episode_num):
    if episode_num == "Latest":
        try:
            with open("training_data/latest_episode.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
    else:
        try:
            # Direct mapping: Episode 787 (display) → episode_787.pkl (file)
            with open(f"training_data/episode_{episode_num}.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

# Function to create warehouse layout plot
def create_warehouse_plot(episode_data, current_step=None):
    """Create the warehouse layout with robot path"""
    if not episode_data:
        # Return empty figure with same fixed layout
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(range=[-6, 7], title="X Position", fixedrange=True),
            yaxis=dict(range=[-6, 7], title="Y Position", fixedrange=True),
            title="Warehouse Environment",
            width=800,
            height=600,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig
    
    # Create figure with STRICT fixed bounds - NO SCALING ALLOWED
    fig = go.Figure()
    
    # FORCE fixed layout before adding any data
    fig.update_layout(
        xaxis=dict(
            range=[-6, 7], 
            title="X Position",
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            autorange=False,
            constrain='domain',  # Constrain to domain
            type='linear',
            dtick=2,  # Fixed tick spacing
            tick0=-6,  # Start ticks at -6
            showspikes=False,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            range=[-6, 7], 
            title="Y Position",
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            autorange=False,
            constrain='domain',  # Constrain to domain
            type='linear',
            dtick=2,  # Fixed tick spacing  
            tick0=-6,  # Start ticks at -6
            showspikes=False,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        title=dict(
            text="Warehouse Environment",
            x=0.5,  # Center title
            font=dict(size=16)
        ),
        showlegend=True,
        width=800,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=60),
        font=dict(size=12),
        autosize=False,
        uirevision='warehouse_layout',  # Constant revision
        dragmode=False,  # Disable all dragging
        selectdirection='d',  # 'd' for diagonal selection
        hovermode='closest'
    )
    
    # Add invisible boundary markers to lock axis bounds (CRITICAL!)
    fig.add_trace(go.Scatter(
        x=[-6, -6, 7, 7], 
        y=[-6, 7, -6, 7],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0.1),  # Completely invisible
        showlegend=False,
        hoverinfo='skip',
        name=''  # Empty name
    ))
    
    # Add obstacles (shelves)
    for i, obs in enumerate(episode_data['obstacles']):
        fig.add_shape(
            type="rect",
            x0=obs[0] - 0.5, y0=obs[1] - 0.25,
            x1=obs[0] + 0.5, y1=obs[1] + 0.25,
            fillcolor="gray", line=dict(color="black", width=2),
            layer="below"
        )
        fig.add_annotation(
            x=obs[0], y=obs[1],
            text=f"Shelf {i+1}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Add delivery area
    delivery_pos = episode_data['delivery_pos']
    fig.add_shape(
        type="rect",
        x0=delivery_pos[0] - 0.5, y0=delivery_pos[1] - 0.5,
        x1=delivery_pos[0] + 0.5, y1=delivery_pos[1] + 0.5,
        fillcolor="green", opacity=0.5,
        line=dict(color="darkgreen", width=2),
        layer="below"
    )
    fig.add_annotation(
        x=delivery_pos[0], y=delivery_pos[1],
        text="🎯 Delivery",
        showarrow=False,
        font=dict(size=12, color="white")
    )
    
    # Get position data
    positions = episode_data['positions']
    if not positions:
        return fig
    
    # Filter positions up to current step if specified
    if current_step is not None:
        positions = [pos for pos in positions if pos['step'] <= current_step]
    
    if not positions:
        return fig
    
    # Extract trajectory data and CLIP to bounds to prevent scaling
    x_coords = [max(-5.9, min(6.9, pos['agent_pos'][0])) for pos in positions]
    y_coords = [max(-5.9, min(6.9, pos['agent_pos'][1])) for pos in positions]
    steps = [pos['step'] for pos in positions]
    
    # Add robot trajectory
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='lines+markers',
        name='Robot Path',
        line=dict(color='blue', width=3),
        marker=dict(size=4, color='blue')
    ))
    
    # Add package positions (clipped to bounds)
    package_x = [max(-5.9, min(6.9, pos['package_pos'][0])) for pos in positions if not pos['has_package']]
    package_y = [max(-5.9, min(6.9, pos['package_pos'][1])) for pos in positions if not pos['has_package']]
    
    if package_x and package_y:
        fig.add_trace(go.Scatter(
            x=package_x, y=package_y,
            mode='markers',
            name='Package',
            marker=dict(size=15, color='red', symbol='square')
        ))
    
    # Add current robot position (clipped to bounds)
    if positions:
        current_pos = positions[-1]
        robot_x = max(-5.9, min(6.9, current_pos['agent_pos'][0]))
        robot_y = max(-5.9, min(6.9, current_pos['agent_pos'][1]))
        
        fig.add_trace(go.Scatter(
            x=[robot_x],
            y=[robot_y],
            mode='markers',
            name='Current Robot',
            marker=dict(size=20, color='orange', symbol='triangle-up')
        ))
        
        # Add direction indicator (also clipped)
        angle = current_pos['agent_angle']
        dx = 0.4 * np.cos(angle)  # Slightly smaller to stay in bounds
        dy = 0.4 * np.sin(angle)
        arrow_x = max(-5.9, min(6.9, robot_x + dx))
        arrow_y = max(-5.9, min(6.9, robot_y + dy))
        
        fig.add_annotation(
            x=arrow_x,
            y=arrow_y,
            ax=robot_x,
            ay=robot_y,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="orange"
        )
    
    return fig

# Function to create metrics over time plot
def create_metrics_plot(episode_data):
    """Create metrics plots for the episode with STRICT bounds"""
    # Create empty figure with LOCKED layout first
    fig = go.Figure()
    
    # FORCE strict layout immediately - NO AUTO-SCALING ALLOWED
    fig.update_layout(
        xaxis=dict(
            range=[0, 800],
            title="Step",
            fixedrange=True,
            autorange=False,
            constrain='domain',
            type='linear',
            dtick=100,  # Fixed tick spacing
            tick0=0,
            showspikes=False
        ),
        yaxis=dict(
            range=[0, 8],
            title="Distance", 
            side='left',
            fixedrange=True,
            autorange=False,
            constrain='domain',
            type='linear',
            dtick=1,  # Fixed tick spacing
            tick0=0,
            showspikes=False
        ),
        yaxis2=dict(
            range=[0, 1],
            title="Speed", 
            side='right', 
            overlaying='y',
            fixedrange=True,
            autorange=False,
            constrain='domain',
            type='linear',
            dtick=0.2,  # Fixed tick spacing
            tick0=0,
            showspikes=False
        ),
        title=dict(
            text="Episode Metrics",
            x=0.5,
            font=dict(size=14)
        ),
        showlegend=True,
        height=400,
        width=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=50),
        autosize=False,
        uirevision='metrics_layout',
        dragmode=False,
        hovermode='closest'
    )
    
    # Lock bounds with invisible anchors
    fig.add_trace(go.Scatter(
        x=[0, 800], y=[0, 8],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0.1),
        showlegend=False,
        hoverinfo='skip',
        name=''
    ))
    
    if not episode_data or not episode_data['positions']:
        return fig
    
    positions = episode_data['positions']
    
    # Calculate metrics with CLIPPING to prevent out-of-bounds values
    steps = [min(799, max(0, pos['step'])) for pos in positions]  # Clip steps
    distances_to_package = []
    distances_to_delivery = []
    speeds = []
    
    for pos in positions:
        if not pos['has_package']:
            dist_pkg = np.linalg.norm(np.array(pos['agent_pos']) - np.array(pos['package_pos']))
            dist_pkg = min(7.9, max(0, dist_pkg))  # Clip distance
            distances_to_package.append(dist_pkg)
            distances_to_delivery.append(None)
        else:
            distances_to_package.append(None)
            dist_del = np.linalg.norm(np.array(pos['agent_pos']) - np.array(episode_data['delivery_pos']))
            dist_del = min(7.9, max(0, dist_del))  # Clip distance
            distances_to_delivery.append(dist_del)
        
        # Calculate speed (clipped)
        if len(speeds) == 0:
            speeds.append(0)
        else:
            prev_pos = positions[len(speeds) - 1]
            distance = np.linalg.norm(np.array(pos['agent_pos']) - np.array(prev_pos['agent_pos']))
            speed = min(0.9, max(0, distance / 0.1))  # Clip speed
            speeds.append(speed)
    
    # Distance to package
    valid_pkg_steps = [s for s, d in zip(steps, distances_to_package) if d is not None]
    valid_pkg_distances = [d for d in distances_to_package if d is not None]
    
    if valid_pkg_steps:
        fig.add_trace(go.Scatter(
            x=valid_pkg_steps, y=valid_pkg_distances,
            mode='lines', name='Distance to Package',
            line=dict(color='red')
        ))
    
    # Distance to delivery
    valid_del_steps = [s for s, d in zip(steps, distances_to_delivery) if d is not None]
    valid_del_distances = [d for d in distances_to_delivery if d is not None]
    
    if valid_del_steps:
        fig.add_trace(go.Scatter(
            x=valid_del_steps, y=valid_del_distances,
            mode='lines', name='Distance to Delivery',
            line=dict(color='green')
        ))
    
    # Speed
    fig.add_trace(go.Scatter(
        x=steps, y=speeds,
        mode='lines', name='Speed',
        line=dict(color='blue'),
        yaxis='y2'
    ))
    
    return fig

# Main visualization
def show_visualization():
    # Load episode data
    episode_data = load_episode_data(selected_episode)
    
    if not episode_data:
        st.error("Could not load episode data")
        return
    
    # Display episode info
    st.subheader(f"Episode {episode_data['episode']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Reward", f"{episode_data['final_reward']:.1f}")
    with col2:
        st.metric("Success", "✅" if episode_data['success'] else "❌")
    with col3:
        st.metric("Steps", len(episode_data['positions']))
    
    # Create two columns for plots
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Warehouse Layout")
        
        # Info about the visualization
        st.success("🔒 **ZERO SCALING** - Completely locked axes and clipped data prevent any shifting or flashing! Extended episodes (800 steps) for full delivery cycles.")
        
        # Animation controls
        col_anim1, col_anim2 = st.columns([3, 1])
        with col_anim1:
            animate_button = st.button("🎬 Animate Episode")
        with col_anim2:
            if st.button("⏹️ Stop Animation"):
                st.session_state.animation_running = False
        
        if animate_button:
            st.session_state.animation_running = True
            plot_container = st.empty()
            progress_container = st.empty()
            
            total_steps = len(episode_data['positions'])
            # Reduce frame count for smoother animation (max 25 frames)
            frame_skip = max(1, total_steps // 25)
            
            # Pre-create all frames to ensure consistency
            frames = []
            for i in range(0, total_steps, frame_skip):
                frames.append(i)
            
            # Animate with consistent timing
            for frame_idx, step in enumerate(frames):
                if not st.session_state.get('animation_running', False):
                    break
                    
                fig = create_warehouse_plot(episode_data, step)
                
                # Ensure exact same configuration for each frame
                fig.update_layout(
                    autosize=False,
                    width=800,
                    height=600,
                    showlegend=True,
                    transition_duration=0,  # Disable transitions
                    dragmode=False,
                    uirevision='warehouse_layout'
                )
                
                # Fixed config for animation frames
                config = {
                    'displayModeBar': False,
                    'staticPlot': False,
                    'responsive': False,
                    'autosizable': False,
                    'fillFrame': False,
                    'frameMargins': 0
                }
                
                plot_container.plotly_chart(
                    fig, 
                    use_container_width=False, 
                    key=f"animation_step_{step}_{frame_idx}",
                    config=config
                )
                
                # Update progress
                progress = frame_idx / len(frames)
                progress_container.progress(progress)
                
                # Consistent timing
                time.sleep(max(0.1, 0.3 / animation_speed))
            
            # Show final state
            if st.session_state.get('animation_running', True):
                fig = create_warehouse_plot(episode_data)
                fig.update_layout(
                    autosize=False,
                    width=800,
                    height=600,
                    showlegend=True,
                    transition_duration=0,
                    dragmode=False,
                    uirevision='warehouse_layout'
                )
                
                config = {
                    'displayModeBar': False,
                    'staticPlot': False,
                    'responsive': False,
                    'autosizable': False,
                    'fillFrame': False,
                    'frameMargins': 0
                }
                
                plot_container.plotly_chart(
                    fig, 
                    use_container_width=False, 
                    key="final_animation_state",
                    config=config
                )
                progress_container.progress(1.0)
                
                # Auto-clear after 2 seconds
                time.sleep(2)
                progress_container.empty()
            
            st.session_state.animation_running = False
        else:
            # Show complete path with consistent layout
            fig = create_warehouse_plot(episode_data)
            
            # Ensure same layout as animation frames - LOCK ALL SETTINGS
            fig.update_layout(
                autosize=False,
                width=800,
                height=600,
                showlegend=True,
                transition_duration=0,
                dragmode=False,
                uirevision='warehouse_layout'
            )
            
            plot_key = f"warehouse_plot_{st.session_state.get('refresh_time', 0)}"
            # Use fixed config to prevent any resizing
            config = {
                'displayModeBar': False,
                'staticPlot': False,
                'responsive': False,
                'autosizable': False,
                'fillFrame': False,
                'frameMargins': 0
            }
            st.plotly_chart(fig, use_container_width=False, key=plot_key, config=config)
    
    with col2:
        st.subheader("📊 Metrics")
        metrics_fig = create_metrics_plot(episode_data)
        metrics_key = f"metrics_plot_{st.session_state.get('refresh_time', 0)}"
        
        # Fixed config for metrics plot too
        config = {
            'displayModeBar': False,
            'staticPlot': False,
            'responsive': False,
            'autosizable': False,
            'fillFrame': False,
            'frameMargins': 0
        }
        st.plotly_chart(metrics_fig, use_container_width=False, key=metrics_key, config=config)
        
        # Show position data table
        if episode_data['positions']:
            st.subheader("📍 Position Data")
            pos_df = pd.DataFrame([
                {
                    'Step': pos['step'],
                    'X': f"{pos['agent_pos'][0]:.2f}",
                    'Y': f"{pos['agent_pos'][1]:.2f}",
                    'Angle': f"{pos['agent_angle']:.2f}",
                    'Has Package': pos['has_package']
                }
                for pos in episode_data['positions'][-10:]  # Show last 10 steps
            ])
            st.dataframe(pos_df, use_container_width=True)

# Auto-refresh logic
if auto_refresh:
    # Clear cache to get fresh data
    st.cache_data.clear()
    
    # Show visualization with timestamp for unique keys
    current_time = int(time.time())
    
    # Store current time in session state for unique keys
    if 'refresh_time' not in st.session_state:
        st.session_state.refresh_time = current_time
    
    # Update keys for unique identification
    st.session_state.refresh_time = current_time
    
    # Show visualization
    show_visualization()
    
    # Auto-refresh
    time.sleep(refresh_interval)
    st.rerun()
else:
    show_visualization()

# Footer
st.markdown("---")
st.markdown("🤖 Real-time warehouse training visualization powered by Streamlit and Plotly") 
