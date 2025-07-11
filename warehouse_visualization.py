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
    
# Episode selection
episode_files = [f for f in os.listdir("training_data") if f.startswith("episode_") and f.endswith(".pkl")]
episode_numbers = [int(f.split("_")[1].split(".")[0]) for f in episode_files]
episode_numbers.sort()

if episode_numbers:
    # Show actual file numbers that match episode data exactly
    # No conversion needed - dropdown number = file number = episode number
    display_options = ["Latest"] + [f"Episode {num + 1}" for num in episode_numbers]  # +1 because files are 0-indexed
    
    selected_display = st.sidebar.selectbox(
        "Select Episode", 
        display_options,
        index=0
    )
    
    # Convert display selection back to file number for loading
    if selected_display == "Latest":
        selected_episode = "Latest"
    else:
        # Extract episode number from "Episode 787" format  
        displayed_num = int(selected_display.split()[1])
        selected_episode = displayed_num  # Direct mapping: Episode 787 → episode_787.pkl
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
    st.metric("Total Episodes", progress["total_episodes"])
with col3:
    st.metric("Latest Reward", f"{progress['latest_reward']:.1f}")
with col4:
    st.metric("Success Rate", f"{progress['success_rate']:.1%}")

# Learning Timeline Section
st.markdown("---")
st.markdown("## 🎯 **Learning Timeline Reference**")

# Dynamically load successful episodes from training data
def get_successful_episodes():
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

# Get actual successful episodes from current training
successful_episodes = get_successful_episodes()

# CRITICAL FIX: Only show episodes from current training session
current_training_episodes = progress.get('total_episodes', 0)
if current_training_episodes > 0:
    # Filter out episodes beyond current training - these are from old sessions
    successful_episodes = [ep for ep in successful_episodes if ep <= current_training_episodes]

# Create clean, progressive timeline
if len(successful_episodes) == 0:
    # CLEAN START - No successes yet in current training
    if current_training_episodes < 10:
        timeline_text = "🚀 **Training Starting...** \n\nSuccessful delivery episodes will appear here as training progresses."
    elif current_training_episodes < 100:
        timeline_text = f"📚 **Early Training** ({current_training_episodes} episodes completed)\n\nSuccessful delivery episodes will appear here when achieved."
    else:
        timeline_text = f"📚 **Learning in Progress** ({current_training_episodes} episodes completed)\n\nNo successful deliveries yet in this training session - keep training!"
else:
    # PROGRESSIVE SUCCESS LIST - Show each success as it happens  
    timeline_text = f"🎯 **Successful Delivery Episodes** ({len(successful_episodes)} total in current training):\n\n"
    
    for i, episode in enumerate(successful_episodes, 1):
        if i == 1:
            timeline_text += f"🎉 **Episode {episode}** - First successful delivery!\n"
        else:
            ordinals = ['Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth']
            ordinal = ordinals[min(i-2, 8)] if i <= 10 else f"{i}th"
            timeline_text += f"🎯 **Episode {episode}** - {ordinal} delivery\n"
    
    timeline_text += f"\n📊 **Current Training Progress:**\n"
    timeline_text += f"- Total Episodes: {current_training_episodes}\n"
    timeline_text += f"- Success Rate: {len(successful_episodes)/max(current_training_episodes, 1)*100:.1f}%\n"
    timeline_text += f"\n💡 *Click any episode number above in the dropdown to view its animation!*"

st.markdown(timeline_text)
st.markdown("*Note: Only episodes from current training session (≤ Episode {}) are shown here.*".format(current_training_episodes) if current_training_episodes > 0 else "*Note: Timeline shows current training session progress.*")
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