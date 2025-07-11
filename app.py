import streamlit as st
import mlflow
import plotly.express as px
import pandas as pd
from PIL import Image
import time

st.set_page_config(layout="wide", page_title="Warehouse MLOps Dashboard")

# Navigation
st.sidebar.title("🏭 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["MLOps Dashboard", "Training Visualization"]
)

if page == "Training Visualization":
    st.markdown("## 🎯 Redirecting to Training Visualization...")
    st.markdown("Please run the following command to start the training visualization:")
    st.code("streamlit run warehouse_visualization.py --server.port 8502", language="bash")
    st.markdown("Or click the link: [Training Visualization](http://localhost:8502)")
    st.stop()

st.title("Warehouse Digital Twin - MLOps Dashboard")

# Auto-refresh controls
col1, col2 = st.columns([3, 1])
with col1:
    auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=False)
with col2:
    if st.button("🔄 Refresh Now"):
        st.rerun()

if auto_refresh:
    time.sleep(10)
    st.rerun()

# MLflow connection with proxy bypass
import os
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

mlflow_tracking_uri = "http://127.0.0.1:5001"
# Set tracking URI globally for MLflow
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)

# Sidebar for experiment selection
st.sidebar.markdown("## 🔍 Connection Status")

try:
    # Test basic MLflow connection
    experiments = client.search_experiments()
    st.sidebar.success(f"✅ MLflow Connected: {len(experiments)} experiments found")
    
    # Debug information
    st.sidebar.markdown("**Debug Info:**")
    st.sidebar.code(f"MLflow URI: {mlflow_tracking_uri}")
    st.sidebar.code(f"Experiments: {[exp.name for exp in experiments]}")
    
    experiment_names = [exp.name for exp in experiments]
    if experiment_names:
        selected_experiment_name = st.sidebar.selectbox("Select Experiment", experiment_names)
        selected_experiment = client.get_experiment_by_name(selected_experiment_name)
        
        # Show experiment details
        if selected_experiment:
            st.sidebar.info(f"Experiment ID: {selected_experiment.experiment_id}")
    else:
        selected_experiment = None
        st.sidebar.warning("No experiments found. Please run training first to create experiments.")
        st.sidebar.markdown("**Expected experiments:**")
        st.sidebar.markdown("- `Warehouse_PPO_Adaptive_Learning`")
        st.sidebar.markdown("- `Warehouse_Smart_Learning`")
        st.sidebar.markdown("- `Warehouse_PPO_Learning`")
        
except Exception as e:
    st.sidebar.error(f"❌ MLflow Connection Failed")
    st.sidebar.error(f"Error: {str(e)}")
    st.sidebar.markdown("**Troubleshooting:**")
    st.sidebar.markdown("1. Start MLflow server:")
    st.sidebar.code("mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri file:./mlruns")
    st.sidebar.markdown("2. Check if server is running:")
    st.sidebar.code("curl http://127.0.0.1:5001/health")
    st.sidebar.markdown("3. Run training to create experiments:")
    st.sidebar.code("python new_train.py")
    selected_experiment = None

# Main dashboard content
if selected_experiment:
    runs = client.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs:
        st.header(f"Experiment: {selected_experiment_name}")

        # Display runs in a table
        run_data = []
        for run in runs:
            # Handle timestamp formatting safely
            start_time_str = "N/A"
            if run.info.start_time:
                try:
                    from datetime import datetime
                    if isinstance(run.info.start_time, int):
                        # Convert milliseconds to datetime
                        start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)
                    else:
                        start_time = run.info.start_time
                    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    start_time_str = str(run.info.start_time)
            
            run_data.append({
                "Run ID": run.info.run_id,
                "Start Time": start_time_str,
                "Status": run.info.status,
            })
        st.dataframe(pd.DataFrame(run_data))

        # Select a run to view details
        selected_run_id = st.selectbox("Select a Run to View Details", [run.info.run_id for run in runs])
        selected_run = client.get_run(selected_run_id)

        if selected_run:
            st.header(f"Run Details: {selected_run_id}")

            # Display hyperparameters
            st.subheader("Hyperparameters")
            st.json(selected_run.data.params)

            # Display metrics
            st.subheader("Metrics")
            metrics_data = {}
            for key, value in selected_run.data.metrics.items():
                history = client.get_metric_history(selected_run_id, key)
                metrics_data[key] = [m.value for m in history]
                metrics_data[f"{key}_step"] = [m.step for m in history]

            # Plot metrics
            metric_to_plot = st.selectbox("Select a metric to plot", list(selected_run.data.metrics.keys()))
            if metric_to_plot:
                df = pd.DataFrame({
                    "step": metrics_data[f"{metric_to_plot}_step"],
                    metric_to_plot: metrics_data[metric_to_plot]
                })
                fig = px.line(df, x="step", y=metric_to_plot, title=f"{metric_to_plot} over time")
                st.plotly_chart(fig, use_container_width=True)

            # Display artifacts
            st.subheader("Artifacts")
            artifacts = client.list_artifacts(selected_run_id)
            
            if artifacts:
                for artifact in artifacts:
                    st.write(f"📁 **{artifact.path}**")
                    
                    try:
                        # Download the artifact to local path
                        local_path = client.download_artifacts(selected_run_id, artifact.path)
                        
                        if artifact.path.endswith(".txt"):
                            # Display text files directly
                            with open(local_path, 'r') as f:
                                content = f.read()
                            with st.expander(f"View {artifact.path}", expanded=False):
                                st.text(content)
                            
                            # Also provide download button
                            with open(local_path, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {artifact.path}",
                                    data=f.read(),
                                    file_name=artifact.path,
                                    mime="text/plain"
                                )
                        
                        elif artifact.path.endswith(".json"):
                            # Display JSON files nicely
                            import json
                            with open(local_path, 'r') as f:
                                json_data = json.load(f)
                            with st.expander(f"View {artifact.path}", expanded=False):
                                st.json(json_data)
                            
                            # Also provide download button
                            with open(local_path, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {artifact.path}",
                                    data=f.read(),
                                    file_name=artifact.path,
                                    mime="application/json"
                                )
                        
                        elif artifact.path.endswith(".gif"):
                            # Display GIF files
                            image = Image.open(local_path)
                            st.image(image, caption="Trained Agent Performance")
                            
                            # Also provide download button
                            with open(local_path, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {artifact.path}",
                                    data=f.read(),
                                    file_name=artifact.path,
                                    mime="image/gif"
                                )
                        
                        else:
                            # For other file types, just provide download
                            with open(local_path, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {artifact.path}",
                                    data=f.read(),
                                    file_name=artifact.path
                                )
                    
                    except Exception as e:
                        st.error(f"Error loading {artifact.path}: {str(e)}")
                    
                    st.write("---")  # Separator between artifacts
            else:
                st.info("No artifacts found for this run.")

    else:
        st.warning("No runs found for this experiment.")
else:
    st.error("Experiment not found.")

# Footer
st.markdown("---")
st.markdown("🚀 For real-time training visualization, visit: [Training Visualization](http://localhost:8502)")
