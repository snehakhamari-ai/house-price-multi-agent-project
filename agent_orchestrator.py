import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agents.data_agent.data_agent import DataAgent
from Agents.feature_agent.feature_agent import FeatureAgent
from Agents.training_agent.training_agent import TrainingAgent
from tools.custom_logger import Logger
from tools.memory_store import MemoryStore


def main():
    logger = Logger()
    memory = MemoryStore()

    logger.log("Starting Multi-Agent House Price Pipeline...")

    # Step 1: Load Data
    data_agent = DataAgent(logger, memory)
    df = data_agent.load_data()

    # Step 2: Feature Engineering
    feature_agent = FeatureAgent(logger, memory)
    df_processed = feature_agent.process_features(df)

    # Step 3: Train Model
    training_agent = TrainingAgent(logger, memory)
    model = training_agent.train_model(df_processed)

    logger.log("Pipeline Completed Successfully!")
    logger.log(f"Model Saved: {model}")


if __name__ == "__main__":
    main()
