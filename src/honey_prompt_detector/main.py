#!/usr/bin/env python3
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import argparse

# Local imports
from .utils.config import Config
from .utils.logging import setup_logger
from .utils.validation import InputValidator
from .agents.token_designer import TokenDesignerAgent
from .agents.context_evaluator import ContextEvaluatorAgent
from .core.orchestrator import DetectionOrchestrator
from .monitoring.metrics import MetricsCollector
from .monitoring.alerts import AlertManager
from .monitoring.dynamic_adaptation import DynamicAdaptation

logger = setup_logger(__name__)


class HoneyPromptSystem:
    """
    Represents a system for detecting prompt injection attacks using various
    AI-based agents and metrics collection.

    The HoneyPromptSystem class is responsible for initializing, starting,
    monitoring, and managing the detection of potential prompt injection
    attacks. It orchestrates multiple components, such as token design,
    context evaluation, alert management, and performance metrics tracking.

    Attributes:
      config: Configuration settings.
      token_designer: Agent to generate honey-prompt tokens.
      context_evaluator: Primary agent for LLM-based evaluation.
      metrics: Metrics collector.
      alert_manager: Manager for alerts.
      orchestrator: Coordinates token embedding, evaluation, and detection.
      is_initialized: Indicates whether the system has been initialized.
    """
    def __init__(self, env_path: Optional[Path] = None, custom_config: Optional[Dict[str, Any]] = None):
        # Load and validate configuration
        self.config = Config(env_path)
        if custom_config:
            for key, value in custom_config.items():
                setattr(self.config, key, value)

        # Initialize components
        self.token_designer = TokenDesignerAgent(
            api_key=self.config.openai_api_key,
            model_name=self.config.model_name
        )
        self.context_evaluator = ContextEvaluatorAgent(
            api_key=self.config.openai_api_key,
            model_name=self.config.model_name
        )
        self.metrics = MetricsCollector(metrics_file=Path('detection_metrics.json'))
        self.alert_manager = AlertManager(self.config.as_dict().get('alert_settings', {}))
        self.orchestrator = DetectionOrchestrator(
            token_designer=self.token_designer,
            context_evaluator=self.context_evaluator,
            config=self.config
        )
        self.is_initialized = False
        logger.info("HoneyPromptSystem initialized")

    async def start(self) -> bool:
        """Initialize system components and start the system."""
        try:
            await self.orchestrator.initialize_system()
            self.is_initialized = True
            self.metrics.record_system_start()
            logger.info("System started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start system: {str(e)}")
            return False

    async def monitor_text(self, text: str) -> Dict[str, Any]:
        """Monitor text for potential prompt injection attacks."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call start() first.")

        start_time = datetime.now()

        try:
            # Validate input
            validation_result = InputValidator.validate_text_input(text)
            if not validation_result.is_valid:
                logger.error(f"Invalid input: {validation_result.errors}")
                return {'detection': False, 'error': validation_result.errors[0]}

            # Perform detection via the orchestrator
            result = await self.orchestrator.monitor_text(text)

            # Record metrics
            self.metrics.record_detection(result)
            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_performance(response_time)

            # Send alert if an attack is detected
            if result['detection']:
                await self.alert_manager.send_alert(result)

            return result

        except Exception as e:
            logger.error(f"Error monitoring text: {str(e)}")
            self.metrics.record_performance((datetime.now() - start_time).total_seconds(), is_error=True)
            return {'detection': False, 'error': str(e), 'confidence': 0.0}

    async def get_system_status(self) -> Dict[str, Any]:
        """Return current system status and collected metrics."""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        return {
            'status': 'running',
            'metrics_summary': self.metrics.get_summary(),
            'active_honey_prompts': len(self.orchestrator.honey_prompts),
            'total_detections': self.metrics.metrics.get('detections', {}).get('total', 0)
        }

    async def run_async(args=None):
        system = HoneyPromptSystem(args.env)
        if not await system.start():
            logger.error("System startup failed")
            return

        # Start dynamic adaptation as a background task.
        adaptation = DynamicAdaptation(system.metrics, system.orchestrator, adaptation_interval=60)
        adaptation_task = asyncio.create_task(adaptation.run())

        try:
            # Interactive loop
            print("\nHoney-Prompt Detection System")
            print("===========================")
            print("Enter text to analyze (or 'quit' to exit)")
            print("Commands:")
            print("  status  - Show system status")
            print("  metrics - Show current metrics")
            print("  quit    - Exit the system")

            while True:
                user_input = input("\nCommand> ")
                if not user_input.strip():
                    continue  # Ignore empty input

                command = user_input.strip().lower()

                if command == 'quit':
                    break
                elif command == 'status':
                    status = await system.get_system_status()
                    print("\nSystem Status:")
                    print("--------------")
                    for key, value in status.items():
                        print(f"{key}: {value}")
                elif command == 'metrics':
                    metrics = system.metrics.get_summary()
                    print("\nSystem Metrics:")
                    print("--------------")
                    print(json.dumps(metrics, indent=2))
                else:
                    result = await system.monitor_text(command)
                    if result.get('error'):
                        print(f"Error: {result['error']}")
                    elif result['detection']:
                        print("\n⚠️  Potential prompt injection detected!")
                        print(f"Confidence: {result['confidence']:.2f}")
                        print(f"Explanation: {result.get('explanation', 'None provided')}")
                        print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
                    else:
                        print("No prompt injection detected")

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error: {str(e)}")

        # Cleanup and shutdown
        adaptation.running = False
        await adaptation_task
        await system.stop()
        print("\nSystem shutdown complete")

    async def stop(self):
        """Gracefully shut down the system."""
        if self.is_initialized:
            logger.info("Shutting down honey-prompt detection system")
            self.metrics.save_metrics()
            self.is_initialized = False


def main():
    """
    Synchronous entry point for the Honey-Prompt Detection System.
    Parses CLI arguments and then calls an async function to run the system.
    """
    parser = argparse.ArgumentParser(description="Honey-Prompt Detection System")
    parser.add_argument("--env", type=Path, help="Path to .env file")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--run-experiments", action="store_true", help="Run comprehensive experiments")

    args = parser.parse_args()

    async def run_async():
        if args.run_experiments:
            # Import and run experiments from the examples module.
            from examples.basic_usage import ExperimentRunner
            experiment = ExperimentRunner()
            await experiment.run_experiments()
            return

        # Otherwise, run the system interactively or for a single text analysis.
        system = HoneyPromptSystem(args.env)
        if not await system.start():
            logger.error("System startup failed")
            return

        if args.text:
            result = await system.monitor_text(args.text)
            print("\nAnalysis Results:")
            print("-----------------")
            print(f"Detection: {'Yes' if result['detection'] else 'No'}")
            if result['detection']:
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Explanation: {result.get('explanation', 'None provided')}")
                print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
            elif 'error' in result:
                print(f"Error: {result['error']}")
        else:
            print("\nHoney-Prompt Detection System")
            print("===========================")
            print("Enter text to analyze (or 'quit' to exit)")
            print("Commands:")
            print("  status  - Show system status")
            print("  metrics - Show current metrics")
            print("  quit    - Exit the system")

            while True:
                try:
                    user_input = input("\nCommand> ")
                    if not user_input.strip():
                        continue

                    command = user_input.strip().lower()

                    if command == 'quit':
                        break
                    elif command == 'status':
                        status = await system.get_system_status()
                        print("\nSystem Status:")
                        print("--------------")
                        for key, value in status.items():
                            print(f"{key}: {value}")
                    elif command == 'metrics':
                        metrics = system.metrics.get_summary()
                        print("\nSystem Metrics:")
                        print("--------------")
                        print(json.dumps(metrics, indent=2))
                    else:
                        result = await system.monitor_text(command)
                        if result.get('error'):
                            print(f"Error: {result['error']}")
                        elif result['detection']:
                            print("\n⚠️  Potential prompt injection detected!")
                            print(f"Confidence: {result['confidence']:.2f}")
                            print(f"Explanation: {result.get('explanation', 'None provided')}")
                            print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
                        else:
                            print("No prompt injection detected")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")

        await system.stop()
        print("\nSystem shutdown complete")

    asyncio.run(run_async())

if __name__ == "__main__":
    main()
