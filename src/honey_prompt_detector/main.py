#!/usr/bin/env python3
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import argparse

from src.honey_prompt_detector.utils.validation import InputValidator
from src.honey_prompt_detector.agents.token_designer_agent import TokenDesignerAgent
from src.honey_prompt_detector.agents.context_evaluator_agent import ContextEvaluatorAgent
from src.honey_prompt_detector.core.orchestrator import DetectionOrchestrator
from src.honey_prompt_detector.monitoring.metrics import MetricsCollector
from src.honey_prompt_detector.monitoring.alerts import AlertManager
from src.honey_prompt_detector.utils.logging import setup_logger
from src.honey_prompt_detector.utils.config import Config

config = Config()
logger = setup_logger(
    name=__name__,
    log_file=config.log_file,
    level=config.log_level,
    retention_days=config.retention_days
)


from src.honey_prompt_detector.core.self_tuner import SelfTunerAgent

class HoneyPromptSystem:
    def __init__(self, env_path: Optional[Path] = None, custom_config: Optional[Dict[str, Any]] = None):
        self.config = Config(env_path)
        if custom_config:
            for key, value in custom_config.items():
                setattr(self.config, key, value)

        self.token_designer = TokenDesignerAgent(
            api_key=self.config.openai_api_key,
            model_name=self.config.model_name
        )
        self.context_evaluator = ContextEvaluatorAgent(
            api_key=self.config.openai_api_key,
            model_name=self.config.model_name
        )
        self.metrics = MetricsCollector(metrics_file=self.config.metrics_file)
        self.alert_manager = AlertManager(self.config.as_dict().get('alert_settings', {}))
        self.orchestrator = DetectionOrchestrator(
            token_designer=self.token_designer,
            context_evaluator=self.context_evaluator,
            config=self.config
        )

        # Self Tuner initialization
        self.self_tuner = SelfTunerAgent(
            detector_agent=self.orchestrator.detector,
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

    async def monitor_text(self, text: str, expected_detection: Optional[bool] = None) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call start() first.")

        start_time = datetime.now()

        try:
            sanitized_inputs = await self.orchestrator.environment_agent.sanitize_external_inputs(
                external_inputs=[text],
                honey_prompts=self.orchestrator.honey_prompts
            )

            if not sanitized_inputs:
                result = {
                    'detection': True,
                    'confidence': 1.0,
                    'explanation': "Indirect injection detected early by EnvironmentAgent",
                    'risk_level': 'high'
                }
            else:
                result = await self.orchestrator.monitor_text(sanitized_inputs[0])

            self.metrics.record_detection(result)

            if expected_detection is not None:
                self.self_tuner.update_metrics(result, expected_detection)
                new_threshold = self.self_tuner.adjust_threshold_if_needed()
                logger.info(f"Adjusted detection threshold: {new_threshold}")

            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_performance(response_time)

            if result['detection']:
                await self.alert_manager.send_alert(result)

            return result

        except Exception as e:
            logger.error(f"Error monitoring text: {str(e)}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_performance(response_time, is_error=True)
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

    # async def run_async(args=None):
    #     system = HoneyPromptSystem(args.env)
    #     if not await system.start():
    #         logger.error("System startup failed")
    #         return
    #
    #     try:
    #         # Interactive loop
    #         print("\nHoney-Prompt Detector")
    #         print("===========================")
    #         print("Enter text to analyze (or 'quit' to exit)")
    #         print("Commands:")
    #         print("  status  - Show system status")
    #         print("  metrics - Show current metrics")
    #         print("  quit    - Exit the system")
    #
    #         while True:
    #             user_input = input("\nCommand> ")
    #             if not user_input.strip():
    #                 continue  # Ignore empty input
    #
    #             command = user_input.strip().lower()
    #
    #             if command == 'quit':
    #                 break
    #             elif command == 'status':
    #                 status = await system.get_system_status()
    #                 print("\nSystem Status:")
    #                 print("--------------")
    #                 for key, value in status.items():
    #                     print(f"{key}: {value}")
    #             elif command == 'metrics':
    #                 metrics = system.metrics.get_summary()
    #                 print("\nSystem Metrics:")
    #                 print("--------------")
    #                 print(json.dumps(metrics, indent=2))
    #             else:
    #                 result = await system.monitor_text(command)
    #                 if result.get('error'):
    #                     print(f"Error: {result['error']}")
    #                 elif result['detection']:
    #                     print("\n⚠️  Potential prompt injection detected!")
    #                     print(f"Confidence: {result['confidence']:.2f}")
    #                     print(f"Explanation: {result.get('explanation', 'None provided')}")
    #                     print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
    #                 else:
    #                     print("No prompt injection detected")
    #
    #     except KeyboardInterrupt:
    #         pass
    #     except Exception as e:
    #         print(f"Error: {str(e)}")
    #
    #     # Cleanup and shutdown
    #     await system.stop()
    #     print("\nSystem shutdown complete")

    async def stop(self):
        """Gracefully shut down the system."""
        if self.is_initialized:
            logger.info("Shutting down honey-prompt detector")
            self.metrics.save_metrics()
            self.is_initialized = False


def main():
    parser = argparse.ArgumentParser(description="Honey-Prompt Detector")
    parser.add_argument("--env", type=Path, help="Path to .env file")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--run-experiments", action="store_true", help="Run comprehensive experiments")

    args = parser.parse_args()

    async def run_async():
        if args.run_experiments:
            from test.basic_usage import ExperimentRunner
            experiment = ExperimentRunner()
            await experiment.run_experiments()
            return

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
            print("\nHoney-Prompt Detector")
            print("===========================")
            print("Enter text to analyze (or 'quit' to exit)")
            print("Commands:")
            print("  status  - Show system status")
            print("  metrics - Show current metrics")
            print("  quit    - Exit the system")

            while True:
                try:
                    user_input = input("\nCommand> ").strip()
                    if not user_input:
                        continue

                    command = user_input.lower()

                    if command == 'quit':
                        break
                    elif command == 'status':
                        status = await system.get_system_status()
                        print("\nSystem Status:")
                        for key, value in status.items():
                            print(f"{key}: {value}")
                    elif command == 'metrics':
                        metrics = system.metrics.get_summary()
                        print("\nSystem Metrics:")
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
