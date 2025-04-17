#!/usr/bin/env python3
"""
Multi-Expert AI System for Sneaker Bot Detection (MEAS-BD) Demo

This script demonstrates the capabilities of the MEAS-BD system
for detecting bots targeting limited-edition sneaker releases.
"""

import os
import json
import argparse
import logging
from datetime import datetime

from src.main import BotDetectionSystem
from src.data import SyntheticDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_interactive_demo(system, num_samples=5):
    """Run an interactive demo with the system"""
    print("\n" + "="*80)
    print("MEAS-BD: Multi-Expert AI System for Sneaker Bot Detection")
    print("="*80)
    print("\nThis demo will generate synthetic sessions and analyze them in real-time.")
    print("You'll be able to see how each expert model contributes to the final decision.")
    print("\nGenerating synthetic sessions...")
    
    # Create synthetic data generator with minimal configuration
    data_gen = SyntheticDataGenerator({
        "human_samples": num_samples,
        "bot_samples": num_samples,
        "noise_level": 0.1
    })
    
    # Generate sessions
    sessions, labels = data_gen.generate_dataset()
    
    # Analyze each session
    correct_count = 0
    
    print("\n" + "-"*80)
    print("Starting analysis of {} synthetic sessions...".format(len(sessions)))
    print("-"*80)
    
    for i, session in enumerate(sessions):
        # Add session ID if not present
        if "sessionId" not in session:
            session["sessionId"] = f"demo-{i+1}"
        
        session_id = session["sessionId"]
        true_label = "Bot" if labels[i] == 1 else "Human"
        
        # Print session info
        print(f"\nSession {i+1}/{len(sessions)}: ID={session_id}, True={true_label}")
        
        # Display key session attributes
        browser = session.get("browserFingerprint", {}).get("browserType", "unknown")
        user_agent = session.get("browserFingerprint", {}).get("userAgent", "")
        checkout_time = session.get("checkout", {}).get("checkoutTime", 0)
        
        print(f"Browser: {browser}, Checkout Time: {checkout_time:.2f}s")
        print(f"User Agent: {user_agent[:80]}..." if len(user_agent) > 80 else f"User Agent: {user_agent}")
        
        # Get number of events
        events = session.get("events", [])
        page_views = sum(1 for e in events if e.get("type") == "pageview")
        clicks = sum(1 for e in events if e.get("type") == "click")
        cart_adds = sum(1 for e in events if e.get("type") == "add_to_cart")
        
        print(f"Events: {len(events)} total, {page_views} page views, {clicks} clicks, {cart_adds} cart adds")
        
        # Analyze the session
        input("Press Enter to analyze this session...")
        
        result = system.analyze_session(session)
        
        predicted_label = "Bot" if result.get("is_bot", False) else "Human"
        probability = result.get("probability", 0.5)
        confidence = result.get("confidence", 0.0)
        is_correct = predicted_label == true_label
        
        if is_correct:
            correct_count += 1
        
        # Print the result
        print("\nAnalysis Result:")
        print(f"  Prediction: {predicted_label} (True label: {true_label}) {'✓' if is_correct else '✗'}")
        print(f"  Probability: {probability:.3f}, Confidence: {confidence:.3f}")
        
        # Print expert breakdown
        print("\nExpert Opinions:")
        expert_results = result.get("expert_results", {})
        
        # Format for readable output
        max_name_len = max(len(name) for name in expert_results.keys()) if expert_results else 0
        
        for expert_name, expert_result in expert_results.items():
            display_name = expert_name.replace("_expert", "").replace("_", " ").title()
            display_name = display_name.ljust(max_name_len + 5)
            
            expert_prob = expert_result.get("probability", 0.5)
            expert_conf = expert_result.get("confidence", 0.0)
            
            # Determine if this expert thinks it's a bot
            expert_vote = "Bot" if expert_prob >= 0.5 else "Human"
            vote_marker = "✓" if (expert_vote == true_label) else "✗"
            
            print(f"  - {display_name}: {expert_vote} ({expert_prob:.3f}) {vote_marker} Confidence: {expert_conf:.3f}")
        
        # Print explanations
        explanations = result.get("explanation", [])
        if explanations:
            print("\nExplanations:")
            for expl in explanations:
                print(f"  - {expl}")
        
        # Print separator
        print("\n" + "-"*80)
    
    # Print overall accuracy
    accuracy = correct_count / len(sessions)
    print(f"\nOverall accuracy: {accuracy:.2f} ({correct_count}/{len(sessions)} correct)")
    
    print("\nDemo completed!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MEAS-BD Demo')
    parser.add_argument('--config', type=str, default='config/default_config.json',
                      help='Path to configuration file')
    parser.add_argument('--samples', type=int, default=5,
                      help='Number of samples per class (human/bot) to generate')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to a pre-trained model (optional)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Initialize system
    print("Initializing MEAS-BD system...")
    system = BotDetectionSystem(args.config)
    if not system.initialize():
        logger.error("Failed to initialize system")
        return 1
    
    # Load model if specified
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        if not system.load_model(args.model_path):
            logger.error("Failed to load model")
            return 1
    else:
        # Train a quick model on synthetic data
        print("No pre-trained model specified. Training a new model on synthetic data...")
        if not system.train():
            logger.error("Training failed")
            return 1
    
    # Run interactive demo
    run_interactive_demo(system, args.samples)
    
    return 0


if __name__ == "__main__":
    exit(main())