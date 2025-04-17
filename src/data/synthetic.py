"""
Synthetic Data Generation Module for MEAS-BD (Multi-Expert AI System for Bot Detection)

This module provides functionality to generate synthetic data for training and testing
bot detection models. It addresses the challenge of limited labeled data through:
1. Adversarial synthetic data generation
2. Semi-supervised learning techniques
3. Active learning procedures

The module can generate synthetic session data that simulates both legitimate human
traffic and various types of bot traffic targeting sneaker e-commerce platforms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import json
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Main class for generating synthetic data for bot detection training and evaluation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the synthetic data generator with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.session_ids = set()
        
        # Set default parameters if not provided in config
        self.human_ratio = self.config.get('human_ratio', 0.6)
        self.bot_types = self.config.get('bot_types', {
            'basic': 0.2,         # Basic bots with simple patterns
            'advanced': 0.5,      # Advanced bots with more human-like behavior
            'sophisticated': 0.3  # Sophisticated bots with highly human-like behavior
        })
        
        # Load device fingerprint templates if available
        self.device_templates = self._load_device_templates()
        
        # Load navigation patterns if available
        self.navigation_patterns = self._load_navigation_patterns()
        
        logger.info(f"Initialized SyntheticDataGenerator with human_ratio={self.human_ratio}")

    def _load_device_templates(self) -> Dict:
        """
        Load device fingerprint templates from config or default templates.
        
        Returns:
            Dictionary of device templates
        """
        if self.config.get('device_templates_path'):
            path = Path(self.config['device_templates_path'])
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        
        # Default device templates
        return {
            'desktop': {
                'user_agents': [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0'
                ],
                'screen_resolutions': ['1920x1080', '2560x1440', '1366x768', '1440x900'],
                'color_depths': [24, 32],
                'platform': ['Win32', 'MacIntel', 'Linux x86_64'],
                'plugins_length_range': (5, 15)
            },
            'mobile': {
                'user_agents': [
                    'Mozilla/5.0 (iPhone; CPU iPhone OS 15_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
                    'Mozilla/5.0 (Linux; Android 12; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.101 Mobile Safari/537.36',
                    'Mozilla/5.0 (iPad; CPU OS 15_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
                ],
                'screen_resolutions': ['390x844', '375x812', '414x896', '360x800'],
                'color_depths': [24, 32],
                'platform': ['iPhone', 'iPad', 'Android'],
                'plugins_length_range': (0, 3)
            },
            'bot': {
                'user_agents': [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15'
                ],
                'screen_resolutions': ['1920x1080', '1366x768'],
                'color_depths': [24],
                'platform': ['Win32', 'MacIntel'],
                'plugins_length_range': (0, 10)
            }
        }

    def _load_navigation_patterns(self) -> Dict:
        """
        Load navigation patterns from config or default patterns.
        
        Returns:
            Dictionary of navigation patterns
        """
        if self.config.get('navigation_patterns_path'):
            path = Path(self.config['navigation_patterns_path'])
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        
        # Default navigation patterns
        return {
            'human': {
                'product_page_time_range': (10, 180),  # seconds
                'checkout_time_range': (30, 300),      # seconds
                'pages_per_session_range': (3, 15),
                'probability_add_to_cart': 0.4,
                'probability_checkout': 0.3,
                'probability_navigate_away': 0.3,
                'page_transition_patterns': [
                    ['home', 'category', 'product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'search', 'product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'product', 'product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'category', 'category', 'product', 'cart', 'checkout', 'error', 'checkout', 'confirmation'],
                    ['category', 'product', 'home', 'search', 'product', 'cart', 'checkout', 'confirmation']
                ]
            },
            'basic_bot': {
                'product_page_time_range': (0.5, 3),   # seconds
                'checkout_time_range': (5, 20),        # seconds
                'pages_per_session_range': (1, 3),
                'probability_add_to_cart': 0.9,
                'probability_checkout': 0.95,
                'probability_navigate_away': 0.05,
                'page_transition_patterns': [
                    ['product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'product', 'cart', 'checkout', 'confirmation'],
                    ['product', 'cart', 'checkout', 'error', 'checkout', 'confirmation']
                ]
            },
            'advanced_bot': {
                'product_page_time_range': (3, 10),    # seconds
                'checkout_time_range': (15, 45),       # seconds
                'pages_per_session_range': (2, 5),
                'probability_add_to_cart': 0.8,
                'probability_checkout': 0.9,
                'probability_navigate_away': 0.1,
                'page_transition_patterns': [
                    ['home', 'category', 'product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'search', 'product', 'cart', 'checkout', 'confirmation'],
                    ['product', 'product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'product', 'cart', 'checkout', 'confirmation']
                ]
            },
            'sophisticated_bot': {
                'product_page_time_range': (5, 60),    # seconds
                'checkout_time_range': (25, 120),      # seconds
                'pages_per_session_range': (3, 8),
                'probability_add_to_cart': 0.6,
                'probability_checkout': 0.7,
                'probability_navigate_away': 0.2,
                'page_transition_patterns': [
                    ['home', 'category', 'product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'search', 'product', 'product', 'cart', 'checkout', 'confirmation'],
                    ['category', 'product', 'cart', 'checkout', 'confirmation'],
                    ['home', 'category', 'category', 'product', 'cart', 'checkout', 'error', 'checkout', 'confirmation'],
                    ['home', 'search', 'product', 'home', 'product', 'cart', 'checkout', 'confirmation']
                ]
            }
        }

    def generate_dataset(self, n_sessions: int = 1000, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a synthetic dataset with a mixture of human and bot sessions.
        
        Args:
            n_sessions: Number of sessions to generate
            output_path: Optional path to save the generated dataset
            
        Returns:
            DataFrame containing synthetic session data
        """
        logger.info(f"Generating synthetic dataset with {n_sessions} sessions")
        
        # Determine number of each type of session
        n_human = int(n_sessions * self.human_ratio)
        n_bot = n_sessions - n_human
        
        bot_type_counts = {
            bot_type: int(n_bot * ratio)
            for bot_type, ratio in self.bot_types.items()
        }
        
        # Adjust for rounding errors
        adjustment = n_bot - sum(bot_type_counts.values())
        if adjustment != 0:
            # Add or subtract from the largest bot type
            largest_bot_type = max(bot_type_counts, key=bot_type_counts.get)
            bot_type_counts[largest_bot_type] += adjustment
        
        logger.info(f"Distribution - Human: {n_human}, Bots: {bot_type_counts}")
        
        # Generate sessions
        human_sessions = [self.generate_human_session() for _ in range(n_human)]
        
        bot_sessions = []
        for bot_type, count in bot_type_counts.items():
            for _ in range(count):
                bot_sessions.append(self.generate_bot_session(bot_type=bot_type))
        
        # Combine and shuffle
        all_sessions = human_sessions + bot_sessions
        random.shuffle(all_sessions)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_sessions)
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved dataset to {output_path}")
        
        return df

    def generate_human_session(self) -> Dict:
        """
        Generate a synthetic session representing human behavior.
        
        Returns:
            Dictionary containing session data
        """
        session_id = self._generate_unique_session_id()
        
        # Choose device type
        device_type = random.choice(['desktop', 'mobile']) if random.random() < 0.3 else 'desktop'
        
        # Generate technical fingerprint
        technical_fingerprint = self._generate_technical_fingerprint(device_type)
        
        # Generate navigation behavior
        navigation_behavior = self._generate_navigation_behavior('human')
        
        # Generate input behavior (mouse movements, keystrokes, etc.)
        input_behavior = self._generate_input_behavior('human')
        
        # Generate temporal patterns
        temporal_patterns = self._generate_temporal_patterns('human')
        
        # Generate purchase patterns if checkout occurred
        purchase_patterns = self._generate_purchase_patterns('human') if 'checkout' in navigation_behavior['page_sequence'] else {}
        
        return {
            'session_id': session_id,
            'is_bot': False,
            'bot_type': None,
            'device_type': device_type,
            'technical_fingerprint': technical_fingerprint,
            'navigation_behavior': navigation_behavior,
            'input_behavior': input_behavior,
            'temporal_patterns': temporal_patterns,
            'purchase_patterns': purchase_patterns,
            'label': 'human'
        }

    def generate_bot_session(self, bot_type: str = 'basic') -> Dict:
        """
        Generate a synthetic session representing bot behavior.
        
        Args:
            bot_type: Type of bot to simulate ('basic', 'advanced', or 'sophisticated')
            
        Returns:
            Dictionary containing session data
        """
        session_id = self._generate_unique_session_id()
        
        # For basic bots, usually use desktop
        device_type = 'desktop'
        if bot_type == 'sophisticated' and random.random() < 0.4:
            device_type = random.choice(['desktop', 'mobile'])
        
        # Generate technical fingerprint
        technical_fingerprint = self._generate_technical_fingerprint('bot')
        
        # Add inconsistencies in sophisticated bots to simulate human fingerprints
        if bot_type == 'sophisticated':
            technical_fingerprint = self._add_human_fingerprint_characteristics(technical_fingerprint)
        
        # Generate navigation behavior
        navigation_behavior = self._generate_navigation_behavior(f'{bot_type}_bot')
        
        # Generate input behavior (mouse movements, keystrokes, etc.)
        input_behavior = self._generate_input_behavior(bot_type)
        
        # Generate temporal patterns
        temporal_patterns = self._generate_temporal_patterns(bot_type)
        
        # Generate purchase patterns if checkout occurred
        purchase_patterns = self._generate_purchase_patterns(bot_type) if 'checkout' in navigation_behavior['page_sequence'] else {}
        
        return {
            'session_id': session_id,
            'is_bot': True,
            'bot_type': bot_type,
            'device_type': device_type,
            'technical_fingerprint': technical_fingerprint,
            'navigation_behavior': navigation_behavior,
            'input_behavior': input_behavior,
            'temporal_patterns': temporal_patterns,
            'purchase_patterns': purchase_patterns,
            'label': f'{bot_type}_bot'
        }

    def _generate_unique_session_id(self) -> str:
        """
        Generate a unique session ID.
        
        Returns:
            Unique session ID string
        """
        session_id = str(uuid.uuid4())
        while session_id in self.session_ids:
            session_id = str(uuid.uuid4())
        self.session_ids.add(session_id)
        return session_id

    def _generate_technical_fingerprint(self, device_type: str) -> Dict:
        """
        Generate a technical fingerprint for the session.
        
        Args:
            device_type: Type of device ('desktop', 'mobile', or 'bot')
            
        Returns:
            Dictionary containing technical fingerprint data
        """
        templates = self.device_templates.get(device_type, self.device_templates['desktop'])
        
        user_agent = random.choice(templates['user_agents'])
        screen_resolution = random.choice(templates['screen_resolutions'])
        color_depth = random.choice(templates['color_depths'])
        platform = random.choice(templates['platform'])
        plugins_length = random.randint(*templates['plugins_length_range'])
        
        # Generate consistent fingerprint based on these parameters
        canvas_hash = self._generate_mock_canvas_hash(user_agent, screen_resolution)
        webgl_hash = self._generate_mock_webgl_hash(user_agent, platform)
        
        # For bots, sometimes we want inconsistencies
        if device_type == 'bot':
            # 30% chance of inconsistent fingerprints for basic bots
            if random.random() < 0.3:
                canvas_hash = self._generate_mock_canvas_hash(random.choice(templates['user_agents']), 
                                                            random.choice(templates['screen_resolutions']))
        
        return {
            'user_agent': user_agent,
            'screen_resolution': screen_resolution,
            'color_depth': color_depth,
            'platform': platform,
            'plugins_length': plugins_length,
            'cookies_enabled': True if device_type != 'bot' or random.random() < 0.8 else False,
            'canvas_fingerprint': canvas_hash,
            'webgl_fingerprint': webgl_hash,
            'language': 'en-US' if random.random() < 0.8 else random.choice(['en-GB', 'en-CA', 'es-US', 'fr-FR']),
            'timezone_offset': -300 if random.random() < 0.6 else random.choice([-330, -600, 0, 60, 120]),  # -300 is US Eastern Time (NY)
            'do_not_track': 'unspecified' if random.random() < 0.7 else random.choice(['true', 'false']),
            'hardware_concurrency': 4 if random.random() < 0.5 else random.choice([2, 8, 16]),
            'session_storage': True if device_type != 'bot' or random.random() < 0.9 else False,
            'local_storage': True if device_type != 'bot' or random.random() < 0.9 else False,
            'indexed_db': True if device_type != 'bot' or random.random() < 0.7 else False,
            'cpu_class': 'unknown' if device_type == 'mobile' else random.choice(['Intel', 'AMD', 'unknown']),
            'touch_points': 0 if device_type == 'desktop' else random.randint(1, 5),
        }
    
    def _add_human_fingerprint_characteristics(self, fingerprint: Dict) -> Dict:
        """
        Add human-like characteristics to a bot fingerprint to make it more sophisticated.
        
        Args:
            fingerprint: Original bot fingerprint
            
        Returns:
            Modified fingerprint with human-like characteristics
        """
        # Copy the fingerprint to avoid modifying the original
        human_like_fingerprint = fingerprint.copy()
        
        # Select a device template that matches the sophisticated behavior we want to mimic
        device_template = random.choice(['desktop', 'mobile'])
        templates = self.device_templates[device_template]
        
        # Adjust a few parameters to be more human-like
        if random.random() < 0.7:
            human_like_fingerprint['user_agent'] = random.choice(templates['user_agents'])
        
        if random.random() < 0.7:
            human_like_fingerprint['screen_resolution'] = random.choice(templates['screen_resolutions'])
        
        # Ensure cookies, storages are enabled as most humans have these
        human_like_fingerprint['cookies_enabled'] = True
        human_like_fingerprint['session_storage'] = True
        human_like_fingerprint['local_storage'] = True
        
        # Add realistic language and timezone combinations
        language_timezone_pairs = [
            ('en-US', -300),  # Eastern Time
            ('en-US', -360),  # Central Time
            ('en-US', -420),  # Mountain Time
            ('en-US', -480),  # Pacific Time
            ('en-GB', 0),     # GMT
            ('fr-FR', 60),    # Central European Time
            ('de-DE', 60),    # Central European Time
            ('ja-JP', 540),   # Japan Standard Time
        ]
        
        if random.random() < 0.8:
            lang, tz = random.choice(language_timezone_pairs)
            human_like_fingerprint['language'] = lang
            human_like_fingerprint['timezone_offset'] = tz
        
        # Match touch_points with device type
        if device_template == 'mobile':
            human_like_fingerprint['touch_points'] = random.randint(1, 5)
        else:
            human_like_fingerprint['touch_points'] = 0
            
        return human_like_fingerprint

    def _generate_mock_canvas_hash(self, user_agent: str, screen_resolution: str) -> str:
        """
        Generate a mock canvas fingerprint hash.
        
        Args:
            user_agent: User agent string
            screen_resolution: Screen resolution string
            
        Returns:
            Mock canvas fingerprint hash
        """
        # This is a simplified mock - in reality canvas fingerprinting is more complex
        combined = user_agent + screen_resolution
        return f"canvas_fp_{hash(combined) % 10000000:07d}"

    def _generate_mock_webgl_hash(self, user_agent: str, platform: str) -> str:
        """
        Generate a mock WebGL fingerprint hash.
        
        Args:
            user_agent: User agent string
            platform: Platform string
            
        Returns:
            Mock WebGL fingerprint hash
        """
        # This is a simplified mock - in reality WebGL fingerprinting is more complex
        combined = user_agent + platform
        return f"webgl_fp_{hash(combined) % 10000000:07d}"

    def _generate_navigation_behavior(self, session_type: str) -> Dict:
        """
        Generate navigation behavior for the session.
        
        Args:
            session_type: Type of session ('human', 'basic_bot', 'advanced_bot', or 'sophisticated_bot')
            
        Returns:
            Dictionary containing navigation behavior data
        """
        # Get the appropriate navigation pattern template
        patterns = self.navigation_patterns.get(session_type, self.navigation_patterns['human'])
        
        # Select a page transition pattern
        page_sequence = random.choice(patterns['page_transition_patterns'])
        
        # For humans and sophisticated bots, add some randomness to the page sequence
        if session_type in ['human', 'sophisticated_bot']:
            # Possibly repeat some pages (like looking at multiple products)
            if random.random() < 0.4:
                product_index = page_sequence.index('product') if 'product' in page_sequence else -1
                if product_index != -1:
                    repeats = random.randint(1, 3)
                    for _ in range(repeats):
                        page_sequence.insert(product_index + 1, 'product')
            
            # Possibly skip some pages
            if random.random() < 0.3 and len(page_sequence) > 3:
                skip_index = random.randint(0, len(page_sequence) - 3)
                page_sequence.pop(skip_index)
        
        # For bots, possibly add direct navigation to target product
        if session_type != 'human' and random.random() < 0.7:
            if 'product' not in page_sequence:
                # Add product page if not present
                page_sequence.insert(0, 'product')
            elif page_sequence.index('product') > 1:
                # Move product page earlier in sequence
                product_index = page_sequence.index('product')
                page_sequence.pop(product_index)
                page_sequence.insert(0, 'product')
        
        # Generate time spent on each page
        page_times = {}
        for page in page_sequence:
            if page == 'product':
                page_times[page] = random.uniform(*patterns['product_page_time_range'])
            elif page == 'checkout':
                page_times[page] = random.uniform(*patterns['checkout_time_range'])
            else:
                # Other pages get random times between 2-30 seconds for humans, shorter for bots
                if session_type == 'human':
                    page_times[page] = random.uniform(2, 30)
                elif session_type == 'sophisticated_bot':
                    page_times[page] = random.uniform(1, 20)
                elif session_type == 'advanced_bot':
                    page_times[page] = random.uniform(0.5, 10)
                else:  # basic_bot
                    page_times[page] = random.uniform(0.1, 3)
        
        # Generate referrer information
        referrer = None
        if random.random() < 0.7:
            referrer_options = {
                'search_engine': 0.4,
                'social_media': 0.3,
                'direct': 0.2,
                'other_site': 0.1
            }
            referrer_type = self._weighted_choice(referrer_options)
            
            if referrer_type == 'search_engine':
                search_engines = {
                    'google.com': 0.7,
                    'bing.com': 0.2,
                    'yahoo.com': 0.1
                }
                referrer = f"https://www.{self._weighted_choice(search_engines)}/search?q=limited+edition+sneakers"
            elif referrer_type == 'social_media':
                social_media = {
                    'instagram.com': 0.5,
                    'twitter.com': 0.3,
                    'facebook.com': 0.2
                }
                referrer = f"https://www.{self._weighted_choice(social_media)}/sneaker_release_post"
            elif referrer_type == 'other_site':
                other_sites = {
                    'sneakernews.com': 0.4,
                    'hypebeast.com': 0.3,
                    'complex.com': 0.2,
                    'highsnobiety.com': 0.1
                }
                referrer = f"https://www.{self._weighted_choice(other_sites)}/upcoming-releases"
        
        return {
            'page_sequence': page_sequence,
            'page_times': page_times,
            'total_pages_visited': len(page_sequence),
            'session_duration': sum(page_times.values()),
            'referrer': referrer,
            'exit_page': page_sequence[-1],
            'add_to_cart': 'cart' in page_sequence,
            'reached_checkout': 'checkout' in page_sequence,
            'completed_purchase': 'confirmation' in page_sequence,
            'encountered_error': 'error' in page_sequence
        }

    def _generate_input_behavior(self, session_type: str) -> Dict:
        """
        Generate input behavior (mouse movements, clicks, keystrokes) for the session.
        
        Args:
            session_type: Type of session ('human', 'basic', 'advanced', or 'sophisticated')
            
        Returns:
            Dictionary containing input behavior data
        """
        if session_type == 'human':
            # Human behavior has natural variations
            mouse_movement_count = random.randint(50, 300)
            mouse_movement_consistency = random.uniform(0.7, 1.0)  # High consistency but with variations
            movement_speed_variation = random.uniform(0.2, 0.5)    # Natural speed variations
            click_count = random.randint(5, 20)
            keyboard_input_speed = random.uniform(100, 300)        # ms between keystrokes
            keyboard_input_consistency = random.uniform(0.7, 0.95)  # Some timing variations
            form_input_natural = True
            keyboard_error_rate = random.uniform(0.01, 0.1)        # Natural typing errors
            
        elif session_type == 'basic':
            # Basic bots have minimal or unnaturally regular movements
            mouse_movement_count = random.randint(0, 10)
            mouse_movement_consistency = random.uniform(0.95, 1.0)  # Very consistent patterns
            movement_speed_variation = random.uniform(0, 0.1)       # Almost no variation
            click_count = random.randint(1, 5)
            keyboard_input_speed = random.uniform(10, 50)           # Very fast typing
            keyboard_input_consistency = random.uniform(0.95, 1.0)   # Very consistent timing
            form_input_natural = False
            keyboard_error_rate = 0.0                               # No typing errors
            
        elif session_type == 'advanced':
            # Advanced bots attempt to simulate some human-like behavior
            mouse_movement_count = random.randint(20, 100)
            mouse_movement_consistency = random.uniform(0.8, 0.95)  # More consistent than humans
            movement_speed_variation = random.uniform(0.1, 0.3)     # Some variation
            click_count = random.randint(3, 10)
            keyboard_input_speed = random.uniform(50, 150)          # Faster than average human
            keyboard_input_consistency = random.uniform(0.8, 0.95)   # Some timing variations
            form_input_natural = random.random() < 0.3               # Usually not natural
            keyboard_error_rate = random.uniform(0, 0.03)           # Few typing errors
            
        else:  # sophisticated bot
            # Sophisticated bots closely mimic human behavior
            mouse_movement_count = random.randint(40, 250)
            mouse_movement_consistency = random.uniform(0.75, 0.9)  # Close to human consistency
            movement_speed_variation = random.uniform(0.15, 0.4)    # Natural-seeming variation
            click_count = random.randint(4, 15)
            keyboard_input_speed = random.uniform(80, 250)          # Close to human speed
            keyboard_input_consistency = random.uniform(0.75, 0.9)   # Human-like variations
            form_input_natural = random.random() < 0.7               # Usually natural
            keyboard_error_rate = random.uniform(0.005, 0.05)       # Some typing errors
        
        # Generate mouse movement entropy (measure of randomness in movements)
        mouse_entropy = self._calculate_mock_mouse_entropy(
            mouse_movement_count, 
            mouse_movement_consistency,
            movement_speed_variation
        )
        
        # For bots, sometimes we want to directly simulate human behavior
        if session_type != 'human' and session_type == 'sophisticated' and random.random() < 0.5:
            # Copy some human-like parameters to make detection harder
            mouse_movement_count = random.randint(50, 300)
            mouse_movement_consistency = random.uniform(0.7, 0.9)
            movement_speed_variation = random.uniform(0.2, 0.5)
            mouse_entropy = self._calculate_mock_mouse_entropy(
                mouse_movement_count, 
                mouse_movement_consistency,
                movement_speed_variation
            )
        
        return {
            'mouse_movement_count': mouse_movement_count,
            'mouse_movement_consistency': mouse_movement_consistency,
            'mouse_movement_entropy': mouse_entropy,
            'movement_speed_variation': movement_speed_variation,
            'click_count': click_count,
            'keyboard_input_speed': keyboard_input_speed,  # ms between keystrokes
            'keyboard_input_consistency': keyboard_input_consistency,
            'form_input_natural': form_input_natural,  # Whether form fields are filled in a natural order
            'keyboard_error_rate': keyboard_error_rate,  # Rate of typos/corrections
            'rage_clicks': random.randint(0, 3) if session_type == 'human' and random.random() < 0.2 else 0,
            'cursor_trail': self._generate_mock_cursor_trail(session_type),
        }

    def _calculate_mock_mouse_entropy(self, movement_count: int, consistency: float, speed_variation: float) -> float:
        """
        Calculate a mock entropy value for mouse movements.
        
        Args:
            movement_count: Number of mouse movements
            consistency: Consistency of mouse movements (0.0-1.0)
            speed_variation: Variation in movement speed (0.0-1.0)
            
        Returns:
            Mock entropy value
        """
        # This is a simplified model - real entropy calculations would analyze actual movement patterns
        base_entropy = random.uniform(0.3, 0.7)  # Base entropy value
        
        # More movements generally mean more entropy
        count_factor = min(1.0, movement_count / 200.0) * 0.3
        
        # Less consistency means more entropy
        consistency_factor = (1.0 - consistency) * 0.4
        
        # More speed variation means more entropy
        variation_factor = speed_variation * 0.3
        
        # Combine factors with some randomness
        entropy = base_entropy + count_factor + consistency_factor + variation_factor
        entropy *= random.uniform(0.9, 1.1)  # Add some randomness
        
        return min(1.0, max(0.0, entropy))  # Ensure value is between 0 and 1

    def _generate_mock_cursor_trail(self, session_type: str) -> List[Dict]:
        """
        Generate a simplified mock cursor trail.
        
        Args:
            session_type: Type of session ('human', 'basic', 'advanced', or 'sophisticated')
            
        Returns:
            List of cursor positions with timestamps
        """
        # For simplicity, we'll just return a representative sample size
        # In a real system, this would be actual cursor coordinates with timestamps
        
        if session_type == 'human':
            # Humans have variable path lengths and natural movements
            trail_length = random.randint(20, 100)
            smoothness = random.uniform(0.6, 0.9)
        elif session_type == 'basic':
            # Basic bots have very little movement or unrealistically straight lines
            trail_length = random.randint(0, 10)
            smoothness = random.uniform(0.9, 1.0)
        elif session_type == 'advanced':
            # Advanced bots attempt to mimic humans but may be too regular
            trail_length = random.randint(10, 50)
            smoothness = random.uniform(0.8, 0.95)
        else:  # sophisticated
            # Sophisticated bots closely mimic human movements
            trail_length = random.randint(15, 80)
            smoothness = random.uniform(0.7, 0.9)
        
        # Return a simplified representation of the trail
        return [{'length': trail_length, 'smoothness': smoothness}]

    def _generate_temporal_patterns(self, session_type: str) -> Dict:
        """
        Generate temporal patterns for the session.
        
        Args:
            session_type: Type of session ('human', 'basic', 'advanced', or 'sophisticated')
            
        Returns:
            Dictionary containing temporal pattern data
        """
        # Current time as base
        now = datetime.now()
        
        if session_type == 'human':
            # Humans shop at various times, with some patterns
            hour_of_day = random.choice([
                # Morning
                random.randint(7, 11),
                # Lunch break
                random.randint(12, 13),
                # After work
                random.randint(17, 22),
                # Random time
                random.randint(0, 23)
            ])
            
            # Humans are more likely to shop on certain days
            day_of_week = random.choices(
                range(7),  # 0 = Monday, 6 = Sunday
                weights=[0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15],  # Higher weights for Thu-Sun
                k=1
            )[0]
            
            # Response times vary naturally
            response_times = [random.uniform(0.5, 10.0) for _ in range(random.randint(3, 8))]
            session_consistency = random.uniform(0.7, 0.95)
            
        elif session_type == 'basic':
            # Basic bots might run at unusual hours
            hour_of_day = random.randint(0, 23)
            
            # Basic bots don't care about day of week
            day_of_week = random.randint(0, 6)
            
            # Very fast or very consistent response times
            if random.random() < 0.5:
                # Very fast
                response_times = [random.uniform(0.05, 0.5) for _ in range(random.randint(2, 4))]
            else:
                # Suspiciously consistent
                base_time = random.uniform(0.1, 1.0)
                response_times = [base_time + random.uniform(-0.05, 0.05) for _ in range(random.randint(2, 4))]
            
            session_consistency = random.uniform(0.95, 1.0)
            
        elif session_type == 'advanced':
            # Advanced bots try to mimic human timing but may not be perfect
            hour_of_day = random.randint(8, 22)  # Business hours-ish
            day_of_week = random.randint(0, 6)
            
            # More human-like response times but still often too consistent
            response_times = [random.uniform(0.3, 3.0) for _ in range(random.randint(2, 6))]
            session_consistency = random.uniform(0.8, 0.95)
            
        else:  # sophisticated
            # Sophisticated bots closely mimic human temporal patterns
            hour_of_day = random.choice([
                random.randint(8, 11),
                random.randint(12, 13),
                random.randint(17, 22)
            ])
            
            day_of_week = random.choices(
                range(7),
                weights=[0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15],
                k=1
            )[0]
            
            # Human-like response times with natural variations
            response_times = [random.uniform(0.5, 5.0) for _ in range(random.randint(3, 7))]
            session_consistency = random.uniform(0.75, 0.9)
        
        # Create timestamp
        timestamp = now.replace(
            hour=hour_of_day,
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
            microsecond=random.randint(0, 999999)
        )
        
        # Adjust day (simple approach, ignores month boundaries for simplicity)
        current_day = now.weekday()  # 0 = Monday, 6 = Sunday
        days_diff = day_of_week - current_day
        timestamp = timestamp + timedelta(days=days_diff)
        
        return {
            'timestamp': timestamp.isoformat(),
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'response_times': response_times,
            'average_response_time': sum(response_times) / len(response_times),
            'response_time_std': self._calculate_std(response_times),
            'session_consistency': session_consistency,
            'time_to_checkout': random.uniform(30, 300) if session_type == 'human' else random.uniform(5, 60)
        }

    def _calculate_std(self, values: List[float]) -> float:
        """
        Calculate standard deviation of a list of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Standard deviation
        """
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        variance = squared_diff_sum / len(values)
        return variance ** 0.5

    def _generate_purchase_patterns(self, session_type: str) -> Dict:
        """
        Generate purchase patterns for the session.
        
        Args:
            session_type: Type of session ('human', 'basic', 'advanced', or 'sophisticated')
            
        Returns:
            Dictionary containing purchase pattern data
        """
        # Common sneaker brands
        brands = ['Nike', 'Adidas', 'Jordan', 'Yeezy', 'New Balance', 'Asics']
        
        # Popular models that might be targeted by bots
        popular_models = {
            'Nike': ['Dunk Low', 'Air Force 1', 'Air Jordan 1', 'SB Dunk'],
            'Adidas': ['Yeezy Boost 350', 'Ultra Boost', 'Forum Low'],
            'Jordan': ['Jordan 1 High OG', 'Jordan 4 Retro', 'Jordan 11'],
            'Yeezy': ['Yeezy Boost 350', 'Yeezy Foam Runner', 'Yeezy Slide'],
            'New Balance': ['990v3', '550', '2002R'],
            'Asics': ['Gel-Lyte III', 'Gel-Kayano 14']
        }
        
        # Sizes vary by region, but US sizes are common
        sizes = ['7', '7.5', '8', '8.5', '9', '9.5', '10', '10.5', '11', '11.5', '12']
        
        if session_type == 'human':
            # Humans typically buy for themselves or as gifts
            brand = random.choice(brands)
            model = random.choice(popular_models[brand])
            size = random.choice(sizes)
            quantity = 1 if random.random() < 0.95 else 2  # Rarely buy more than one
            shipping_method = random.choice(['standard', 'expedited', 'premium'])
            payment_method = random.choice(['credit_card', 'paypal', 'apple_pay', 'google_pay', 'shop_pay'])
            
            # Humans sometimes check inventory, compare products, etc.
            browsing_depth = random.randint(2, 10)
            cart_modifications = random.randint(0, 3)
            
        elif session_type == 'basic':
            # Basic bots target high-resale items and popular sizes
            brand = random.choice(['Nike', 'Jordan', 'Yeezy'])  # Focus on hype brands
            model = random.choice(popular_models[brand][:2])    # Focus on most hyped models
            size = random.choice(['9', '9.5', '10', '10.5'])    # Focus on most common sizes
            quantity = 1  # Usually just try to get one
            shipping_method = 'standard'  # Don't care about shipping speed
            payment_method = random.choice(['credit_card', 'paypal'])
            
            # Basic bots don't browse much
            browsing_depth = random.randint(0, 2)
            cart_modifications = 0
            
        elif session_type == 'advanced':
            # Advanced bots show some variation but still focus on resale value
            brand = random.choice(['Nike', 'Jordan', 'Yeezy', 'Adidas'])
            model = random.choice(popular_models[brand])
            size = random.choice(sizes)
            quantity = 1
            shipping_method = random.choice(['standard', 'expedited'])
            payment_method = random.choice(['credit_card', 'paypal', 'apple_pay'])
            
            # Some limited browsing to appear more human
            browsing_depth = random.randint(1, 4)
            cart_modifications = random.randint(0, 1)
            
        else:  # sophisticated
            # Sophisticated bots mimic human behavior closely
            brand = random.choice(brands)
            model = random.choice(popular_models[brand])
            size = random.choice(sizes)
            quantity = 1 if random.random() < 0.9 else 2
            shipping_method = random.choice(['standard', 'expedited', 'premium'])
            payment_method = random.choice(['credit_card', 'paypal', 'apple_pay', 'google_pay'])
            
            # Mimic human browsing behavior
            browsing_depth = random.randint(2, 8)
            cart_modifications = random.randint(0, 2)
        
        return {
            'brand': brand,
            'model': model,
            'size': size,
            'quantity': quantity,
            'shipping_method': shipping_method,
            'payment_method': payment_method,
            'browsing_depth': browsing_depth,
            'cart_modifications': cart_modifications,
            'saved_address': random.random() < 0.7,  # 70% chance of using saved address
            'saved_payment': random.random() < 0.6,  # 60% chance of using saved payment
            'promo_code_attempted': random.random() < 0.3,  # 30% chance of trying a promo code
        }

    def _weighted_choice(self, choices: Dict[str, float]) -> str:
        """
        Make a weighted random choice from a dictionary of options and weights.
        
        Args:
            choices: Dictionary mapping options to weights
            
        Returns:
            Randomly selected option based on weights
        """
        options = list(choices.keys())
        weights = list(choices.values())
        return random.choices(options, weights=weights, k=1)[0]

    # Semi-supervised learning methods
    def generate_unlabeled_data(self, n_sessions: int = 5000) -> pd.DataFrame:
        """
        Generate unlabeled data for semi-supervised learning.
        
        Args:
            n_sessions: Number of sessions to generate
            
        Returns:
            DataFrame containing unlabeled session data
        """
        logger.info(f"Generating {n_sessions} unlabeled sessions for semi-supervised learning")
        
        # For unlabeled data, we'll generate a mix but not include the labels
        human_ratio = random.uniform(0.4, 0.8)  # More variation in distribution
        n_human = int(n_sessions * human_ratio)
        n_bot = n_sessions - n_human
        
        # Determine bot distribution with more variation
        bot_types = {
            'basic': random.uniform(0.1, 0.4),
            'advanced': random.uniform(0.3, 0.6),
            'sophisticated': random.uniform(0.2, 0.5)
        }
        
        # Normalize bot_types to sum to 1
        bot_type_sum = sum(bot_types.values())
        bot_types = {k: v / bot_type_sum for k, v in bot_types.items()}
        
        bot_type_counts = {
            bot_type: int(n_bot * ratio)
            for bot_type, ratio in bot_types.items()
        }
        
        # Adjust for rounding errors
        adjustment = n_bot - sum(bot_type_counts.values())
        if adjustment != 0:
            largest_bot_type = max(bot_type_counts, key=bot_type_counts.get)
            bot_type_counts[largest_bot_type] += adjustment
        
        # Generate sessions
        human_sessions = [self.generate_human_session() for _ in range(n_human)]
        
        bot_sessions = []
        for bot_type, count in bot_type_counts.items():
            for _ in range(count):
                bot_sessions.append(self.generate_bot_session(bot_type=bot_type))
        
        # Combine and shuffle
        all_sessions = human_sessions + bot_sessions
        random.shuffle(all_sessions)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_sessions)
        
        # Remove labels for semi-supervised learning
        df = df.drop(columns=['label'])
        
        return df

    def generate_adversarial_examples(self, model_predictions: pd.DataFrame, n_examples: int = 100) -> pd.DataFrame:
        """
        Generate adversarial examples based on model predictions.
        
        Args:
            model_predictions: DataFrame containing model predictions on existing data
            n_examples: Number of adversarial examples to generate
            
        Returns:
            DataFrame containing adversarial examples
        """
        logger.info(f"Generating {n_examples} adversarial examples")
        
        # Find misclassified examples to learn from
        misclassified = model_predictions[model_predictions['true_label'] != model_predictions['predicted_label']]
        
        if len(misclassified) == 0:
            logger.warning("No misclassified examples found, generating random adversarial examples")
            
            # Generate sophisticated bots that mimic humans closely
            adversarial_examples = [
                self.generate_adversarial_bot_session() for _ in range(n_examples)
            ]
            
        else:
            # Learn from misclassifications
            logger.info(f"Found {len(misclassified)} misclassified examples to learn from")
            
            # Generate new examples that amplify the features that led to misclassification
            adversarial_examples = []
            
            # Human sessions misclassified as bots
            human_as_bot = misclassified[
                (misclassified['true_label'] == 'human') & 
                (misclassified['predicted_label'].str.contains('bot'))
            ]
            
            # Bot sessions misclassified as humans
            bot_as_human = misclassified[
                (misclassified['true_label'].str.contains('bot')) & 
                (misclassified['predicted_label'] == 'human')
            ]
            
            # Determine how many of each type to generate
            n_human_as_bot = min(len(human_as_bot), n_examples // 2)
            n_bot_as_human = min(len(bot_as_human), n_examples - n_human_as_bot)
            
            # Fill remaining slots if necessary
            remaining = n_examples - n_human_as_bot - n_bot_as_human
            n_random = remaining if remaining > 0 else 0
            
            # Generate adversarial examples based on human-as-bot misclassifications
            for _ in range(n_human_as_bot):
                if len(human_as_bot) > 0:
                    # Select a random misclassified human session
                    sample = human_as_bot.sample(1).iloc[0]
                    adversarial_examples.append(
                        self.generate_human_session_with_bot_characteristics(sample)
                    )
            
            # Generate adversarial examples based on bot-as-human misclassifications
            for _ in range(n_bot_as_human):
                if len(bot_as_human) > 0:
                    # Select a random misclassified bot session
                    sample = bot_as_human.sample(1).iloc[0]
                    adversarial_examples.append(
                        self.generate_bot_session_with_human_characteristics(
                            sample, bot_type=sample['bot_type']
                        )
                    )
            
            # Fill with random adversarial examples if needed
            for _ in range(n_random):
                adversarial_examples.append(self.generate_adversarial_bot_session())
        
        # Convert to DataFrame
        df = pd.DataFrame(adversarial_examples)
        
        return df

    def generate_adversarial_bot_session(self) -> Dict:
        """
        Generate an adversarial bot session designed to evade detection.
        
        Returns:
            Dictionary containing session data for an adversarial bot
        """
        # Start with a sophisticated bot as the base
        session = self.generate_bot_session(bot_type='sophisticated')
        
        # Modify it to be even more human-like
        
        # 1. Technical fingerprint modifications
        device_templates = self.device_templates['desktop']
        session['technical_fingerprint']['user_agent'] = random.choice(device_templates['user_agents'])
        session['technical_fingerprint']['screen_resolution'] = random.choice(device_templates['screen_resolutions'])
        session['technical_fingerprint']['color_depth'] = random.choice(device_templates['color_depths'])
        session['technical_fingerprint']['platform'] = random.choice(device_templates['platform'])
        
        # Ensure all storage options are enabled like most humans
        session['technical_fingerprint']['cookies_enabled'] = True
        session['technical_fingerprint']['session_storage'] = True
        session['technical_fingerprint']['local_storage'] = True
        session['technical_fingerprint']['indexed_db'] = True
        
        # 2. Navigation behavior modifications
        # Make page times more human-like with more variation
        for page in session['navigation_behavior']['page_times']:
            if page == 'product':
                session['navigation_behavior']['page_times'][page] = random.uniform(20, 120)
            elif page == 'checkout':
                session['navigation_behavior']['page_times'][page] = random.uniform(45, 180)
            else:
                session['navigation_behavior']['page_times'][page] = random.uniform(5, 30)
        
        # 3. Input behavior modifications
        session['input_behavior']['mouse_movement_count'] = random.randint(80, 250)
        session['input_behavior']['mouse_movement_consistency'] = random.uniform(0.65, 0.85)
        session['input_behavior']['movement_speed_variation'] = random.uniform(0.25, 0.5)
        session['input_behavior']['keyboard_input_speed'] = random.uniform(120, 250)
        session['input_behavior']['keyboard_input_consistency'] = random.uniform(0.7, 0.85)
        session['input_behavior']['form_input_natural'] = True
        session['input_behavior']['keyboard_error_rate'] = random.uniform(0.02, 0.08)
        
        # Add some rage clicks like humans occasionally do
        if random.random() < 0.3:
            session['input_behavior']['rage_clicks'] = random.randint(1, 3)
        
        # 4. Temporal pattern modifications
        session['temporal_patterns']['hour_of_day'] = random.choice([
            random.randint(8, 11),
            random.randint(12, 13),
            random.randint(17, 22)
        ])
        
        # Use weekend days more often like humans
        session['temporal_patterns']['day_of_week'] = random.choices(
            range(7),
            weights=[0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15],
            k=1
        )[0]
        
        # More variation in response times
        response_times = [random.uniform(0.7, 6.0) for _ in range(random.randint(4, 7))]
        session['temporal_patterns']['response_times'] = response_times
        session['temporal_patterns']['average_response_time'] = sum(response_times) / len(response_times)
        session['temporal_patterns']['response_time_std'] = self._calculate_std(response_times)
        session['temporal_patterns']['session_consistency'] = random.uniform(0.7, 0.85)
        session['temporal_patterns']['time_to_checkout'] = random.uniform(60, 240)
        
        # Keep label as sophisticated_bot but mark as adversarial
        session['label'] = 'sophisticated_bot'
        session['is_adversarial'] = True
        
        return session

    def generate_human_session_with_bot_characteristics(self, sample: pd.Series) -> Dict:
        """
        Generate a human session with some bot-like characteristics based on misclassifications.
        
        Args:
            sample: Series containing a misclassified human session
            
        Returns:
            Dictionary containing session data
        """
        # Start with a normal human session
        session = self.generate_human_session()
        
        # Analyze the sample to see what might have caused the misclassification
        predicted_bot_type = sample['predicted_label'].replace('_bot', '')
        
        # Introduce some characteristics of the predicted bot type
        if 'technical_fingerprint' in sample and random.random() < 0.5:
            # Use some technical aspects from the misclassified sample
            for key in ['platform', 'plugins_length', 'canvas_fingerprint']:
                if key in sample['technical_fingerprint'] and random.random() < 0.7:
                    session['technical_fingerprint'][key] = sample['technical_fingerprint'][key]
        
        if 'navigation_behavior' in sample and random.random() < 0.5:
            # Make navigation slightly more bot-like
            if predicted_bot_type == 'basic':
                # Faster page times
                for page in session['navigation_behavior']['page_times']:
                    session['navigation_behavior']['page_times'][page] *= random.uniform(0.3, 0.7)
            
            # More direct path
            if random.random() < 0.6 and len(session['navigation_behavior']['page_sequence']) > 4:
                # Remove some intermediate pages
                remove_count = random.randint(1, min(2, len(session['navigation_behavior']['page_sequence']) - 3))
                for _ in range(remove_count):
                    if len(session['navigation_behavior']['page_sequence']) > 3:
                        # Don't remove first, last, or product pages
                        removable_indices = [
                            i for i in range(1, len(session['navigation_behavior']['page_sequence']) - 1)
                            if session['navigation_behavior']['page_sequence'][i] != 'product'
                        ]
                        if removable_indices:
                            remove_idx = random.choice(removable_indices)
                            removed_page = session['navigation_behavior']['page_sequence'].pop(remove_idx)
                            if removed_page in session['navigation_behavior']['page_times']:
                                del session['navigation_behavior']['page_times'][removed_page]
        
        if 'input_behavior' in sample and random.random() < 0.6:
            # Adjust input behavior to be slightly more bot-like
            session['input_behavior']['mouse_movement_count'] = min(
                session['input_behavior']['mouse_movement_count'], 
                random.randint(15, 50)
            )
            session['input_behavior']['mouse_movement_consistency'] = min(
                session['input_behavior']['mouse_movement_consistency'] + random.uniform(0.1, 0.2),
                0.95
            )
            session['input_behavior']['keyboard_input_speed'] = min(
                session['input_behavior']['keyboard_input_speed'], 
                random.uniform(50, 100)
            )
        
        # Keep label as human but mark as adversarial
        session['label'] = 'human'
        session['is_adversarial'] = True
        
        return session

    def generate_bot_session_with_human_characteristics(self, sample: pd.Series, bot_type: str) -> Dict:
        """
        Generate a bot session with more human-like characteristics based on misclassifications.
        
        Args:
            sample: Series containing a misclassified bot session
            bot_type: Type of bot to generate
            
        Returns:
            Dictionary containing session data
        """
        # Start with a normal bot session of specified type
        session = self.generate_bot_session(bot_type=bot_type)
        
        # Make it more human-like based on what might have led to misclassification
        
        if 'technical_fingerprint' in sample and random.random() < 0.7:
            # Use some human-like technical aspects
            device_type = random.choice(['desktop', 'mobile'])
            templates = self.device_templates[device_type]
            
            session['technical_fingerprint']['user_agent'] = random.choice(templates['user_agents'])
            session['technical_fingerprint']['cookies_enabled'] = True
            session['technical_fingerprint']['session_storage'] = True
            session['technical_fingerprint']['local_storage'] = True
        
        if 'navigation_behavior' in sample and random.random() < 0.7:
            # Make navigation more human-like
            # More time spent on pages
            for page in session['navigation_behavior']['page_times']:
                if page == 'product':
                    session['navigation_behavior']['page_times'][page] = random.uniform(20, 180)
                elif page == 'checkout':
                    session['navigation_behavior']['page_times'][page] = random.uniform(45, 300)
                else:
                    session['navigation_behavior']['page_times'][page] = random.uniform(5, 30)
            
            # Add more intermediate pages
            for _ in range(random.randint(1, 3)):
                # Insert browsing pages in the middle of the sequence
                if len(session['navigation_behavior']['page_sequence']) > 2:
                    insert_idx = random.randint(1, len(session['navigation_behavior']['page_sequence']) - 1)
                    additional_page = random.choice(['category', 'search', 'product'])
                    session['navigation_behavior']['page_sequence'].insert(insert_idx, additional_page)
                    session['navigation_behavior']['page_times'][additional_page] = random.uniform(10, 60)
        
        if 'input_behavior' in sample and random.random() < 0.8:
            # Make input behavior more human-like
            session['input_behavior']['mouse_movement_count'] = random.randint(80, 250)
            session['input_behavior']['mouse_movement_consistency'] = random.uniform(0.65, 0.85)
            session['input_behavior']['movement_speed_variation'] = random.uniform(0.25, 0.5)
            session['input_behavior']['keyboard_input_speed'] = random.uniform(120, 250)
            session['input_behavior']['keyboard_input_consistency'] = random.uniform(0.7, 0.85)
            
            # Add occasional errors/corrections
            session['input_behavior']['keyboard_error_rate'] = random.uniform(0.02, 0.08)
            
            # Add occasional rage clicks
            if random.random() < 0.3:
                session['input_behavior']['rage_clicks'] = random.randint(1, 3)
        
        if 'temporal_patterns' in sample and random.random() < 0.7:
            # Make temporal patterns more human-like
            session['temporal_patterns']['hour_of_day'] = random.choice([
                random.randint(8, 11),
                random.randint(12, 13),
                random.randint(17, 22)
            ])
            
            # More human-like day of week distribution
            session['temporal_patterns']['day_of_week'] = random.choices(
                range(7),
                weights=[0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15],
                k=1
            )[0]
            
            # More natural response time variations
            response_times = [random.uniform(0.8, 8.0) for _ in range(random.randint(4, 7))]
            session['temporal_patterns']['response_times'] = response_times
            session['temporal_patterns']['average_response_time'] = sum(response_times) / len(response_times)
            session['temporal_patterns']['response_time_std'] = self._calculate_std(response_times)
            session['temporal_patterns']['session_consistency'] = random.uniform(0.7, 0.9)
        
        # Keep label as bot type but mark as adversarial
        session['label'] = f'{bot_type}_bot'
        session['is_adversarial'] = True
        
        return session

    # Active learning methods
    def generate_informative_samples(self, model, unlabeled_data: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate the most informative samples for labeling using active learning.
        
        Args:
            model: A trained model that can provide prediction probabilities
            unlabeled_data: DataFrame containing unlabeled session data
            n_samples: Number of samples to select for labeling
            
        Returns:
            DataFrame containing the most informative samples
        """
        logger.info(f"Selecting {n_samples} informative samples for active learning")
        
        # Check if model has predict_proba method (for uncertainty sampling)
        if hasattr(model, 'predict_proba'):
            # Get prediction probabilities
            try:
                probs = model.predict_proba(unlabeled_data)
                
                # Calculate uncertainty scores (1 - max probability)
                max_probs = np.max(probs, axis=1)
                uncertainty_scores = 1 - max_probs
                
                # Select most uncertain samples
                uncertain_indices = np.argsort(uncertainty_scores)[-n_samples:]
                informative_samples = unlabeled_data.iloc[uncertain_indices].copy()
                
                logger.info(f"Selected {len(informative_samples)} samples using uncertainty sampling")
                
                return informative_samples
                
            except Exception as e:
                logger.warning(f"Error in uncertainty sampling: {e}. Falling back to random sampling.")
                return unlabeled_data.sample(n_samples)
        else:
            logger.warning("Model does not support probability predictions. Using random sampling.")
            return unlabeled_data.sample(n_samples)
    
    def generate_balanced_dataset(self, n_sessions: int = 1000, balance_ratio: float = 0.5) -> pd.DataFrame:
        """
        Generate a balanced dataset with a specified ratio of human to bot sessions.
        
        Args:
            n_sessions: Total number of sessions to generate
            balance_ratio: Ratio of human sessions (0.5 means equal humans and bots)
            
        Returns:
            DataFrame containing balanced session data
        """
        logger.info(f"Generating balanced dataset with {n_sessions} sessions and {balance_ratio:.1f} human ratio")
        
        # Determine counts
        n_human = int(n_sessions * balance_ratio)
        n_bot = n_sessions - n_human
        
        # For bots, we want an equal distribution of types
        bot_types = ['basic', 'advanced', 'sophisticated']
        bot_per_type = n_bot // len(bot_types)
        remainder = n_bot % len(bot_types)
        
        bot_type_counts = {bot_type: bot_per_type for bot_type in bot_types}
        
        # Distribute remainder
        for i in range(remainder):
            bot_type_counts[bot_types[i]] += 1
        
        logger.info(f"Distribution - Human: {n_human}, Bots: {bot_type_counts}")
        
        # Generate sessions
        human_sessions = [self.generate_human_session() for _ in range(n_human)]
        
        bot_sessions = []
        for bot_type, count in bot_type_counts.items():
            for _ in range(count):
                bot_sessions.append(self.generate_bot_session(bot_type=bot_type))
        
        # Combine and shuffle
        all_sessions = human_sessions + bot_sessions
        random.shuffle(all_sessions)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_sessions)
        
        return df
    
    def augment_minority_class(self, dataset: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        Augment the minority class in an imbalanced dataset.
        
        Args:
            dataset: DataFrame containing the imbalanced dataset
            target_count: Target number of samples for each class
            
        Returns:
            DataFrame with augmented minority classes
        """
        logger.info(f"Augmenting minority classes to achieve {target_count} samples per class")
        
        # Count samples per class
        class_counts = dataset['label'].value_counts().to_dict()
        
        # Identify classes that need augmentation
        augmentation_needed = {
            label: max(0, target_count - count)
            for label, count in class_counts.items()
        }
        
        logger.info(f"Augmentation needed: {augmentation_needed}")
        
        # Generate additional samples for minority classes
        augmented_samples = []
        
        for label, count in augmentation_needed.items():
            if count <= 0:
                continue
                
            # Get existing samples of this class
            class_samples = dataset[dataset['label'] == label]
            
            if label == 'human':
                # Generate new human sessions
                new_samples = [self.generate_human_session() for _ in range(count)]
                augmented_samples.extend(new_samples)
            else:
                # Extract bot type from label
                bot_type = label.replace('_bot', '')
                
                # Generate new bot sessions
                new_samples = [self.generate_bot_session(bot_type=bot_type) for _ in range(count)]
                augmented_samples.extend(new_samples)
        
        # Convert new samples to DataFrame
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            
            # Combine with original dataset
            combined_df = pd.concat([dataset, augmented_df], ignore_index=True)
            
            # Shuffle the combined dataset
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
            
            logger.info(f"Added {len(augmented_samples)} new samples. New dataset size: {len(combined_df)}")
            
            return combined_df
        else:
            logger.info("No augmentation needed")
            return dataset

    def generate_validation_data(self, test_ratio: float = 0.2, validation_ratio: float = 0.2, 
                                n_sessions: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate training, validation, and test datasets.
        
        Args:
            test_ratio: Ratio of data to use for testing
            validation_ratio: Ratio of data to use for validation
            n_sessions: Total number of sessions to generate
            
        Returns:
            Tuple of (training_data, validation_data, test_data) DataFrames
        """
        logger.info(f"Generating {n_sessions} sessions for train/validation/test datasets")
        
        # Calculate sizes
        n_test = int(n_sessions * test_ratio)
        n_validation = int(n_sessions * validation_ratio)
        n_train = n_sessions - n_test - n_validation
        
        logger.info(f"Split: Train={n_train}, Validation={n_validation}, Test={n_test}")
        
        # Generate full dataset
        full_dataset = self.generate_dataset(n_sessions=n_sessions)
        
        # Ensure balanced representation in each split
        # Stratify by label to maintain class distribution
        labels = full_dataset['label']
        
        # Use stratified split if possible
        try:
            from sklearn.model_selection import train_test_split
            
            # First split off test set
            train_val_data, test_data = train_test_split(
                full_dataset, test_size=n_test, stratify=labels, random_state=42
            )
            
            # Then split validation from training
            val_ratio = n_validation / (n_train + n_validation)
            train_data, val_data = train_test_split(
                train_val_data, 
                test_size=val_ratio, 
                stratify=train_val_data['label'], 
                random_state=42
            )
            
            logger.info(f"Stratified split successful. Train={len(train_data)}, "
                      f"Validation={len(val_data)}, Test={len(test_data)}")
                      
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.warning(f"Error in stratified split: {e}. Using random split instead.")
            
            # Simple random split
            indices = list(range(len(full_dataset)))
            random.shuffle(indices)
            
            test_indices = indices[:n_test]
            val_indices = indices[n_test:n_test+n_validation]
            train_indices = indices[n_test+n_validation:]
            
            test_data = full_dataset.iloc[test_indices].copy()
            val_data = full_dataset.iloc[val_indices].copy()
            train_data = full_dataset.iloc[train_indices].copy()
            
            logger.info(f"Random split completed. Train={len(train_data)}, "
                      f"Validation={len(val_data)}, Test={len(test_data)}")
                      
            return train_data, val_data, test_data

    def save_dataset(self, dataset: pd.DataFrame, path: str) -> None:
        """
        Save a dataset to file.
        
        Args:
            dataset: DataFrame to save
            path: Path to save file
        """
        # Ensure directory exists
        output_dir = Path(path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        if path.endswith('.csv'):
            dataset.to_csv(path, index=False)
            logger.info(f"Saved dataset to {path}")
        elif path.endswith('.json'):
            dataset.to_json(path, orient='records')
            logger.info(f"Saved dataset to {path}")
        elif path.endswith('.pkl') or path.endswith('.pickle'):
            dataset.to_pickle(path)
            logger.info(f"Saved dataset to {path}")
        else:
            # Default to CSV
            path = f"{path}.csv"
            dataset.to_csv(path, index=False)
            logger.info(f"Saved dataset to {path}")

# Helper functions
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from file.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        DataFrame containing the loaded dataset
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.json':
        return pd.read_json(path)
    elif path.suffix in ['.pkl', '.pickle']:
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of DataFrames to merge
        
    Returns:
        Merged DataFrame
    """
    if not datasets:
        raise ValueError("No datasets provided for merging")
    
    # Check for consistent columns
    ref_columns = set(datasets[0].columns)
    for i, df in enumerate(datasets[1:], 1):
        if set(df.columns) != ref_columns:
            logging.warning(f"Dataset {i} has different columns than the first dataset")
    
    # Concatenate datasets
    merged = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates based on session_id
    if 'session_id' in merged.columns:
        merged = merged.drop_duplicates(subset=['session_id'])
    
    return merged

def extract_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and flatten features from a complex dataset structure.
    
    Args:
        dataset: DataFrame with nested features
        
    Returns:
        DataFrame with flattened features
    """
    # Initialize empty DataFrame for features
    features = pd.DataFrame(index=dataset.index)
    
    # Add basic features
    features['is_bot'] = dataset['is_bot']
    
    if 'label' in dataset.columns:
        features['label'] = dataset['label']
    
    # Process technical fingerprint features
    tech_columns = [
        'user_agent', 'screen_resolution', 'color_depth', 'platform', 
        'plugins_length', 'cookies_enabled', 'session_storage', 'local_storage'
    ]
    
    for col in tech_columns:
        try:
            features[f'tech_{col}'] = dataset['technical_fingerprint'].apply(
                lambda x: x.get(col) if isinstance(x, dict) else None
            )
        except:
            pass
    
    # Process navigation features
    try:
        features['nav_total_pages'] = dataset['navigation_behavior'].apply(
            lambda x: x.get('total_pages_visited') if isinstance(x, dict) else None
        )
        features['nav_session_duration'] = dataset['navigation_behavior'].apply(
            lambda x: x.get('session_duration') if isinstance(x, dict) else None
        )
        features['nav_add_to_cart'] = dataset['navigation_behavior'].apply(
            lambda x: x.get('add_to_cart') if isinstance(x, dict) else None
        )
        features['nav_checkout'] = dataset['navigation_behavior'].apply(
            lambda x: x.get('reached_checkout') if isinstance(x, dict) else None
        )
    except:
        pass
    
    # Process input behavior features
    input_columns = [
        'mouse_movement_count', 'mouse_movement_consistency', 'mouse_movement_entropy',
        'movement_speed_variation', 'keyboard_input_speed', 'keyboard_input_consistency'
    ]
    
    for col in input_columns:
        try:
            features[f'input_{col}'] = dataset['input_behavior'].apply(
                lambda x: x.get(col) if isinstance(x, dict) else None
            )
        except:
            pass
    
    # Process temporal features
    try:
        features['temp_hour_of_day'] = dataset['temporal_patterns'].apply(
            lambda x: x.get('hour_of_day') if isinstance(x, dict) else None
        )
        features['temp_day_of_week'] = dataset['temporal_patterns'].apply(
            lambda x: x.get('day_of_week') if isinstance(x, dict) else None
        )
        features['temp_avg_response_time'] = dataset['temporal_patterns'].apply(
            lambda x: x.get('average_response_time') if isinstance(x, dict) else None
        )
        features['temp_response_std'] = dataset['temporal_patterns'].apply(
            lambda x: x.get('response_time_std') if isinstance(x, dict) else None
        )
    except:
        pass
    
    # Drop any NaN columns that might have been created
    features = features.dropna(axis=1, how='all')
    
    return features

if __name__ == "__main__":
    # Simple demonstration
    generator = SyntheticDataGenerator()
    
    # Generate a small dataset
    dataset = generator.generate_dataset(n_sessions=100)
    
    # Display some statistics
    print(f"Generated {len(dataset)} sessions")
    print(f"Human sessions: {sum(~dataset['is_bot'])}")
    print(f"Bot sessions: {sum(dataset['is_bot'])}")
    
    if 'bot_type' in dataset.columns:
        bot_types = dataset.loc[dataset['is_bot'], 'bot_type'].value_counts()
        print("Bot type distribution:")
        for bot_type, count in bot_types.items():
            print(f"  {bot_type}: {count}")