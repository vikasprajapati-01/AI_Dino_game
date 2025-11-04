# Installation commands
"""
# For CPU version 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU version
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install stable-baselines3[extra]
pip install gymnasium
pip install mss
pip install pydirectinput
pip install pytesseract
pip install opencv-python
pip install matplotlib
pip install numpy<2.0
pip install pandas<2.1
pip install protobuf
"""

import os
import time
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mss import mss
import pydirectinput

warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import pytesseract
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
        print("Pytesseract and Tesseract engine loaded successfully")
    except:
        TESSERACT_AVAILABLE = False
        print("Pytesseract installed but Tesseract engine not found")
        print("Will use pixel-based detection instead")
except ImportError as e:
    TESSERACT_AVAILABLE = False
    print(f"Pytesseract not available: {e}")
    print("Will use pixel-based detection instead")

class ModernWebGame(Env):
    def __init__(self, 
                 game_location={'top': 150, 'left': 400, 'width': 600, 'height': 400},
                 done_location={'top': 230, 'left': 680, 'width': 500, 'height': 50},
                 use_ocr=True):
        super().__init__()
        self.observation_space = Box(
            low=0, high=255, 
            shape=(1, 83, 100), 
            dtype=np.uint8
        )
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = game_location
        self.done_location = done_location
        self.use_ocr = use_ocr and TESSERACT_AVAILABLE
        self.action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        } 
        self.current_frame = None
        self.episode_steps = 0 
        self.max_episode_steps = 1500
        self.game_over_threshold = 9001000
        print("ModernWebGame environment initialized")
        print(f"Game capture area: {game_location}")
        print(f"Done detection area: {done_location}")
        print(f"OCR enabled: {self.use_ocr}")
        if not self.use_ocr:
            print(f"Using pixel threshold: {self.game_over_threshold:,}")
    
    def step(self, action):
        self.episode_steps += 1
        if action != 2:
            try:
                pydirectinput.press(self.action_map[action])
                time.sleep(0.1)
            except Exception as e:
                print(f"Input error: {e}")
        observation = self.get_observation()
        done, done_cap = self.get_done()
        terminated = bool(done)
        reward = self._calculate_reward(terminated)
        truncated = bool(self.episode_steps >= self.max_episode_steps)
        info = {
            'episode_steps': self.episode_steps,
            'action_taken': self.action_map[action],
            'done_method': 'ocr' if self.use_ocr else 'pixel'
        }
        return observation, float(reward), terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        time.sleep(0.5)
        try:
            pydirectinput.click(x=700, y=400)
            time.sleep(0.2)
            pydirectinput.press('space')
        except Exception as e:
            print(f"Reset input error: {e}")
        time.sleep(0.5)
        observation = self.get_observation()
        info = {'episode_steps': 0}
        return observation, info
    
    def render(self, mode='human'):
        if self.current_frame is not None:
            if mode == 'human':
                cv2.imshow('Game Frame', self.current_frame)
                cv2.waitKey(1)
            elif mode == 'rgb_array':
                return self.current_frame
            elif mode == 'matplotlib':
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 2, 1)
                plt.imshow(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
                plt.title(f"Game Frame - Step {self.episode_steps}")
                plt.axis('off')
                obs = self.get_observation()
                plt.subplot(2, 2, 2)
                plt.imshow(obs[0], cmap='gray')
                plt.title('Processed Observation (83x100)')
                plt.axis('off')
                done, done_cap = self.get_done()
                plt.subplot(2, 2, 3)
                plt.imshow(done_cap)
                plt.title(f'Done Detection Area - Done: {done}')
                plt.axis('off')
                plt.subplot(2, 2, 4)
                info_text = f"""
Step: {self.episode_steps}
Detection: {'OCR' if self.use_ocr else 'Pixel'}
Threshold: {self.game_over_threshold:,}
Time: {time.strftime('%H:%M:%S')}
                """
                plt.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis('off')
                plt.title('Environment Info')
                plt.tight_layout()
                plt.show()
    
    def close(self):
        cv2.destroyAllWindows()
        print("Environment closed")
    
    def get_observation(self):
        try:
            raw = np.array(self.cap.grab(self.game_location))[:, :, :3].astype(np.uint8)
            self.current_frame = raw.copy()
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 83))
            channel = np.reshape(resized, (1, 83, 100))
            return channel
        except Exception as e:
            print(f"Screen capture error: {e}")
            return np.zeros((1, 83, 100), dtype=np.uint8)
    
    def get_done(self):
        try:
            done_cap = np.array(self.cap.grab(self.done_location))
            done = False
            if self.use_ocr and TESSERACT_AVAILABLE:
                try:
                    text = pytesseract.image_to_string(done_cap)[:10].upper()
                    done_strings = ['GAME', 'OVER', 'END', 'GAHE', 'RESTART']
                    done = any(s in text for s in done_strings)
                    if done:
                        print(f"OCR detected game over: '{text.strip()}'")
                except Exception as ocr_error:
                    pixel_sum = np.sum(done_cap)
                    done = pixel_sum > self.game_over_threshold
            else:
                pixel_sum = np.sum(done_cap)
                done = pixel_sum > self.game_over_threshold
                if self.episode_steps <= 5:
                    print(f"Pixel sum: {pixel_sum:,}, Threshold: {self.game_over_threshold:,}, Done: {done}")
            return bool(done), done_cap
        except Exception as e:
            print(f"Done detection error: {e}")
            return False, np.zeros((50, 500, 3), dtype=np.uint8)
    
    def _calculate_reward(self, done):
        if done:
            return -100.0
        else:
            return 1.0

class ModernTrainingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'model_step_{self.n_calls}')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}")
        return True

def test_environment():
    print("Testing Environment Setup")
    print("=" * 50)
    try:
        env = ModernWebGame()
        print("Environment created successfully")
        obs, info = env.reset()
        print(f"Reset successful - Observation shape: {obs.shape}")
        print(f"   Observation type: {type(obs)}, dtype: {obs.dtype}")
        print("Testing random actions...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: Action={action} ({env.action_map[action]}), Reward={reward}, Done={done} (type: {type(done)}), Truncated={truncated} (type: {type(truncated)})")
            assert isinstance(done, bool), f"done must be bool, got {type(done)}"
            assert isinstance(truncated, bool), f"truncated must be bool, got {type(truncated)}"
            assert isinstance(reward, (int, float)), f"reward must be numeric, got {type(reward)}"
            if done or truncated:
                print(f"   Episode ended: Done={done}, Truncated={truncated}")
                break
        done, done_cap = env.get_done()
        print(f"Done detection working - Status: {done} (type: {type(done)})")
        print(f"   Done capture shape: {done_cap.shape}")
        print("Testing rendering...")
        env.render(mode='matplotlib')
        env.close()
        print("Environment test completed successfully!")
        return True
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_random_episodes(num_episodes=3):
    print(f"Running {num_episodes} Random Episodes")
    print("=" * 50)
    env = ModernWebGame()
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}")
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            while not (done or truncated) and steps < 100:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if steps % 10 == 0:
                    print(f"   Step {steps}: Reward={total_reward:.1f}")
                if done or truncated:
                    reason = "Game Over" if done else "Time Limit"
                    print(f"   Episode ended: {reason}")
                    break
            print(f"   Episode {episode + 1} completed: {steps} steps, Total reward: {total_reward:.1f}")
            time.sleep(1)
    except Exception as e:
        print(f"Random episodes failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()



print("\nStarting DQN Training...")

env = ModernWebGame()
env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    target_update_interval=500,
    train_freq=4,
    batch_size=32,
)

callback = ModernTrainingCallback(check_freq=25_000, save_path="./models/")

model.learn(total_timesteps=1000, callback=callback)

model.save("dino_dqn_final")
env.close()


# def test_trained_model(model_path, num_episodes=3):
#     print(f"\nTesting Trained Model: {model_path}")
#     print("=" * 60)
    
#     env = ModernWebGame()
    
#     try:
#         model = DQN.load(model_path)
#         print(f"Model loaded successfully from {model_path}")
        
#         episode_results = []
        
#         for episode in range(num_episodes):
#             print(f"\nTest Episode {episode + 1}")
#             print("-" * 30)
            
#             obs, info = env.reset()
#             done = False
#             truncated = False
#             total_reward = 0
#             steps = 0
            
#             while not (done or truncated):
#                 action, _states = model.predict(obs, deterministic=True)
#                 action = int(action)
                
#                 obs, reward, done, truncated, info = env.step(action)
#                 total_reward += reward
#                 steps += 1
                
#                 if steps % 50 == 0:
#                     print(f"   Step {steps}: Action={env.action_map[action]}, Reward={total_reward:.1f}")
                
#                 if done:
#                     print(f"   Game Over detected at step {steps}")
#                     break
#                 elif truncated:
#                     print(f"   Episode truncated at step {steps} (time limit)")
#                     break
                
#                 time.sleep(0.02)
            
#             episode_result = {
#                 'episode': episode + 1,
#                 'steps': steps,
#                 'total_reward': total_reward,
#                 'outcome': 'died' if done else 'survived'
#             }
#             episode_results.append(episode_result)
            
#             print(f"   Episode {episode + 1} completed:")
#             print(f"     Steps: {steps}")
#             print(f"     Total Reward: {total_reward:.1f}")
#             print(f"     Outcome: {episode_result['outcome']}")
            
#             time.sleep(1)
        
#         print(f"\n📊 Testing Summary:")
#         print("=" * 40)
#         avg_steps = sum(r['steps'] for r in episode_results) / len(episode_results)
#         avg_reward = sum(r['total_reward'] for r in episode_results) / len(episode_results)
#         survival_count = sum(1 for r in episode_results if r['outcome'] == 'survived')
#         survival_rate = (survival_count / len(episode_results)) * 100
        
#         print(f"Average Steps: {avg_steps:.1f}")
#         print(f"Average Reward: {avg_reward:.1f}")
#         print(f"Survival Rate: {survival_rate:.1f}% ({survival_count}/{len(episode_results)})")
        
#         print(f"\nDetailed Results:")
#         for result in episode_results:
#             print(f"  Episode {result['episode']}: {result['steps']} steps, "
#                   f"{result['total_reward']:.1f} reward, {result['outcome']}")
    
#     except FileNotFoundError:
#         print(f"Model file not found: {model_path}")
#         print("Available models:")
#         if os.path.exists("./models/"):
#             for file in os.listdir("./models/"):
#                 if file.endswith('.zip'):
#                     print(f"  - ./models/{file}")
#         if os.path.exists("dino_dqn_final.zip"):
#             print(f"  - dino_dqn_final.zip")
    
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         env.close()

# # test_trained_model("./models/model_step_25000.zip", num_episodes=3)
 
# def main():
#     print("\nTesting Environment...")
#     if test_environment():
#         print("Environment test passed!")

#         run_random_episodes(num_episodes=4)
        
#     else:
#         print("Environment test failed")

# if __name__ == "__main__":
#     main()