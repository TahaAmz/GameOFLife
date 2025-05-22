# Import Necessary Libraries
import os
import random
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import colorchooser, messagebox, ttk

import numpy as np

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    CUDA_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

class AIAgent(nn.Module):
    """AI Agent for Game of Life"""
    
    def __init__(self):
        super(AIAgent, self).__init__()
        self.device = DEVICE if CUDA_AVAILABLE else "cpu"
        
        if CUDA_AVAILABLE:
            # Optimized neural network
            self.neuralNetwork = nn.Sequential(
                nn.Linear(29, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()).to(self.device)
            
            self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
            self.criterion = nn.BCELoss()
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        self.training_data = []
        self.training_labels = []
        self.is_trained = False
        self.training_count = 0
        self.last_predictions = None
        self.last_suggestions = []
        
    def feature_extraction(self, grid, x, y):
        """Extract Features from Neighborhood"""
        features = []
        height, width = len(grid), len(grid[0])
        
        # 5x5 Neighborhood (25 Features)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    features.append(1.0 if grid[ny][nx] else 0.0)
                else:
                    features.append(0.0) 
        
        # Additional Context Features (4 features)
        neighbors_3x3 = sum(
            1 for dy in range(-1, 2) for dx in range(-1, 2)
            if dx != 0 or dy != 0
            and 0 <= x + dx < width and 0 <= y + dy < height
            and grid[y + dy][x + dx])
        
        features.extend([
            neighbors_3x3 / 8.0,
            1.0 if grid[y][x] else 0.0,
            x / width,
            y / height])
        
        return np.array(features, dtype=np.float32)
    
    def generation_data(self, old_grid, new_grid):
        """Collect Training Data"""
        if not CUDA_AVAILABLE:
            return
            
        height, width = len(old_grid), len(old_grid[0])
        sample_points = []
        
        # Sample Areas with Activity
        for y in range(1, height-1, 2):
            for x in range(1, width-1, 2):
                neighbors = sum(
                    1 for dy in range(-1, 2) for dx in range(-1, 2)
                    if old_grid[y + dy][x + dx]
                )
                if neighbors > 0 or new_grid[y][x]:
                    sample_points.append((x, y))
        
        # Add Random Samples for Diversity
        for _ in range(min(50, len(sample_points) // 2)):
            x = random.randint(1, width-2)
            y = random.randint(1, height-2)
            sample_points.append((x, y))
        
        # Extract Features and Labels
        for x, y in sample_points[:200]:
            try:
                features = self.feature_extraction(old_grid, x, y)
                label = 1.0 if new_grid[y][x] else 0.0
                
                self.training_data.append(features)
                self.training_labels.append(label)
            except Exception as e:
                continue
        
        # Keep Only the Recent Data
        if len(self.training_data) > 2000:
            self.training_data = self.training_data[-1500:]
            self.training_labels = self.training_labels[-1500:]
        
        # Train Periodically
        if len(self.training_data) >= 200 and len(self.training_data) % 100 == 0:
            self.train_model()
    
    def train_model(self):
        """Train the NN Model"""
        if not CUDA_AVAILABLE or len(self.training_data) < 50:
            return
        
        try:
            # Data Preperation
            X = torch.FloatTensor(self.training_data).to(self.device)
            y = torch.FloatTensor(self.training_labels).unsqueeze(1).to(self.device)
            
            if torch.isnan(X).any() or torch.isnan(y).any():
                print("Warning: NaN values in training data")
                return
            
            self.train()
            self.optimizer.zero_grad()
            
            # Forward Pass with Mixed Precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.neuralNetwork(X)
                    loss = self.criterion(predictions, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.neuralNetwork(X)
                loss = self.criterion(predictions, y)
                loss.backward()
                self.optimizer.step()
            
            self.is_trained = True
            self.training_count += 1
            
            self.last_predictions = None
            self.last_suggestions = []
            
        except Exception as e:
            print(f"Training error: {e}")
    
    def next_state(self, grid):
        """Predict Next State Using the Model"""
        if not CUDA_AVAILABLE or not self.is_trained:
            return None
        
        # Use Cached Predictions 
        if self.last_predictions is not None:
            return self.last_predictions
        
        height, width = len(grid), len(grid[0])
        predictions = np.zeros((height, width), dtype=np.float32)
        
        try:
            self.eval()
            with torch.no_grad():
                batch_size = 32
                
                for y_start in range(0, height, 4):
                    for x_start in range(0, width, 4):
                        batch_features = []
                        batch_positions = []
                        
                        # Collect Batch
                        for y in range(y_start, min(y_start + 4, height)):
                            for x in range(x_start, min(x_start + 4, width)):
                                try:
                                    features = self.feature_extraction(grid, x, y)
                                    batch_features.append(features)
                                    batch_positions.append((x, y))
                                    
                                    if len(batch_features) >= batch_size:
                                        break
                                except:
                                    continue
                            if len(batch_features) >= batch_size:
                                break
                        
                        # Process Batch
                        if batch_features:
                            try:
                                X_batch = torch.FloatTensor(batch_features).to(self.device)
                                pred_batch = self.neuralNetwork(X_batch).cpu().numpy().flatten()
                                
                                for (x, y), pred in zip(batch_positions, pred_batch):
                                    predictions[y][x] = pred
                                    
                            except Exception as e:
                                continue
                
                # Interpolate Skipped Cells
                for y in range(height):
                    for x in range(width):
                        if predictions[y][x] == 0 and (x % 4 != 0 or y % 4 != 0):
                            nearby_values = []
                            for dy in [-4, 0, 4]:
                                for dx in [-4, 0, 4]:
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < height and 0 <= nx < width and predictions[ny][nx] > 0:
                                        nearby_values.append(predictions[ny][nx])
                            
                            if nearby_values:
                                predictions[y][x] = np.mean(nearby_values)
                
                self.last_predictions = predictions
                
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None
        
        return predictions
    
    def suggest_intervention(self, grid):
        """Suggest Intervention Points"""
        if not CUDA_AVAILABLE or not self.is_trained:
            return []
        
        # Use Cached Suggestions
        if self.last_suggestions:
            return self.last_suggestions
        
        height, width = len(grid), len(grid[0])
        suggestions = []
        
        try:
            self.eval()
            with torch.no_grad():
                for y in range(2, height - 2, 3):
                    for x in range(2, width - 2, 3):
                        if not grid[y][x]:
                            try:
                                features = self.feature_extraction(grid, x, y)
                                X = torch.FloatTensor([features]).to(self.device)
                                probability = self.neuralNetwork(X).item()
                                
                                if probability > 0.7:
                                    suggestions.append((x, y, probability))
                                    
                            except Exception as e:
                                continue
                
                self.last_suggestions = sorted(suggestions, key=lambda x: x[2], reverse=True)[:10]
                
        except Exception as e:
            print(f"Suggestion Error: {e}")
        
        return self.last_suggestions

class SimpleAgent:
    """Simple Fallback Agent (Cuda is not Available)"""
    
    def __init__(self):
        self.is_trained = False
        self.generation_count = 0
        self.last_predictions = None
        self.last_suggestions = []
    
    def generation_data(self, old_grid, new_grid):
        """Simple Learning Simulation"""
        self.generation_count += 1
        if self.generation_count > 5:
            self.is_trained = True
        
        self.last_predictions = None
        self.last_suggestions = []
    
    def next_state(self, grid):
        """Simple Rule-Based Prediction"""
        if not self.is_trained:
            return None
        
        if self.last_predictions is not None:
            return self.last_predictions
        
        height, width = len(grid), len(grid[0])
        predictions = np.zeros((height, width))
        
        # Conway's Game of Life Rules
        for y in range(height):
            for x in range(width):
                neighbors = sum(
                    1 for dy in range(-1, 2) for dx in range(-1, 2)
                    if dx != 0 or dy != 0
                    and 0 <= x + dx < width and 0 <= y + dy < height
                    and grid[y + dy][x + dx])
                
                if grid[y][x]:
                    predictions[y][x] = 0.8 if neighbors in [2, 3] else 0.2
                else:
                    predictions[y][x] = 0.8 if neighbors == 3 else 0.1
        
        self.last_predictions = predictions
        return predictions
    
    def suggest_intervention(self, grid):
        """Simple Heuristic Suggestions"""
        if self.last_suggestions:
            return self.last_suggestions
        
        suggestions = []
        height, width = len(grid), len(grid[0])
        
        # Suggest Cells that Would Become Alive
        for y in range(1, height - 1, 3):
            for x in range(1, width - 1, 3):
                if not grid[y][x]:
                    neighbors = sum(
                        1 for dy in range(-1, 2) for dx in range(-1, 2)
                        if dx != 0 or dy != 0
                        and grid[y + dy][x + dx])
                    
                    if neighbors == 3:
                        suggestions.append((x, y, 0.9))
                    elif neighbors == 2:
                        suggestions.append((x, y, 0.6))
        
        self.last_suggestions = sorted(suggestions, key=lambda x: x[2], reverse=True)[:8]
        return self.last_suggestions
    
    def train_model(self):
        """Training the Simulation"""
        self.is_trained = True

class GameOfLifeGUI:
    """Conway's Game of Life GUI"""
    
    def __init__(self, width=100, height=60):
        self.width = width
        self.height = height
        self.cell_size = 7
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        self.generation = 0
        self.is_running = False
        
        # Speed Options
        self.speed_options = {"Very Fast": 30, "Fast": 80, "Medium": 150, "Slow": 300, "Very Slow": 600}
        self.current_speed = 150
        
        # AI Agent
        if CUDA_AVAILABLE:
            self.agent = AIAgent()
        else:
            self.agent = SimpleAgent()
            
        self.show_predictions = False
        self.show_suggestions = False
        self.update_counter = 0
        
        # Colors
        self.colors = {
            'alive': '#FFFFFF',
            'dead': '#000000',
            'prediction_high': '#FF3366',
            'prediction_medium': '#FF9933',
            'prediction_low': '#FFCC33',
            'suggestion': '#3399FF',
            'grid_line': '#1a1a1a'}
        
        # Enhanced Patterns
        self.patterns = self.init_patterns()
        
        self.setup_gui()
        self.pattern_history = deque(maxlen=10)
        
    def init_patterns(self):
        """Initialize classic Game of Life patterns."""
        return {
            # Oscillators
            'blinker': [(1, 0), (1, 1), (1, 2)],
            'toad': [(1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)],
            'beacon': [(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)],
            'pulsar': [
                (2, 0), (3, 0), (4, 0), (8, 0), (9, 0), (10, 0),
                (0, 2), (5, 2), (7, 2), (12, 2), (0, 3), (5, 3), (7, 3), (12, 3),
                (0, 4), (5, 4), (7, 4), (12, 4), (2, 5), (3, 5), (4, 5), (8, 5), (9, 5), (10, 5),
                (2, 7), (3, 7), (4, 7), (8, 7), (9, 7), (10, 7), (0, 8), (5, 8), (7, 8), (12, 8),
                (0, 9), (5, 9), (7, 9), (12, 9), (0, 10), (5, 10), (7, 10), (12, 10),
                (2, 12), (3, 12), (4, 12), (8, 12), (9, 12), (10, 12)],
            
            # Spaceships
            'glider': [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
            'lightweight_spaceship': [(0, 1), (3, 1), (4, 2), (0, 3), (4, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
            'middleweight_spaceship': [(2, 0), (0, 1), (4, 1), (5, 2), (0, 3), (5, 3), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4)],
            
            # Still Lifes
            'block': [(0, 0), (1, 0), (0, 1), (1, 1)],
            'beehive': [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)],
            'loaf': [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (3, 2), (2, 3)],
            
            # Special Patterns
            'heart': [(1, 0), (2, 0), (5, 0), (6, 0), (0, 1), (3, 1), (4, 1), (7, 1),
                     (0, 2), (7, 2), (1, 3), (6, 3), (2, 4), (5, 4), (3, 5), (4, 5)],
            'arrow': [(4, 0), (2, 1), (4, 1), (6, 1), (1, 2), (4, 2), (7, 2),
                     (0, 3), (4, 3), (8, 3), (1, 4), (4, 4), (7, 4), (2, 5), (4, 5), (6, 5), (4, 6)],
            'exploder': [(0, 0), (2, 0), (4, 0), (0, 1), (4, 1), (0, 2), (4, 2),
                        (0, 3), (2, 3), (4, 3), (0, 4), (2, 4), (4, 4)]}
        
    def setup_gui(self):
        """Initialize the GUI"""
        self.root = tk.Tk()
        self.root.title(f"John Conway's Game of Life with AI ({'PyTorch + ' + str(DEVICE).upper() if CUDA_AVAILABLE else 'CPU Only'})")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        # Main Frame with Scrollbars
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=8)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Simulation Controls
        sim_frame = ttk.Frame(control_frame)
        sim_frame.pack(fill=tk.X, pady=3)
        
        self.start_button = ttk.Button(sim_frame, text="Start", command=self.toggle_simulation, width=10)
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(sim_frame, text="Step", command=self.step_generation, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="Clear", command=self.clear_grid, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="Random", command=self.randomize_grid, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="Colors", command=self.choose_colors, width=10).pack(side=tk.LEFT, padx=2)
        
        # Speed Control
        ttk.Label(sim_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.StringVar(value="Medium")
        speed_dropdown = ttk.Combobox(sim_frame, textvariable=self.speed_var,
                                     values=list(self.speed_options.keys()),
                                     state="readonly", width=12)
        speed_dropdown.pack(side=tk.LEFT, padx=2)
        speed_dropdown.bind("<<ComboboxSelected>>", self.update_speed)
        
        # Pattern controls
        pattern_frame = ttk.LabelFrame(control_frame, text="Patterns", padding=5)
        pattern_frame.pack(fill=tk.X, pady=3)
        
        # Pattern buttons in rows
        pattern_buttons = [
            [("Glider", 'glider'), ("Blinker", 'blinker'), ("Beacon", 'beacon'), ("Block", 'block'), ("Toad", 'toad')],
            [("Light Ship", 'lightweight_spaceship'), ("Mid Ship", 'middleweight_spaceship'), ("Pulsar", 'pulsar'), ("Beehive", 'beehive')],
            [("Heart", 'heart'), ("Arrow", 'arrow'), ("Exploder", 'exploder'), ("Loaf", 'loaf')]]
        
        for row_buttons in pattern_buttons:
            row = ttk.Frame(pattern_frame)
            row.pack(fill=tk.X, pady=1)
            for text, pattern in row_buttons:
                ttk.Button(row, text=text, command=lambda p=pattern: self.add_pattern(p), width=10).pack(side=tk.LEFT, padx=1)
        
        # AI Controls
        ai_frame = ttk.LabelFrame(control_frame, text=f"AI Controls", padding=5)
        ai_frame.pack(fill=tk.X, pady=3)
        
        ai_row1 = ttk.Frame(ai_frame)
        ai_row1.pack(fill=tk.X, pady=1)
        
        self.prediction_var = tk.BooleanVar()
        self.suggestion_var = tk.BooleanVar()
        
        ttk.Checkbutton(ai_row1, text="Predictions", variable=self.prediction_var,
                       command=self.toggle_predictions).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(ai_row1, text="Suggestions", variable=self.suggestion_var,
                       command=self.toggle_suggestions).pack(side=tk.LEFT, padx=3)
        
        ai_row2 = ttk.Frame(ai_frame)
        ai_row2.pack(fill=tk.X, pady=1)
        
        ttk.Button(ai_row2, text="AI Intervene", command=self.ai_intervene, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(ai_row2, text="Force Train", command=self.train_ai, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(ai_row2, text="Save Pattern", command=self.save_pattern, width=12).pack(side=tk.LEFT, padx=2)
        
        # Info Panel
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=3)
        
        self.info_label = ttk.Label(info_frame, text="Generation: 0 | Living Cells: 0 | AI Status: Initializing")
        self.info_label.pack(side=tk.LEFT)
        
        # Canvas for the Grid
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add Scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        
        self.canvas = tk.Canvas(canvas_frame,
                               width=min(800, self.width * self.cell_size),
                               height=min(600, self.height * self.cell_size),
                               bg='black',
                               scrollregion=(0, 0, self.width * self.cell_size, self.height * self.cell_size),
                               xscrollcommand=h_scrollbar.set,
                               yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        # Pack Scrollbars and Canvas
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind Mouse Events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
        # Keyboard Shortcuts
        self.root.bind("<space>", lambda e: self.toggle_simulation())
        self.root.bind("<Return>", lambda e: self.step_generation())
        self.root.bind("<Delete>", lambda e: self.clear_grid())
        
        self.root.focus_set()
        
        self.draw_grid()
        self.update_display()
        
    def on_mousewheel(self, event):
        """Handle Mouse Wheel Scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def toggle_predictions(self):
        """Toggle Prediction Display"""
        self.show_predictions = self.prediction_var.get()
        if self.show_predictions:
            self.agent.last_predictions = None
        self.update_display()
        
    def toggle_suggestions(self):
        """Toggle Suggestion Display"""
        self.show_suggestions = self.suggestion_var.get()
        if self.show_suggestions:
            self.agent.last_suggestions = []
        self.update_display()
        
    def ai_intervene(self):
        """Let AI Suggest and Apply Interventions"""
        if not self.agent.is_trained:
            messagebox.showinfo("AI Not Ready",
                              "AI agent needs more training data!\n"
                              "Run the simulation for a few generations first.")
            return
        
        suggestions = self.agent.suggest_intervention(self.grid)
        if not suggestions:
            messagebox.showinfo("AI Intervention", "No good intervention points found.")
            return
            
        applied = 0
        confidence_levels = []
        
        for x, y, prob in suggestions[:5]:
            if 0 <= x < self.width and 0 <= y < self.height and not self.grid[y][x]:
                self.grid[y][x] = True
                applied += 1
                confidence_levels.append(f"{prob:.2f}")
        
        if applied > 0:
            self.update_display()
            messagebox.showinfo("AI Intervention",
                              f"Applied {applied} AI suggestions!\n"
                              f"Confidence levels: {', '.join(confidence_levels)}")
        else:
            messagebox.showinfo("AI Intervention", "All suggested positions were already occupied.")
    
    def train_ai(self):
        """Force Train the AI"""
        training_data_size = len(getattr(self.agent, 'training_data', []))
        if training_data_size < 50:
            messagebox.showwarning("Insufficient Data",
                                 f"Need more training data! Current: {training_data_size}\n"
                                 "Run the simulation for several generations first.")
            return
        
        # Train in Background Thread
        def train_thread():
            self.agent.train_model()
            self.root.after(100, lambda: messagebox.showinfo("AI Training", "AI model training completed!"))
        
        threading.Thread(target=train_thread, daemon=True).start()
        messagebox.showinfo("Training Started", "AI training started in background...")
        
    def choose_colors(self):
        """Open Color Chooser"""
        color = colorchooser.askcolor(title="Choose Living Cell Color ")[1]
        if color:
            self.colors['alive'] = color
            self.update_display()
    
    def update_speed(self, event=None):
        """Update Simulation Speed"""
        speed_name = self.speed_var.get()
        if speed_name in self.speed_options:
            self.current_speed = self.speed_options[speed_name]
    
    def on_canvas_click(self, event):
        """Handle Mouse Clicks on Canvas"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)
        
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = not self.grid[y][x]
            self.clear_ai_cache()
            self.update_display()
    
    def on_canvas_drag(self, event):
        """Handle Mouse Drag on Canvas"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)
        
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = True
            self.clear_ai_cache()
            self.update_display()
    
    def clear_ai_cache(self):
        """Clear AI Predictions Cache When Grid Changes"""
        if hasattr(self.agent, 'last_predictions'):
            self.agent.last_predictions = None
        if hasattr(self.agent, 'last_suggestions'):
            self.agent.last_suggestions = []
    
    def draw_grid(self):
        """Draw Grid Lines"""
        for i in range(0, self.width + 1, 1):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.height * self.cell_size,
                                  fill=self.colors['grid_line'], width=1, tags="grid")
        
        for i in range(0, self.height + 1, 1):
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.width * self.cell_size, y,
                                  fill=self.colors['grid_line'], width=1, tags="grid")
    
    def update_display(self):
        """Update the Visual Display"""
        self.canvas.delete("cell")
        
        # Get AI Data
        predictions = None
        suggestions = []
        
        if self.show_predictions and self.agent.is_trained:
            predictions = self.agent.next_state(self.grid)
        
        if self.show_suggestions and self.agent.is_trained:
            suggestions = self.agent.suggest_intervention(self.grid)
        
        # Create Suggestion Lookup for O(1) Access
        suggestion_dict = {(x, y): prob for x, y, prob in suggestions}
        
        # Batch Create Rectangles
        rectangles_to_draw = []
        
        for y in range(self.height):
            for x in range(self.width):
                x1, y1 = x * self.cell_size + 1, y * self.cell_size + 1
                x2, y2 = (x + 1) * self.cell_size - 1, (y + 1) * self.cell_size - 1
                
                color = None
                
                if self.grid[y][x]:
                    color = self.colors['alive']
                elif (x, y) in suggestion_dict and self.show_suggestions:
                    color = self.colors['suggestion']
                elif predictions is not None and self.show_predictions:
                    prob = predictions[y][x]
                    if prob > 0.7:
                        color = self.colors['prediction_high']
                    elif prob > 0.4:
                        color = self.colors['prediction_medium']
                    elif prob > 0.2:
                        color = self.colors['prediction_low']
                
                if color:
                    rectangles_to_draw.append((x1, y1, x2, y2, color))
        
        # Draw Rectangles
        for x1, y1, x2, y2, color in rectangles_to_draw:
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags="cell")
        
        # Update Info
        if self.update_counter % 3 == 0:
            self.update_info()
        self.update_counter += 1
    
    def update_info(self):
        """Update Information Display"""
        living_cells = sum(sum(row) for row in self.grid)
        
        if CUDA_AVAILABLE:
            training_samples = len(getattr(self.agent, 'training_data', []))
            training_count = getattr(self.agent, 'training_count', 0)
            ai_status = f"Trained ({training_count} epochs)" if self.agent.is_trained else f"Learning ({training_samples} samples)"
        else:
            ai_status = "CPU Mode (Basic Features)"
        
        self.info_label.config(
            text=f"Generation: {self.generation} | Living Cells: {living_cells} | AI Status: {ai_status}"
        )
    
    def next_generation(self):
        """Calculate and Display Next Generation with Optimizations"""
        old_grid = [row[:] for row in self.grid]
        new_grid = [[False for _ in range(self.width)] for _ in range(self.height)]
        
        # Optimized Neighbor Counting with Boundary Checking
        for y in range(self.height):
            for x in range(self.width):
                neighbors = 0
                
                # Count Neighbors Efficiently
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height and old_grid[ny][nx]:
                            neighbors += 1
                
                # Apply Conway's rules
                if old_grid[y][x]:
                    if neighbors in [2, 3]:
                        new_grid[y][x] = True
                else:
                    if neighbors == 3:
                        new_grid[y][x] = True
        
        # Let AI Learn in Background Thread
        if hasattr(self.agent, 'generation_data'):
            def learn_background():
                try:
                    self.agent.generation_data(old_grid, new_grid)
                except Exception as e:
                    print(f"Background Learning Error: {e}")
            
            threading.Thread(target=learn_background, daemon=True).start()
        
        self.grid = new_grid
        self.generation += 1
        
        # Clear AI Cache
        self.clear_ai_cache()
        
        self.update_display()
        
        # Pattern Detection
        pattern_hash = hash(str(self.grid))
        self.pattern_history.append(pattern_hash)
        
        # Detect Stable Patterns or Cycles
        if len(self.pattern_history) >= 4:
            recent_patterns = list(self.pattern_history)[-4:]
            if len(set(recent_patterns)) <= 2:
                self.is_running = False
                self.start_button.config(text="Start")
                pattern_type = "stable pattern" if len(set(recent_patterns)) == 1 else "oscillating pattern"
                messagebox.showinfo("Pattern Detected", f"Detected {pattern_type}! Simulation auto-stopped.")
    
    def count_neighbors(self, x, y):
        """Count Living Neighbors of a Cell"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx]:
                    count += 1
        return count
    
    def toggle_simulation(self):
        """Start or Stop the Simulation"""
        self.is_running = not self.is_running
        if self.is_running:
            self.start_button.config(text="Stop")
            self.run_simulation()
        else:
            self.start_button.config(text="Start")
    
    def run_simulation(self):
        """Run the Simulation Loop"""
        if self.is_running:
            try:
                self.next_generation()
                self.root.after(self.current_speed, self.run_simulation)
            except Exception as e:
                print(f"Simulation Error: {e}")
                self.is_running = False
                self.start_button.config(text="Start")
                messagebox.showerror("Simulation Error", f"Simulation Stopped: {e}")
    
    def step_generation(self):
        """Advance One Generation"""
        try:
            self.next_generation()
        except Exception as e:
            print(f"Step Error: {e}")
            messagebox.showerror("Step Error", f"Could Not Advance Generation: {e}")
    
    def clear_grid(self):
        """Clear Grid and Reset"""
        self.grid = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.generation = 0
        self.pattern_history.clear()
        self.clear_ai_cache()
        self.update_display()
    
    def randomize_grid(self):
        """Randomly Populate Grid"""
        density = 0.15
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = random.random() < density
        
        self.clear_ai_cache()
        self.update_display()
    
    def add_pattern(self, pattern_name):
        """Add Predefined Patterns with Smart Placement"""
        if pattern_name not in self.patterns:
            messagebox.showerror("Pattern Error", f"Pattern '{pattern_name}' not found!")
            return
        
        pattern = self.patterns[pattern_name]
        
        # Find Optimal Placement
        start_x, start_y = self.find_optimal_placement(pattern)
        
        # Place Pattern
        placed_cells = 0
        for dx, dy in pattern:
            x, y = start_x + dx, start_y + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = True
                placed_cells += 1
        
        if placed_cells > 0:
            self.clear_ai_cache()
            self.update_display()
            # messagebox.showinfo("Pattern Added", f"Added {pattern_name} ({placed_cells} cells)")
        else:
            messagebox.showwarning("Pattern Error", "Could Not Place Pattern - No Valid Space Found!")
    
    def find_optimal_placement(self, pattern):
        """Find the Best Location to Place Patterns"""
        if not pattern:
            return self.width // 2, self.height // 2
        
        min_x = min(dx for dx, dy in pattern)
        max_x = max(dx for dx, dy in pattern)
        min_y = min(dy for dx, dy in pattern)
        max_y = max(dy for dx, dy in pattern)
        
        pattern_width = max_x - min_x + 1
        pattern_height = max_y - min_y + 1
        
        # Try Center
        center_x = self.width // 2 - pattern_width // 2
        center_y = self.height // 2 - pattern_height // 2
        
        if self.is_area_suitable(center_x, center_y, pattern_width + 4, pattern_height + 4):
            return center_x - min_x, center_y - min_y
        
        # Try Random
        for attempt in range(20):
            rand_x = random.randint(5, self.width - pattern_width - 5)
            rand_y = random.randint(5, self.height - pattern_height - 5)
            
            if self.is_area_suitable(rand_x, rand_y, pattern_width + 2, pattern_height + 2):
                return rand_x - min_x, rand_y - min_y
        
        # Fallback to Center
        return max(5, center_x - min_x), max(5, center_y - min_y)
    
    def is_area_suitable(self, start_x, start_y, width, height):
        """Check if an Area is Suitable for Pattern Placement"""
        if start_x < 0 or start_y < 0 or start_x + width >= self.width or start_y + height >= self.height:
            return False
        
        # Count Existing Cells in Area
        occupied_cells = 0
        total_cells = width * height
        
        for y in range(start_y, start_y + height):
            for x in range(start_x, start_x + width):
                if self.grid[y][x]:
                    occupied_cells += 1
        
        # Area is Suitable if Less Than 20% Occupied
        return occupied_cells / total_cells < 0.2
    
    def save_pattern(self):
        """Save Current Pattern to File with Metadata"""
        try:
            timestamp = int(time.time())
            filename = f"gol_pattern_{timestamp}.txt"
            
            living_cells = sum(sum(row) for row in self.grid)
            
            with open(filename, 'w') as f:
                f.write(f"# Game of Life Pattern\n")
                f.write(f"# Generated: {time.ctime()}\n")
                f.write(f"# Generation: {self.generation}\n")
                f.write(f"# Grid Size: {self.width}x{self.height}\n")
                f.write(f"# Living Cells: {living_cells}\n")
                f.write(f"# AI Status: {'Trained' if self.agent.is_trained else 'Untrained'}\n")
                f.write("# Format: 1=alive, 0=dead\n\n")
                
                for row in self.grid:
                    f.write(''.join('1' if cell else '0' for cell in row) + '\n')
            
            messagebox.showinfo("Pattern Saved", 
                              f"Pattern Saved Sucessfully!\n"
                              f"File: {filename}\n"
                              f"Size: {living_cells} Living Cells")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Could Not Save Pattern: {e}")
    
    def run(self):
        """Start the GUI Application"""
        try:
            try:
                self.root.iconbitmap('game_of_life.ico')  # Optional icon file
            except:
                pass
            
            # Center Window on Screen
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            # Start the Main Loop
            self.root.mainloop()
            
        except Exception as e:
            print(f"GUI Error: {e}")
            messagebox.showerror("Application Error", f"Application Error: {e}")

def main():
    """Run the Game of Life"""
    print("\n" + "=" * 40)
    print("John Conway's Game of Life with AI")
    print("=" * 40)
    
    if CUDA_AVAILABLE:
        print(f"\nPyTorch Engine: {DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {DEVICE}")
    
    print("\nControls:")
    print("- Click/Drag: Draw cells")
    print("- Space: Start/Stop simulation")
    print("- Enter: Step one generation")
    print("- Delete: Clear grid")
    
    print("\nAI Features:")
    print("- Predictions: Shows where cells will likely appear")
    print("- Suggestions: AI recommends cell placements")
    print("- Auto-learning: AI improves as it watches patterns\n")
    
    try:
        game = GameOfLifeGUI(width=100, height=60)
        game.run()
        
    except ImportError as e:
        print(f"\nMissing Dependency: {e}")
        
    except Exception as e:
        print(f"\nError Running the Game: {e}")

if __name__ == "__main__":
    main()