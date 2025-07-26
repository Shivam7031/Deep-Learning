<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Learning Repository</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Roboto+Mono:wght@400;500&display=swap');
        
        :root {
            --primary: #6c63ff;
            --secondary: #4a42e8;
            --dark: #1e1e2c;
            --light: #f8f9fa;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
            --info: #17a2b8;
            --transition: all 0.3s ease;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background-color: var(--light);
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: var(--shadow);
            transform: translateY(20px);
            opacity: 0;
            animation: fadeIn 0.5s ease forwards;
        }
        
        @keyframes fadeIn {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 15px;
            color: var(--primary);
            position: relative;
            display: inline-block;
        }
        
        h1::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 50px;
            height: 4px;
            background: var(--primary);
            border-radius: 2px;
            transition: var(--transition);
        }
        
        h1:hover::after {
            width: 100px;
            background: var(--secondary);
        }
        
        h2 {
            font-size: 1.8rem;
            margin: 30px 0 15px;
            color: var(--secondary);
            padding-bottom: 8px;
            border-bottom: 2px dashed #eee;
            transition: var(--transition);
        }
        
        h2:hover {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }
        
        h3 {
            font-size: 1.4rem;
            margin: 20px 0 10px;
            color: var(--dark);
        }
        
        p {
            margin-bottom: 15px;
        }
        
        a {
            color: var(--primary);
            text-decoration: none;
            transition: var(--transition);
        }
        
        a:hover {
            color: var(--secondary);
            text-decoration: underline;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 8px;
            transition: var(--transition);
        }
        
        .badge-primary {
            background-color: var(--primary);
            color: white;
        }
        
        .badge-secondary {
            background-color: var(--secondary);
            color: white;
        }
        
        .badge-success {
            background-color: var(--success);
            color: white;
        }
        
        .badge-warning {
            background-color: var(--warning);
            color: var(--dark);
        }
        
        .badge-danger {
            background-color: var(--danger);
            color: white;
        }
        
        .badge-info {
            background-color: var(--info);
            color: white;
        }
        
        ul, ol {
            margin-left: 20px;
            margin-bottom: 20px;
        }
        
        li {
            margin-bottom: 8px;
            transition: var(--transition);
        }
        
        li:hover {
            transform: translateX(5px);
        }
        
        code {
            font-family: 'Roboto Mono', monospace;
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        pre {
            font-family: 'Roboto Mono', monospace;
            background-color: #282c34;
            color: #abb2bf;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 20px 0;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: var(--shadow);
            transition: var(--transition);
            border: 1px solid #eee;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
            border-color: var(--primary);
        }
        
        .card h3 {
            margin-top: 0;
            color: var(--primary);
        }
        
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background: var(--primary);
            color: white;
            border-radius: 5px;
            font-weight: 600;
            transition: var(--transition);
            border: none;
            cursor: pointer;
            margin: 5px;
        }
        
        .btn:hover {
            background: var(--secondary);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(108, 99, 255, 0.3);
            text-decoration: none;
        }
        
        .btn-outline {
            background: transparent;
            border: 2px solid var(--primary);
            color: var(--primary);
        }
        
        .btn-outline:hover {
            background: var(--primary);
            color: white;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }
        
        .floating {
            animation: floating 3s ease-in-out infinite;
        }
        
        @keyframes floating {
            0% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-10px);
            }
            100% {
                transform: translateY(0px);
            }
        }
        
        .highlight {
            position: relative;
            z-index: 1;
        }
        
        .highlight::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(108, 99, 255, 0.1);
            z-index: -1;
            border-radius: 5px;
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }
        
        .highlight:hover::before {
            transform: scaleX(1);
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: var(--dark);
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            h2 {
                font-size: 1.5rem;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1 class="floating">Deep Learning Repository</h1>
            <p>A comprehensive guide to deep learning concepts, architectures, tools, and implementations</p>
            
            <div style="margin: 20px 0;">
                <span class="badge badge-primary">Neural Networks</span>
                <span class="badge badge-secondary">TensorFlow</span>
                <span class="badge badge-success">PyTorch</span>
                <span class="badge badge-warning">Computer Vision</span>
                <span class="badge badge-danger">NLP</span>
                <span class="badge badge-info">Reinforcement Learning</span>
            </div>
            
            <div style="margin-top: 30px;">
                <a href="#components" class="btn pulse">Explore Concepts</a>
                <a href="https://github.com/yourusername/deep-learning-repo" target="_blank" class="btn btn-outline">View on GitHub</a>
            </div>
        </header>
        
        <section id="components">
            <h2>🧠 Core Components of Deep Learning</h2>
            <p>Deep learning systems consist of several fundamental components:</p>
            
            <div class="grid">
                <div class="card">
                    <h3>Neural Networks</h3>
                    <ul>
                        <li>Basic building blocks of deep learning</li>
                        <li>Composed of interconnected layers of neurons</li>
                        <li>Learn hierarchical representations of data</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Activation Functions</h3>
                    <ul>
                        <li>ReLU, Sigmoid, Tanh, Softmax</li>
                        <li>Introduce non-linearity to the network</li>
                        <li>Determine neuron output</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Loss Functions</h3>
                    <ul>
                        <li>MSE, Cross-Entropy, Hinge Loss</li>
                        <li>Measure difference between prediction and truth</li>
                        <li>Guide the learning process</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Optimizers</h3>
                    <ul>
                        <li>SGD, Adam, RMSprop</li>
                        <li>Update model weights to minimize loss</li>
                        <li>Control learning dynamics</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Regularization</h3>
                    <ul>
                        <li>Dropout, L1/L2, BatchNorm</li>
                        <li>Prevent overfitting</li>
                        <li>Improve generalization</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Backpropagation</h3>
                    <ul>
                        <li>Algorithm for training neural networks</li>
                        <li>Calculates gradients of loss function</li>
                        <li>Chain rule of calculus</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <section id="types">
            <h2>🏗️ Types of Deep Learning Models</h2>
            <p>Different architectures for different tasks:</p>
            
            <div class="grid">
                <div class="card">
                    <h3>Feedforward Networks</h3>
                    <ul>
                        <li>Basic neural networks</li>
                        <li>Information flows in one direction</li>
                        <li>Good for structured data</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Convolutional Networks (CNNs)</h3>
                    <ul>
                        <li>Specialized for grid-like data (images)</li>
                        <li>Use convolutional layers</li>
                        <li>Translation invariant</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Recurrent Networks (RNNs)</h3>
                    <ul>
                        <li>For sequential data (text, time series)</li>
                        <li>Have memory of previous inputs</li>
                        <li>Variants: LSTM, GRU</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Transformers</h3>
                    <ul>
                        <li>Attention-based architectures</li>
                        <li>State-of-the-art for NLP</li>
                        <li>Examples: BERT, GPT</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Autoencoders</h3>
                    <ul>
                        <li>Unsupervised learning</li>
                        <li>Dimensionality reduction</li>
                        <li>Anomaly detection</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>GANs</h3>
                    <ul>
                        <li>Generative Adversarial Networks</li>
                        <li>Generator vs Discriminator</li>
                        <li>Create synthetic data</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <section id="tools">
            <h2>🛠️ Deep Learning Tools & Frameworks</h2>
            
            <div class="grid">
                <div class="card">
                    <h3>TensorFlow</h3>
                    <ul>
                        <li>Google's deep learning framework</li>
                        <li>High-level Keras API</li>
                        <li>Excellent production deployment</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>PyTorch</h3>
                    <ul>
                        <li>Facebook's research-focused framework</li>
                        <li>Dynamic computation graphs</li>
                        <li>Popular in academia</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>JAX</h3>
                    <ul>
                        <li>NumPy with automatic differentiation</li>
                        <li>Functional programming approach</li>
                        <li>Growing in popularity</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>ONNX</h3>
                    <ul>
                        <li>Open Neural Network Exchange</li>
                        <li>Model interoperability</li>
                        <li>Framework-agnostic deployment</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Hugging Face</h3>
                    <ul>
                        <li>State-of-the-art NLP models</li>
                        <li>Transformer library</li>
                        <li>Model hub</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Weights & Biases</h3>
                    <ul>
                        <li>Experiment tracking</li>
                        <li>Visualization</li>
                        <li>Collaboration</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <section id="getting-started">
            <h2>🚀 Getting Started</h2>
            
            <h3>Installation</h3>
            <pre><code># With conda
conda create -n dl-env python=3.8
conda activate dl-env
conda install pytorch torchvision -c pytorch

# With pip
pip install torch torchvision tensorflow</code></pre>
            
            <h3>Basic Neural Network Example</h3>
            <pre><code>import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
print(model)</code></pre>
            
            <h3>Training Loop</h3>
            <pre><code>criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')</code></pre>
        </section>
        
        <section id="resources">
            <h2>📚 Learning Resources</h2>
            
            <div class="grid">
                <div class="card">
                    <h3>Books</h3>
                    <ul>
                        <li>Deep Learning by Goodfellow, Bengio, Courville</li>
                        <li>Neural Networks and Deep Learning by Nielsen</li>
                        <li>Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Courses</h3>
                    <ul>
                        <li>Deep Learning Specialization (Andrew Ng)</li>
                        <li>Fast.ai Practical Deep Learning</li>
                        <li>CS231n: CNN for Visual Recognition</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>Research Papers</h3>
                    <ul>
                        <li>Attention Is All You Need (Transformers)</li>
                        <li>ResNet (Deep Residual Learning)</li>
                        <li>GANs (Generative Adversarial Networks)</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <footer class="footer">
            <p>Made with ❤️ by <a href="https://github.com/yourusername" target="_blank">Your Name</a></p>
            <p>© 2023 Deep Learning Repository</p>
        </footer>
    </div>
</body>
</html>
