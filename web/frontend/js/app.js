/**
 * Aplicación Principal - Portal de Información LLM-HPC
 * Carga y muestra información del proyecto mediante API
 */

const API_BASE_URL = 'http://localhost:5000/api';

// ==================== AUTENTICACIÓN ====================

/**
 * Obtiene el token JWT del localStorage
 */
function getToken() {
    return localStorage.getItem('jwt_token');
}

/**
 * Obtiene el CSRF token del localStorage
 */
function getCSRFToken() {
    return localStorage.getItem('csrf_token');
}

/**
 * Verifica autenticación y redirige si no está autenticado
 */
function checkAuth() {
    const token = getToken();

    if (!token) {
        window.location.href = 'login.html';
        return false;
    }

    // Verificar validez del token
    fetch(`${API_BASE_URL}/verify`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            localStorage.clear();
            window.location.href = 'login.html';
        }
    })
    .catch(() => {
        localStorage.clear();
        window.location.href = 'login.html';
    });

    return true;
}

/**
 * Cierra sesión
 */
function logout() {
    localStorage.clear();
    window.location.href = 'login.html';
}

// ==================== API CALLS ====================

/**
 * Realiza petición autenticada a la API
 */
async function fetchAPI(endpoint) {
    const token = getToken();

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.status === 401) {
            // Token inválido o expirado
            localStorage.clear();
            window.location.href = 'login.html';
            return null;
        }

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

// ==================== CARGA DE DATOS ====================

/**
 * Carga información general del proyecto
 */
async function loadProjectInfo() {
    const data = await fetchAPI('/project/info');
    if (!data) return;

    // Abstract
    const abstractDiv = document.getElementById('project-abstract');
    if (abstractDiv) {
        abstractDiv.classList.remove('loading');
        abstractDiv.innerHTML = `<p>${data.abstract}</p>`;
    }

    // Contribuciones principales
    const contributionsList = document.getElementById('key-contributions');
    if (contributionsList) {
        contributionsList.classList.remove('loading');
        contributionsList.innerHTML = data.key_contributions
            .map(item => `<li>✅ ${item}</li>`)
            .join('');
    }
}

/**
 * Carga detalles técnicos
 */
async function loadTechnicalDetails() {
    const data = await fetchAPI('/project/technical');
    if (!data) return;

    // Información del modelo
    const modelInfo = document.getElementById('model-info');
    if (modelInfo) {
        modelInfo.classList.remove('loading');
        modelInfo.innerHTML = `
            <ul>
                <li><strong>Nombre:</strong> ${data.model.name}</li>
                <li><strong>Parámetros:</strong> ${data.model.parameters}</li>
                <li><strong>Arquitectura:</strong> ${data.model.architecture}</li>
                <li><strong>Contexto:</strong> ${data.model.context_length}</li>
                <li><strong>Tokenizador:</strong> ${data.model.tokenizer}</li>
            </ul>
        `;
    }

    // Configuración de entrenamiento
    const trainingInfo = document.getElementById('training-info');
    if (trainingInfo) {
        trainingInfo.classList.remove('loading');
        trainingInfo.innerHTML = `
            <ul>
                <li><strong>Técnica:</strong> ${data.training.technique}</li>
                <li><strong>Cuantización:</strong> ${data.training.quantization}</li>
                <li><strong>LoRA Rank:</strong> ${data.training.lora_rank}</li>
                <li><strong>LoRA Alpha:</strong> ${data.training.lora_alpha}</li>
                <li><strong>Learning Rate:</strong> ${data.training.learning_rate}</li>
                <li><strong>Épocas:</strong> ${data.training.epochs}</li>
                <li><strong>Batch Size:</strong> ${data.training.batch_size}</li>
                <li><strong>Gradient Accumulation:</strong> ${data.training.gradient_accumulation}</li>
                <li><strong>Max Seq Length:</strong> ${data.training.max_seq_length}</li>
            </ul>
        `;
    }
}

/**
 * Carga métricas de entrenamiento
 */
async function loadMetrics() {
    const data = await fetchAPI('/project/metrics');
    if (!data) return;

    // Resumen de métricas
    const metricsSummary = document.getElementById('metrics-summary');
    if (metricsSummary) {
        metricsSummary.classList.remove('loading');
        metricsSummary.innerHTML = `
            <div class="metric-card">
                <h4>📉 Train Loss</h4>
                <div class="metric-value">
                    <span class="initial">${data.summary.train_loss.initial}</span>
                    <span class="arrow">→</span>
                    <span class="final">${data.summary.train_loss.final}</span>
                </div>
                <div class="metric-reduction">Reducción: ${data.summary.train_loss.reduction}</div>
            </div>
            <div class="metric-card">
                <h4>📊 Eval Loss</h4>
                <div class="metric-value">
                    <span class="initial">${data.summary.eval_loss.initial}</span>
                    <span class="arrow">→</span>
                    <span class="final">${data.summary.eval_loss.final}</span>
                </div>
                <div class="metric-reduction">Reducción: ${data.summary.eval_loss.reduction}</div>
            </div>
            <div class="metric-card">
                <h4>🎯 Perplexity</h4>
                <div class="metric-value">
                    <span class="initial">${data.summary.perplexity.initial}</span>
                    <span class="arrow">→</span>
                    <span class="final">${data.summary.perplexity.final}</span>
                </div>
                <div class="metric-reduction">Reducción: ${data.summary.perplexity.reduction}</div>
            </div>
            <div class="metric-card">
                <h4>📏 Grad Norm</h4>
                <div class="metric-value">
                    <span class="initial">${data.summary.grad_norm.initial}</span>
                    <span class="arrow">→</span>
                    <span class="final">${data.summary.grad_norm.final}</span>
                </div>
                <div class="metric-reduction">Reducción: ${data.summary.grad_norm.reduction}</div>
            </div>
        `;
    }

    // Progreso por época
    const epochProgress = document.getElementById('epoch-progress');
    if (epochProgress) {
        epochProgress.classList.remove('loading');

        const table = `
            <div class="table-responsive">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Época</th>
                            <th>Train Loss</th>
                            <th>Eval Loss</th>
                            <th>Perplexity</th>
                            <th>Learning Rate</th>
                            <th>Grad Norm</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.epoch_progress.map(epoch => `
                            <tr>
                                <td><strong>${epoch.epoch}</strong></td>
                                <td>${epoch.train_loss.toFixed(4)}</td>
                                <td>${epoch.eval_loss.toFixed(4)}</td>
                                <td>${epoch.perplexity.toFixed(2)}</td>
                                <td>${epoch.learning_rate.toExponential(2)}</td>
                                <td>${epoch.grad_norm.toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        epochProgress.innerHTML = table;
    }
}

/**
 * Carga información de infraestructura
 */
async function loadInfrastructure() {
    const data = await fetchAPI('/project/infrastructure');
    if (!data) return;

    // Hardware
    const hardwareInfo = document.getElementById('hardware-info');
    if (hardwareInfo) {
        hardwareInfo.classList.remove('loading');
        hardwareInfo.innerHTML = `
            <h4>🖥️ Nodos GPU</h4>
            <ul>
                <li><strong>Modelo:</strong> ${data.hpc.gpu_nodes.model}</li>
                <li><strong>GPUs por nodo:</strong> ${data.hpc.gpu_nodes.gpu_per_node} × ${data.hpc.gpu_nodes.gpu_model}</li>
                <li><strong>Memoria GPU:</strong> ${data.hpc.gpu_nodes.gpu_memory} (${data.hpc.gpu_nodes.total_gpu_memory} total)</li>
                <li><strong>CPU:</strong> ${data.hpc.gpu_nodes.cpu_model} (${data.hpc.gpu_nodes.cpu_cores} cores)</li>
                <li><strong>RAM/core:</strong> ${data.hpc.gpu_nodes.ram_per_core}</li>
            </ul>
            <h4>💻 Nodos CPU</h4>
            <ul>
                <li><strong>Cantidad:</strong> ${data.hpc.cpu_nodes.count} nodos</li>
                <li><strong>Cores totales:</strong> ${data.hpc.cpu_nodes.total_cores}</li>
            </ul>
        `;
    }

    // Software
    const softwareInfo = document.getElementById('software-info');
    if (softwareInfo) {
        softwareInfo.classList.remove('loading');
        softwareInfo.innerHTML = `
            <ul>
                <li><strong>OS:</strong> ${data.software.os}</li>
                <li><strong>Python:</strong> ${data.software.python}</li>
                <li><strong>PyTorch:</strong> ${data.software.pytorch}</li>
                <li><strong>CUDA:</strong> ${data.software.cuda}</li>
                <li><strong>Transformers:</strong> ${data.software.transformers}</li>
                <li><strong>Unsloth:</strong> ${data.software.unsloth}</li>
            </ul>
            <h4>📦 Frameworks</h4>
            <ul>
                ${data.software.frameworks.map(f => `<li>${f}</li>`).join('')}
            </ul>
        `;
    }
}

/**
 * Carga información de metodología
 */
async function loadMethodology() {
    const data = await fetchAPI('/project/methodology');
    if (!data) return;

    const methodologyContent = document.getElementById('methodology-content');
    if (methodologyContent) {
        methodologyContent.classList.remove('loading');
        methodologyContent.innerHTML = `
            <div class="card">
                <h3>🔬 Visión General</h3>
                <p>${data.overview}</p>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h3>⚡ QLoRA</h3>
                    <p>${data.techniques.qlora.description}</p>
                    <h4>Beneficios:</h4>
                    <ul>
                        ${data.techniques.qlora.benefits.map(b => `<li>${b}</li>`).join('')}
                    </ul>
                    <h4>Componentes:</h4>
                    <ul>
                        ${data.techniques.qlora.components.map(c => `<li>${c}</li>`).join('')}
                    </ul>
                </div>

                <div class="card">
                    <h3>🎯 LoRA</h3>
                    <p>${data.techniques.lora.description}</p>
                    <h4>Parámetros:</h4>
                    <ul>
                        <li><strong>Rank:</strong> ${data.techniques.lora.parameters.rank}</li>
                        <li><strong>Alpha:</strong> ${data.techniques.lora.parameters.alpha}</li>
                        <li><strong>Dropout:</strong> ${data.techniques.lora.parameters.dropout}</li>
                        <li><strong>Módulos objetivo:</strong> ${data.techniques.lora.parameters.target_modules.join(', ')}</li>
                    </ul>
                </div>
            </div>

            <div class="card">
                <h3>🚀 Optimización</h3>
                <ul>
                    <li><strong>Optimizador:</strong> ${data.techniques.optimization.optimizer}</li>
                    <li><strong>Scheduler:</strong> ${data.techniques.optimization.scheduler}</li>
                    <li><strong>Warmup Ratio:</strong> ${data.techniques.optimization.warmup_ratio}</li>
                    <li><strong>Weight Decay:</strong> ${data.techniques.optimization.weight_decay}</li>
                    <li><strong>Max Grad Norm:</strong> ${data.techniques.optimization.max_grad_norm}</li>
                </ul>
            </div>
        `;
    }
}

/**
 * Carga información de colaboración
 */
async function loadCollaboration() {
    const data = await fetchAPI('/project/collaboration');
    if (!data) return;

    const collaborationContent = document.getElementById('collaboration-content');
    if (collaborationContent) {
        collaborationContent.classList.remove('loading');
        collaborationContent.innerHTML = `
            <div class="card">
                <h3>🌎 ${data.project}</h3>
                <p><strong>Organización:</strong> ${data.organization}</p>
                <p><strong>Objetivo:</strong> ${data.objective}</p>
            </div>

            <div class="card">
                <h3>🤝 Contribuciones de este Proyecto</h3>
                <ul>
                    ${data.contributions.map(c => `<li>✅ ${c}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    const regionalImpact = document.getElementById('regional-impact');
    if (regionalImpact) {
        regionalImpact.classList.remove('loading');
        regionalImpact.innerHTML = `
            <ul>
                ${data.regional_impact.map(i => `<li>🌟 ${i}</li>`).join('')}
            </ul>
        `;
    }
}

// ==================== INTERFAZ ====================

/**
 * Muestra información del usuario
 */
function displayUserInfo() {
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    const usernameSpan = document.getElementById('username');

    if (usernameSpan && user.username) {
        usernameSpan.textContent = user.username;
    }
}

/**
 * Navegación suave
 */
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Resalta sección activa en navegación
 */
function setupActiveSection() {
    const sections = document.querySelectorAll('.content-section');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (window.pageYOffset >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ==================== INICIALIZACIÓN ====================

document.addEventListener('DOMContentLoaded', async () => {
    // Verificar autenticación
    if (!checkAuth()) return;

    // Mostrar información del usuario
    displayUserInfo();

    // Configurar logout
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }

    // Configurar navegación
    setupSmoothScroll();
    setupActiveSection();

    // Cargar datos del proyecto
    await Promise.all([
        loadProjectInfo(),
        loadTechnicalDetails(),
        loadMetrics(),
        loadInfrastructure(),
        loadMethodology(),
        loadCollaboration()
    ]);

    console.log('✅ Aplicación inicializada correctamente');
});
