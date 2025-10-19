// ====================================
// SKIN DISEASE DETECTOR - MAIN JAVASCRIPT
// ====================================

// Variables globales
let uploadedFile = null;
let isProcessing = false;
let progressInterval = null;

// Inicialización cuando el DOM está listo
$(document).ready(function() {
    initializeApp();
});

// ====================================
// INICIALIZACIÓN DE LA APLICACIÓN
// ====================================
function initializeApp() {
    console.log('🚀 Inicializando Skin Disease Detector...');
    
    // Configurar drag and drop
    setupDragAndDrop();
    
    // Configurar upload de archivos
    setupFileUpload();
    
    // Configurar quick prediction
    setupQuickPrediction();
    
    // Configurar tooltips
    initializeTooltips();
    
    // Configurar smooth scrolling
    setupSmoothScrolling();
    
    // Mostrar disclaimer al cargar
    setTimeout(showMedicalDisclaimer, 2000);
    
    console.log('✅ Aplicación inicializada correctamente');
}

// ====================================
// DRAG AND DROP FUNCTIONALITY
// ====================================
function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    if (!dropZone) return;
    
    // Prevenir comportamiento por defecto
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Resaltar zona de drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    // Manejar el drop
    dropZone.addEventListener('drop', handleDrop, false);
    
    console.log('📁 Sistema de drag & drop configurado');
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    document.getElementById('dropZone').classList.add('drag-over');
}

function unhighlight(e) {
    document.getElementById('dropZone').classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
}

// ====================================
// FILE UPLOAD FUNCTIONALITY
// ====================================
function setupFileUpload() {
    const fileInput = document.getElementById('imageFile');
    if (!fileInput) return;
    
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
    
    console.log('📤 Sistema de upload configurado');
}

function handleFileSelection(file) {
    // Validar tipo de archivo
    if (!validateFile(file)) {
        return;
    }
    
    uploadedFile = file;
    
    // Mostrar preview
    showImagePreview(file);
    
    // Habilitar botones
    updateUIState('file_selected');
    
    // Análisis automático si está habilitado
    if (document.getElementById('autoAnalyze') && document.getElementById('autoAnalyze').checked) {
        setTimeout(() => startPrediction(), 1000);
    }
}

function validateFile(file) {
    // Validar tipo
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('error', 'Tipo de archivo no válido', 'Por favor seleccione una imagen en formato JPG, PNG o GIF.');
        return false;
    }
    
    // Validar tamaño (máximo 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB en bytes
    if (file.size > maxSize) {
        showAlert('error', 'Archivo muy grande', 'El archivo debe ser menor a 10MB.');
        return false;
    }
    
    return true;
}

function showImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const previewContainer = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        
        if (previewImg) {
            previewImg.src = e.target.result;
            previewContainer.classList.remove('d-none');
            
            // Añadir animación
            previewImg.style.opacity = '0';
            setTimeout(() => {
                previewImg.style.transition = 'opacity 0.5s ease';
                previewImg.style.opacity = '1';
            }, 100);
        }
        
        // Actualizar información del archivo
        updateFileInfo(file);
    };
    
    reader.readAsDataURL(file);
}

function updateFileInfo(file) {
    const fileInfoElement = document.getElementById('fileInfo');
    if (!fileInfoElement) return;
    
    const fileSize = formatFileSize(file.size);
    const fileName = file.name;
    
    fileInfoElement.innerHTML = `
        <div class="alert alert-info">
            <i class="fas fa-file-image me-2"></i>
            <strong>${fileName}</strong> (${fileSize})
        </div>
    `;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ====================================
// PREDICTION FUNCTIONALITY
// ====================================
function setupQuickPrediction() {
    const quickPredictBtn = document.getElementById('quickPredictBtn');
    if (quickPredictBtn) {
        quickPredictBtn.addEventListener('click', startPrediction);
    }
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', startPrediction);
    }
    
    console.log('🔬 Sistema de predicción configurado');
}

function startPrediction() {
    if (!uploadedFile) {
        showAlert('warning', 'No hay imagen', 'Por favor seleccione una imagen antes de analizar.');
        return;
    }
    
    if (isProcessing) {
        console.log('⏳ Ya hay una predicción en proceso...');
        return;
    }
    
    // Preparar formulario
    const formData = new FormData();
    formData.append('image', uploadedFile);
    formData.append('csrfmiddlewaretoken', $('[name=csrfmiddlewaretoken]').val());
    
    // Iniciar proceso
    isProcessing = true;
    updateUIState('processing');
    
    // Llamada AJAX
    $.ajax({
        url: '/api/predict/',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        xhr: function() {
            const xhr = new window.XMLHttpRequest();
            
            // Monitor de progreso de upload
            xhr.upload.addEventListener('progress', function(evt) {
                if (evt.lengthComputable) {
                    const percentComplete = (evt.loaded / evt.total) * 100;
                    updateProgress(Math.round(percentComplete), 'Subiendo imagen...');
                }
            }, false);
            
            return xhr;
        },
        success: function(response) {
            console.log('✅ Predicción exitosa:', response);
            handlePredictionSuccess(response);
        },
        error: function(xhr, status, error) {
            console.error('❌ Error en predicción:', error);
            handlePredictionError(xhr, error);
        },
        complete: function() {
            isProcessing = false;
            updateUIState('completed');
        }
    });
}

function handlePredictionSuccess(response) {
    // Simular progreso de procesamiento
    updateProgress(100, 'Procesando imagen...');
    
    setTimeout(() => {
        // Mostrar resultados
        displayResults(response);
        
        // Redireccionar si hay ID
        if (response.prediction_id) {
            setTimeout(() => {
                window.location.href = `/prediction/${response.prediction_id}/`;
            }, 2000);
        }
    }, 1500);
}

function handlePredictionError(xhr, error) {
    let errorMessage = 'Error interno del servidor';
    
    if (xhr.responseJSON && xhr.responseJSON.error) {
        errorMessage = xhr.responseJSON.error;
    } else if (xhr.status === 413) {
        errorMessage = 'La imagen es demasiado grande';
    } else if (xhr.status === 415) {
        errorMessage = 'Formato de imagen no soportado';
    }
    
    showAlert('error', 'Error en el análisis', errorMessage);
    updateUIState('error');
}

function displayResults(response) {
    const resultsContainer = document.getElementById('predictionResults');
    if (!resultsContainer) return;
    
    // Crear HTML de resultados
    const resultsHTML = `
        <div class="card mt-4 fade-in-up">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-microscope me-2"></i>
                    Resultado del Análisis
                </h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4 text-center">
                        <div class="progress-circle mb-3" data-percentage="${response.confidence || 0}">
                            <div class="progress-text">${response.confidence || 0}%</div>
                        </div>
                        <h6>Confianza</h6>
                    </div>
                    <div class="col-md-8">
                        <h4 class="mb-3">${response.disease_name || 'Procesando...'}</h4>
                        <p class="text-muted mb-3">${response.description || ''}</p>
                        
                        ${response.severity ? `
                        <div class="mb-3">
                            <span class="severity-badge severity-${response.severity.toLowerCase().replace(' ', '-')} text-white">
                                ${response.severity}
                            </span>
                        </div>
                        ` : ''}
                        
                        ${response.recommendation ? `
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            <strong>Recomendación:</strong> ${response.recommendation}
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsContainer.innerHTML = resultsHTML;
    resultsContainer.classList.remove('d-none');
    
    // Actualizar círculo de progreso
    if (response.confidence) {
        setTimeout(() => {
            updateProgressCircle('.progress-circle', response.confidence);
        }, 500);
    }
    
    // Scroll suave a resultados
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// ====================================
// UI STATE MANAGEMENT
// ====================================
function updateUIState(state) {
    const elements = {
        dropZone: document.getElementById('dropZone'),
        quickPredictBtn: document.getElementById('quickPredictBtn'),
        analyzeBtn: document.getElementById('analyzeBtn'),
        progressContainer: document.getElementById('progressContainer')
    };
    
    switch (state) {
        case 'initial':
            // Estado inicial - sin archivo
            if (elements.dropZone) elements.dropZone.style.display = 'block';
            if (elements.quickPredictBtn) elements.quickPredictBtn.disabled = true;
            if (elements.analyzeBtn) elements.analyzeBtn.disabled = true;
            hideProgress();
            break;
            
        case 'file_selected':
            // Archivo seleccionado
            if (elements.quickPredictBtn) {
                elements.quickPredictBtn.disabled = false;
                elements.quickPredictBtn.innerHTML = '<i class="fas fa-zap me-2"></i>Análisis Rápido';
            }
            if (elements.analyzeBtn) {
                elements.analyzeBtn.disabled = false;
                elements.analyzeBtn.innerHTML = '<i class="fas fa-microscope me-2"></i>Analizar Imagen';
            }
            break;
            
        case 'processing':
            // Procesando
            if (elements.quickPredictBtn) {
                elements.quickPredictBtn.disabled = true;
                elements.quickPredictBtn.innerHTML = '<span class="loading-spinner me-2"></span>Procesando...';
            }
            if (elements.analyzeBtn) {
                elements.analyzeBtn.disabled = true;
                elements.analyzeBtn.innerHTML = '<span class="loading-spinner me-2"></span>Analizando...';
            }
            showProgress();
            break;
            
        case 'completed':
            // Completado
            if (elements.quickPredictBtn) {
                elements.quickPredictBtn.disabled = false;
                elements.quickPredictBtn.innerHTML = '<i class="fas fa-check me-2"></i>¡Completado!';
            }
            if (elements.analyzeBtn) {
                elements.analyzeBtn.disabled = false;
                elements.analyzeBtn.innerHTML = '<i class="fas fa-check me-2"></i>Análisis Completo';
            }
            hideProgress();
            break;
            
        case 'error':
            // Error
            if (elements.quickPredictBtn) {
                elements.quickPredictBtn.disabled = false;
                elements.quickPredictBtn.innerHTML = '<i class="fas fa-redo me-2"></i>Reintentar';
            }
            if (elements.analyzeBtn) {
                elements.analyzeBtn.disabled = false;
                elements.analyzeBtn.innerHTML = '<i class="fas fa-redo me-2"></i>Reintentar';
            }
            hideProgress();
            break;
    }
}

// ====================================
// PROGRESS MANAGEMENT
// ====================================
function showProgress() {
    const container = document.getElementById('progressContainer');
    if (container) {
        container.classList.remove('d-none');
    }
    
    // Iniciar progreso simulado
    startProgressSimulation();
}

function hideProgress() {
    const container = document.getElementById('progressContainer');
    if (container) {
        container.classList.add('d-none');
    }
    
    // Detener simulación
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

function updateProgress(percentage, message = '') {
    const progressBar = document.querySelector('#progressContainer .progress-bar');
    const progressText = document.getElementById('progressText');
    
    if (progressBar) {
        progressBar.style.width = percentage + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
    }
    
    if (progressText && message) {
        progressText.textContent = message;
    }
}

function startProgressSimulation() {
    let progress = 0;
    const messages = [
        'Preparando imagen...',
        'Cargando modelo de IA...',
        'Procesando características...',
        'Analizando patrones...',
        'Generando predicción...',
        'Finalizando análisis...'
    ];
    
    progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        
        if (progress >= 95) {
            progress = 95; // Dejar espacio para el 100% real
            clearInterval(progressInterval);
        }
        
        const messageIndex = Math.floor((progress / 100) * messages.length);
        const message = messages[Math.min(messageIndex, messages.length - 1)];
        
        updateProgress(Math.round(progress), message);
    }, 800);
}

// ====================================
// PROGRESS CIRCLE ANIMATIONS
// ====================================
function updateProgressCircle(selector, percentage) {
    const circles = document.querySelectorAll(selector);
    
    circles.forEach(circle => {
        const degrees = (percentage / 100) * 360;
        let color;
        
        // Determinar color basado en porcentaje
        if (percentage >= 80) color = 'var(--success-color)';
        else if (percentage >= 60) color = 'var(--warning-color)';
        else if (percentage >= 40) color = 'var(--info-color)';
        else color = 'var(--danger-color)';
        
        // Aplicar gradiente cónico animado
        circle.style.background = `conic-gradient(${color} ${degrees}deg, #e9ecef ${degrees}deg)`;
        
        // Actualizar texto
        const textElement = circle.querySelector('.progress-text');
        if (textElement) {
            textElement.textContent = percentage + '%';
        }
    });
}

// ====================================
// ALERT SYSTEM
// ====================================
function showAlert(type, title, message, duration = 5000) {
    const alertContainer = document.getElementById('alertContainer') || createAlertContainer();
    
    const alertHTML = `
        <div class="alert alert-${getAlertClass(type)} alert-dismissible fade show" role="alert">
            <i class="fas ${getAlertIcon(type)} me-2"></i>
            <strong>${title}:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    alertContainer.insertAdjacentHTML('beforeend', alertHTML);
    
    // Auto-dismiss después del tiempo especificado
    if (duration > 0) {
        setTimeout(() => {
            const alerts = alertContainer.querySelectorAll('.alert');
            if (alerts.length > 0) {
                const latestAlert = alerts[alerts.length - 1];
                const bsAlert = new bootstrap.Alert(latestAlert);
                bsAlert.close();
            }
        }, duration);
    }
}

function createAlertContainer() {
    let container = document.getElementById('alertContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'alertContainer';
        container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 1050; max-width: 400px;';
        document.body.appendChild(container);
    }
    return container;
}

function getAlertClass(type) {
    const classes = {
        'success': 'success',
        'error': 'danger',
        'warning': 'warning',
        'info': 'info'
    };
    return classes[type] || 'info';
}

function getAlertIcon(type) {
    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    };
    return icons[type] || 'fa-info-circle';
}

// ====================================
// MEDICAL DISCLAIMER
// ====================================
function showMedicalDisclaimer() {
    // Solo mostrar si no se ha mostrado antes en esta sesión
    if (sessionStorage.getItem('disclaimerShown')) {
        return;
    }
    
    const disclaimerModal = document.getElementById('disclaimerModal');
    if (disclaimerModal) {
        const modal = new bootstrap.Modal(disclaimerModal);
        modal.show();
        
        // Marcar como mostrado
        sessionStorage.setItem('disclaimerShown', 'true');
    }
}

// ====================================
// UTILITY FUNCTIONS
// ====================================
function initializeTooltips() {
    // Inicializar tooltips de Bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function setupSmoothScrolling() {
    // Smooth scrolling para enlaces internos
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
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

function resetForm() {
    // Resetear formulario principal
    const form = document.getElementById('uploadForm');
    if (form) {
        form.reset();
    }
    
    // Limpiar variables
    uploadedFile = null;
    
    // Ocultar elementos
    const elementsToHide = ['imagePreview', 'predictionResults', 'fileInfo'];
    elementsToHide.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.classList.add('d-none');
        }
    });
    
    // Resetear UI
    updateUIState('initial');
}

// ====================================
// KEYBOARD SHORTCUTS
// ====================================
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + U para upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        const fileInput = document.getElementById('imageFile');
        if (fileInput) fileInput.click();
    }
    
    // Enter para analizar (si hay imagen)
    if (e.key === 'Enter' && uploadedFile && !isProcessing) {
        e.preventDefault();
        startPrediction();
    }
    
    // Escape para resetear
    if (e.key === 'Escape') {
        resetForm();
    }
});

// ====================================
// EXPORT FUNCTIONS (para uso global)
// ====================================
window.SkinDetector = {
    startPrediction,
    resetForm,
    showAlert,
    updateProgressCircle,
    showMedicalDisclaimer
};

console.log('🎯 Skin Disease Detector JavaScript cargado correctamente');
