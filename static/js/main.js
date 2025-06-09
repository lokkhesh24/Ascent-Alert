document.addEventListener('DOMContentLoaded', function() {
    
    // --- Theme Toggler ---
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeIconMoon = document.getElementById('themeIconMoon');
    const themeIconSun = document.getElementById('themeIconSun');
    // Default to system preference if no explicit choice, else light
    let preferredTheme = localStorage.getItem('theme');
    if (!preferredTheme) {
        preferredTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark-mode' : 'light-mode';
    }
    
    function applyTheme(theme) {
        document.body.classList.remove('light-mode', 'dark-mode');
        document.body.classList.add(theme);
        if (themeIconMoon && themeIconSun) { // Ensure elements exist
            if (theme === 'dark-mode') {
                themeIconMoon.style.display = 'none';
                themeIconSun.style.display = 'inline';
            } else {
                themeIconMoon.style.display = 'inline';
                themeIconSun.style.display = 'none';
            }
        }
        // Update chart colors if Chart.js is used and charts are present
        // This requires charts to be re-rendered or updated with new options.
        // For now, CSS variables should handle most of it if charts are styled with them.
    }

    applyTheme(preferredTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', function() {
            let newTheme = document.body.classList.contains('light-mode') ? 'dark-mode' : 'light-mode';
            localStorage.setItem('theme', newTheme);
            applyTheme(newTheme);
        });
    }

    // --- Mobile Navigation Hamburger ---
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
            // Toggle body scroll
            document.body.style.overflow = navMenu.classList.contains('active') ? 'hidden' : '';
        });

        document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
            if (navMenu.classList.contains('active')) {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
                document.body.style.overflow = '';
            }
        }));
    }

    // --- Auto-dismiss flash messages (excluding error and prediction success) ---
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(function(message) {
        // Check if it's an error or a specific success message we want to keep longer or handle differently
        const isError = message.classList.contains('flash-error');
        // Prediction success is now handled by modal, so flash-success_prediction might not be used.
        // const isPredictionSuccess = message.classList.contains('flash-success_prediction');

        if (!isError) { // Auto-dismiss non-error messages
            setTimeout(function() {
                let opacity = 1;
                const fadeInterval = setInterval(function() {
                    if (opacity <= 0.1) {
                        clearInterval(fadeInterval);
                        if (message.parentNode) { // Check if still in DOM
                           message.style.display = 'none'; // Hide it first
                           message.remove(); // Then remove from DOM
                        }
                    } else {
                        message.style.opacity = opacity;
                        opacity -= opacity * 0.1;
                    }
                }, 30); // Faster fade
            }, 4000); // 4 seconds delay
        }
    });
    
    // --- Intersection Observer for Animations (subtle fade-in/up) ---
    const animatedElements = document.querySelectorAll('.flexible-card, .hero-content, .hero-visual, .pillar-card, .feature-card, .team-member-card, .value-item, .ghat-card');
    if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries, observerInstance) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                    observerInstance.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        animatedElements.forEach(el => {
            el.classList.add('animatable'); // Initial state class
            observer.observe(el);
        });
    } else { // Fallback for older browsers
        animatedElements.forEach(el => el.classList.add('animate-in'));
    }

    // --- Dashboard PDF Download ---
    const downloadPdfBtn = document.getElementById('downloadDashboardPdf');
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener('click', function() {
            const dashboardElement = document.querySelector('.dashboard-grid'); // Or a more specific container
            if (dashboardElement && window.html2canvas && window.jspdf) {
                const { jsPDF } = window.jspdf;
                
                // Show a temporary loading/processing message
                const originalButtonText = downloadPdfBtn.innerHTML;
                downloadPdfBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
                downloadPdfBtn.disabled = true;

                html2canvas(dashboardElement, { 
                    scale: 2, // Increase scale for better quality
                    useCORS: true, // If you have external images/resources
                    backgroundColor: getComputedStyle(document.body).getPropertyValue('--card-background-color').trim() // Use card bg for canvas
                }).then(canvas => {
                    const imgData = canvas.toDataURL('image/png');
                    const pdf = new jsPDF({
                        orientation: 'landscape',
                        unit: 'pt', // points
                        format: 'a4'
                    });

                    const pdfWidth = pdf.internal.pageSize.getWidth();
                    const pdfHeight = pdf.internal.pageSize.getHeight();
                    const canvasWidth = canvas.width;
                    const canvasHeight = canvas.height;
                    
                    // Calculate the aspect ratio
                    const ratio = canvasWidth / canvasHeight;
                    let newCanvasWidth = pdfWidth;
                    let newCanvasHeight = newCanvasWidth / ratio;

                    // If new height is still too large, scale by height instead
                    if (newCanvasHeight > pdfHeight) {
                        newCanvasHeight = pdfHeight;
                        newCanvasWidth = newCanvasHeight * ratio;
                    }
                    
                    // Center the image on the PDF page
                    const x = (pdfWidth - newCanvasWidth) / 2;
                    const y = (pdfHeight - newCanvasHeight) / 2;

                    pdf.addImage(imgData, 'PNG', x, y, newCanvasWidth, newCanvasHeight);
                    pdf.save('AscentAlert_Dashboard.pdf');
                }).catch(err => {
                    console.error("Error generating PDF: ", err);
                    alert("Sorry, an error occurred while generating the PDF.");
                }).finally(() => {
                    // Restore button
                    downloadPdfBtn.innerHTML = originalButtonText;
                    downloadPdfBtn.disabled = false;
                });
            } else {
                alert("PDF generation library not loaded or dashboard element not found.");
            }
        });
    }

});
