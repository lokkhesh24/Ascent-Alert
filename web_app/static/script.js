document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictor-form');
    const riskCanvas = document.getElementById('riskCanvas');
    const pred = document.getElementById('prediction');

    if (form && riskCanvas) {
        const ctx = riskCanvas.getContext('2d');
        let probability = pred ? parseFloat(pred.textContent.match(/\d+\.\d+/)[0]) / 100 : 0;

        // Risk Meter Animation
        let angle = 0;
        const animateMeter = () => {
            ctx.clearRect(0, 0, 200, 200);
            // Draw background circle
            ctx.beginPath();
            ctx.arc(100, 100, 80, 0, Math.PI * 2);
            ctx.lineWidth = 10;
            ctx.strokeStyle = '#ccc';
            ctx.stroke();

            // Draw risk arc
            ctx.beginPath();
            ctx.arc(100, 100, 80, -Math.PI / 2, -Math.PI / 2 + angle);
            ctx.strokeStyle = probability > 0.7 ? '#F44336' : probability > 0.3 ? '#FFC107' : '#4CAF50';
            ctx.stroke();

            // Animate
            angle += 0.05;
            if (angle < (probability * 2 * Math.PI)) {
                requestAnimationFrame(animateMeter);
            }
        };

        if (probability > 0) {
            animateMeter();
        }
    }

    if (pred) {
        if (pred.textContent.includes('High')) pred.classList.add('high');
        else if (pred.textContent.includes('Medium')) pred.classList.add('medium');
        else if (pred.textContent.includes('Low')) pred.classList.add('low');
    }
});