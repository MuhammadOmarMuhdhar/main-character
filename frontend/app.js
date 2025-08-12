// Initialize Preline UI components
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all Preline components
    if (window.HSStaticMethods) {
        window.HSStaticMethods.autoInit();
    }
    
    // Force re-initialize dropdowns with hover trigger
    setTimeout(() => {
        if (window.HSDropdown) {
            window.HSDropdown.autoInit();
        }
    }, 100);
    
    console.log('Preline UI initialized successfully');
});