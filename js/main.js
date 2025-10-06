(function(){
    const app = document.getElementById('app');
    const toggle = document.getElementById('toggle');
    const sidePanel = document.getElementById('sidePanel');

    let isOpen = false;

    function setOpen(open){
        if(open){
            app.classList.add('open');
            toggle.setAttribute('aria-expanded','true');
            sidePanel.setAttribute('aria-hidden','false');
        } else {
            app.classList.remove('open');
            toggle.setAttribute('aria-expanded','false');
            sidePanel.setAttribute('aria-hidden','true');
        }
    }

    setOpen(isOpen);

    toggle.addEventListener('click', function(){
        isOpen = !isOpen;
        setOpen(isOpen);
    });

    document.addEventListener('keydown', function(e){
        if(e.key === 'Escape' && isOpen){
            isOpen = false;
            setOpen(isOpen);
            toggle.focus();
        }
    });
})();

(function(){
    const submitButton = document.getElementById('submit-button');
    const searchInput = document.getElementById('search-input');
    const cancelCompute = document.getElementById('cancel-compute');
    
    let searchTimeout;

    function handleSearch() {
        const query = searchInput.value;
        performSearch(query);
    }

    submitButton.addEventListener('click', handleSubmit);

    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            handleSearch();
        }, 300);
    });

    cancelCompute.addEventListener('click', hideLoading);
})();

document.addEventListener('DOMContentLoaded', initializeData);