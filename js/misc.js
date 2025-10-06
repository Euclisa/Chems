

function returnToBlank() {
    selectedCIDs.clear();
    directedEdges.clear();
    document.getElementById('main').innerHTML = originalMainContent;
    refreshResults();
}


function removeNode(cid, force = true) {
    let edges_to_del = []
    let secondary_parents = []
    let has_children = false;
    for(const edge of directedEdges) {
        let [a, b] = edge.split('-').map(Number);
        if(a === cid) {
            edges_to_del.push(edge);
            has_children = true;
        }
        else if(b === cid) {
            edges_to_del.push(edge);
            if(secondaryNodes.has(a))
                secondary_parents.push(a);
        }
    }

    if(force || !has_children) {
        selectedCIDs.delete(cid);
        secondaryNodes.delete(cid);
        sourceNodes.delete(cid);
        targetNodes.delete(cid);

        for(const edge of edges_to_del)
            directedEdges.delete(edge);
        for(const cid of secondary_parents)
            removeNode(cid, false);
    }
}


function loadStructureSVG(cid, container) {
    const svgPath = `data/structures/${cid}.svg`;
    
    fetch(svgPath)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.text();
        })
        .then(svgContent => {
            container.innerHTML = svgContent;
            container.classList.remove('loading');
        })
        .catch(error => {
            container.innerHTML = '<i class="fa fa-flask" style="color: #ccc;"></i>';
            container.classList.remove('loading');
        });
}


function showLoading(message, showCancel = false) {
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-message').textContent = message;
    document.getElementById('cancel-compute').style.display = showCancel ? 'block' : 'none';
    overlay.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}