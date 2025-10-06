function selectLevel(level) {
    selectedLevel = level;
    document.getElementById('level1').classList.toggle('selected', level === 1);
    document.getElementById('level2').classList.toggle('selected', level === 2);
}

function renderPathList() {
    const level1Items = document.querySelector('#level1 .level-items');
    const level2Items = document.querySelector('#level2 .level-items');
    level1Items.innerHTML = '';
    level2Items.innerHTML = '';

    sourceNodes.forEach(cid => {
        const comp = cidToCompound.get(cid);
        if (comp) {
            const item = createPathItem(comp);
            level1Items.appendChild(item);
        }
    });

    targetNodes.forEach(cid => {
        const comp = cidToCompound.get(cid);
        if (comp) {
            const item = createPathItem(comp);
            level2Items.appendChild(item);
        }
    });
}

function createItemInfo(compound) {
    const itemInfo = document.createElement('div');
    itemInfo.className = 'item-info';
    
    const itemName = document.createElement('div');
    itemName.className = 'item-name';
    itemName.textContent = compound.cmpdname || 'Unknown compound';
    
    const itemExtra = document.createElement('div');
    itemExtra.className = 'item-extra';

    const cidLink = document.createElement('a');
    cidLink.href = `https://pubchem.ncbi.nlm.nih.gov/compound/${compound.cid}`;
    cidLink.textContent = `${compound.cid}`;
    cidLink.target = '_blank';
    const cidPrefix = document.createTextNode('CID: ');
    itemExtra.appendChild(cidPrefix);
    itemExtra.appendChild(cidLink);

    if (compound.wiki) {
        const wikiLink = document.createElement('a');
        wikiLink.href = compound.wiki;
        wikiLink.textContent = `${compound.wiki.split('/').at(-1).replace('_', ' ')}`;
        wikiLink.target = '_blank';
        
        const sep_wiki_prefix = document.createTextNode(' | Wiki: ');
        itemExtra.appendChild(sep_wiki_prefix);
        itemExtra.appendChild(wikiLink);
    }
    
    itemInfo.appendChild(itemName);
    itemInfo.appendChild(itemExtra);

    return itemInfo;
}

function createPathItem(compound) {
    const item = document.createElement('div');
    item.className = 'result-item';
    item.dataset.cid = compound.cid;

    const iconContainer = document.createElement('div');
    iconContainer.className = 'item-icon loading';
    iconContainer.textContent = 'Loading...';
    loadStructureSVG(compound.cid, iconContainer);

    const itemInfo = createItemInfo(compound);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-item';
    removeBtn.textContent = 'X';
    removeBtn.addEventListener('click', () => {
        if (selectedLevel === 1) {
            sourceNodes.delete(compound.cid);
        } else {
            targetNodes.delete(compound.cid);
        }
        renderPathList();
    });

    item.appendChild(iconContainer);
    item.appendChild(itemInfo);
    //item.appendChild(removeBtn);

    item.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (selectedLevel === 1) {
            sourceNodes.delete(compound.cid);
        } else {
            targetNodes.delete(compound.cid);
        }
        renderPathList();
    });

    return item;
}


function createResultItem(compound) {
    const item = document.createElement('div');
    item.className = 'result-item';
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'item-check';
    checkbox.checked = selectedCIDs.has(compound.cid);
    checkbox.addEventListener('change', function() {
        if (this.checked) {
            selectedCIDs.add(compound.cid);
            // Updates graph adding edges attached to this node
            updateGraph(compound.cid);
        } else {
            // Removes node with attached edges
            removeNode(compound.cid);
            // Updates graph without updating edges
            updateGraph();
        }
    });
    
    const iconContainer = document.createElement('div');
    iconContainer.className = 'item-icon loading';
    iconContainer.textContent = 'Loading...';
    iconContainer.dataset.cid = `${compound.cid}`;
    iconContainer.addEventListener('click', (e) => {
        const target = event.currentTarget;
        const cid = Number(target.dataset.cid);
        showCompoundInfoPopup(cid);
    });
    
    loadStructureSVG(compound.cid, iconContainer);
    
    const itemInfo = createItemInfo(compound);
    
    item.appendChild(checkbox);
    item.appendChild(iconContainer);
    item.appendChild(itemInfo);

    item.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (selectedLevel === 1) {
            sourceNodes.add(compound.cid);
        } else {
            targetNodes.add(compound.cid);
        }
        renderPathList();
    });
    
    return item;
}


function displayResults(results) {
    const resultsContainer = document.getElementById('results-container');
    
    const existingBtn = resultsContainer.querySelector('.load-more-btn');
    if (existingBtn) {
        existingBtn.remove();
    }
    
    if (results.length === 0) {
        resultsContainer.innerHTML = '<div class="no-results">No compounds found</div>';
        return;
    }
    
    resultsContainer.innerHTML = '';
    results.forEach(compound => {
        const resultData = compound.item || compound;
        const resultItem = createResultItem(resultData);
        resultsContainer.appendChild(resultItem);
    });
}

function addLoadMoreButton() {
    const resultsContainer = document.getElementById('results-container');
    const loadMoreBtn = document.createElement('button');
    loadMoreBtn.className = 'load-more-btn';
    loadMoreBtn.textContent = 'Load More Results';
    loadMoreBtn.onclick = loadMoreResults;
    resultsContainer.appendChild(loadMoreBtn);
}

function loadMoreResults() {
    const endIndex = (currentPage + 1) * RESULTS_PER_PAGE + RESULTS_PER_PAGE;
    const nextResults = currentResults.slice(0, endIndex);
    
    if (nextResults.length > 0) {
        displayResults(nextResults);
        currentPage++;
        
        if (endIndex < currentResults.length) {
            addLoadMoreButton();
        }
    }
}

function refreshResults() {
    const endIndex = (currentPage + 1) * RESULTS_PER_PAGE + RESULTS_PER_PAGE;
    const nextResults = currentResults.slice(0, endIndex);
    
    if (nextResults.length > 0) {
        displayResults(nextResults);
        
        if (endIndex < currentResults.length) {
            addLoadMoreButton();
        }
    }
}

function performSearch(query) {
    const resultsContainer = document.getElementById('results-container');
    
    if (!query.trim()) {
        currentResults = chemsData;
        currentPage = 0;
        displayResults(chemsData.slice(0, RESULTS_PER_PAGE));
        
        if (chemsData.length > RESULTS_PER_PAGE) {
            addLoadMoreButton();
        }
        return;
    }
    
    if (!fuse) {
        resultsContainer.innerHTML = '<div class="no-results">Search not available</div>';
        return;
    }
    
    const searchResults = fuse.search(query);
    currentResults = searchResults;
    currentPage = 0;
    
    const firstPageResults = searchResults.slice(0, RESULTS_PER_PAGE);
    displayResults(firstPageResults);
    
    if (searchResults.length > RESULTS_PER_PAGE) {
        addLoadMoreButton();
    }
}