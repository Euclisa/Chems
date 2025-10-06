let chemsData = [];
let edgesData = [];
let reactionsData = [];
let fuse = null;
let currentResults = [];
let currentPage = 0;
const RESULTS_PER_PAGE = 20;
let selectedCIDs = new Set();
let cidToCompound = new Map();
let originalMainContent = '';
let hoverTimeout;
let selectedLevel = 1;
let graph = new Map();
let graph_reverse = new Map()
let edgeToReactionID = new Map();
let RIDToReaction = new Map();
let currentEdges = new Set();
let sourceNodes = new Set();
let targetNodes = new Set();
let secondaryNodes = new Set();
let directedEdges = new Set();
let backgroundCids;
let cidToDescription = new Map();
let ridToDescription = new Map();
let commonnesSortedCids = []
let cidToEdges = new Map();

async function loadData(fileName) {
    try {
        const response = await fetch(fileName, { cache: "no-store" });
        const text = await response.text();

        let data = [];
        if (fileName.endsWith('.jsonl')) {
            data = text.split('\n').filter(line => line.trim() !== '');
            data = data.map(line => JSON.parse(line));
        }
        else if (fileName.endsWith('.json')) {
            data = JSON.parse(text);
        }

        return data;
    } catch(error) {
        console.error(`Error during fetching: ${fileName}; ${error}`);
        return [];
    }
}

async function initializeData() {
    showLoading('Loading data...');
    const resultsContainer = document.getElementById('results-container');
    originalMainContent = document.getElementById('main').innerHTML;
    
    try {
        chemsData = await loadData('data/chems/chems.jsonl');
        edgesData = await loadData('data/chems/chems_edges.jsonl');
        reactionsData = await loadData('data/reactions_parsed/reactions_parsed_balanced.jsonl');
        backgroundCids = await loadData('data/misc/background_cids.json');
        backgroundCids = new Set(backgroundCids.map(Number));
        commonnesSortedCids = await loadData('data/misc/commonness_sorted_cids.json');
        commonnesSortedCids = commonnesSortedCids.map(Number);
        
        let chemsDescriptionsLoaded = await loadData('data/chems/chems_descriptions.jsonl');
        chemsDescriptionsLoaded.forEach(entry => cidToDescription.set(entry.cid, entry.description));

        let reactionsDescriptionsLoaded = await loadData('data/reactions_details/reactions_details.jsonl');
        reactionsDescriptionsLoaded.forEach(entry => ridToDescription.set(entry.rid, entry.description));
        
        if (chemsData.length === 0) {
            resultsContainer.innerHTML = '<div class="no-results">No compound data available</div>';
            hideLoading();
            return;
        }
        
        edgesData.forEach(entry => {
            const edgeStr = `${entry.first}-${entry.second}`;
            edgeToReactionID.set(edgeStr, entry.reactions);
        });
        
        reactionsData.forEach(entry => {
            const rid = entry.rid;
            const copy = { ...entry };
            delete copy.rid;
            RIDToReaction.set(rid, copy);
        });

        chemsData.forEach(comp => {
            cidToCompound.set(comp.cid, comp);
        });

        edgesData.forEach(edge => {
            if (!graph.has(edge.first)) graph.set(edge.first, []);
            graph.get(edge.first).push(edge.second);

            if (!graph_reverse.has(edge.second)) graph_reverse.set(edge.second, []);
            graph_reverse.get(edge.second).push(edge.first);
        });

        for (const cid of commonnesSortedCids) cidToEdges.set(cid, []);
        edgesData.forEach(edge => {
            cidToEdges.get(edge.first).push(edge);
            cidToEdges.get(edge.second).push(edge);
        });

        const fuseOptions = {
            keys: ['cmpdname', 'synonyms'],
            threshold: 0.3,
            includeScore: true,
            minMatchCharLength: 2
        };
        
        fuse = new Fuse(chemsData, fuseOptions);
        
        chemsData = commonnesSortedCids.map(cid => cidToCompound.get(cid));

        displayResults(chemsData.slice(0, RESULTS_PER_PAGE));
        currentResults = chemsData;
        currentPage = 0;
        
        if (chemsData.length > RESULTS_PER_PAGE) {
            addLoadMoreButton();
        }
        
    } catch (error) {
        console.error('Error initializing data:', error);
        resultsContainer.innerHTML = '<div class="no-results">Error loading compound data</div>';
    }
    hideLoading();

    renderPathList();

    document.getElementById('level1').addEventListener('click', () => selectLevel(1));
    document.getElementById('level2').addEventListener('click', () => selectLevel(2));

    const kSlider = document.getElementById('k-slider');
    const kValue = document.getElementById('k-value');
    kSlider.addEventListener('input', () => {
        kValue.textContent = kSlider.value === '11' ? 'All' : kSlider.value;
    });

    const nSlider = document.getElementById('n-slider');
    const nValue = document.getElementById('n-value');
    nSlider.addEventListener('input', () => {
        nValue.textContent = nSlider.value;
    });
}