function bitCount(n) {
    n = n - ((n >> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
    return ((n + (n >> 4) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function computeTanimoto(fp1, fp2) {
    let andPop = 0;
    for (let i = 0; i < 32; i++) {
        const and = fp1.bits[i] & fp2.bits[i];
        andPop += bitCount(and);
    }
    const orPop = fp1.popcount + fp2.popcount - andPop;
    return orPop === 0 ? 1 : andPop / orPop;
}

function minH(node, ends) {
    let maxSim = 0;
    for (let end of ends) {
        const sim = computeTanimoto(cidToCompound.get(node).ECFP4_fp, cidToCompound.get(end).ECFP4_fp);
        if (sim > maxSim) maxSim = sim;
    }
    return 1 - maxSim;
}

class PriorityQueue {
    constructor(comparator = (a, b) => a - b) {
        this._heap = [];
        this._comparator = comparator;
    }
    size() {
        return this._heap.length;
    }
    peek() {
        return this._heap[0];
    }
    push(value) {
        this._heap.push(value);
        this._siftUp();
        return this.size();
    }
    pop() {
        const poppedValue = this.peek();
        const bottom = this.size() - 1;
        if (bottom > 0) {
            this._swap(0, bottom);
        }
        this._heap.pop();
        this._siftDown();
        return poppedValue;
    }
    _greater(i, j) {
        return this._comparator(this._heap[i], this._heap[j]) < 0;
    }
    _swap(i, j) {
        [this._heap[i], this._heap[j]] = [this._heap[j], this._heap[i]];
    }
    _siftUp() {
        let node = this.size() - 1;
        while (node > 0 && this._greater(node, Math.floor((node - 1) / 2))) {
            this._swap(node, Math.floor((node - 1) / 2));
            node = Math.floor((node - 1) / 2);
        }
    }
    _siftDown() {
        let node = 0;
        while (
            (2 * node + 1 < this.size() && this._greater(2 * node + 1, node)) ||
            (2 * node + 2 < this.size() && this._greater(2 * node + 2, node))
        ) {
            let maxChild = (2 * node + 2 < this.size() && this._greater(2 * node + 2, 2 * node + 1)) ? 2 * node + 2 : 2 * node + 1;
            this._swap(node, maxChild);
            node = maxChild;
        }
    }
}

function findKShortestPathsWithSources(sources, targets, k, maxLen) {
    // Finds paths from targets to sources using reversed graph and then reverses paths

    const sourceSet = new Set(sources);
    const pq = new PriorityQueue((a, b) => a.f - b.f);
    const paths = [];

    for (let target of targets) {
        const h = minH(target, sources);
        pq.push({f: 0 + h, cost: 0, node: target, path: [target]});
    }

    while (pq.size() > 0 && paths.length < k) {
        const current = pq.pop();
        const {cost, node, path} = current;

        if (cost > maxLen) continue;

        if (sourceSet.has(node) && cost > 0) {
            paths.push({path, length: cost});
            continue;
        }

        for (let neigh of graph_reverse.get(node) || []) {
            if (path.includes(neigh)) continue;
            if (backgroundCids.has(neigh) && !sourceSet.has(neigh)) continue;
            const newCost = cost + 1;
            const h = minH(neigh, sources);
            const newF = newCost + h;
            const newPath = [...path, neigh];
            pq.push({f: newF, cost: newCost, node: neigh, path: newPath});
        }
    }

    paths.forEach(path => { path.path.reverse() })

    return paths;
}

function findKShortestPathsNoSources(targets, k, maxLen) {
    // Finds paths from targets to compounds with complexity <= complexityThr using complexity difference as distance
    
    const queue = []
    const paths = [];

    const complexityThr = 10
    
    const hMetrics = node => {
        const nodeComplexity = cidToCompound.get(node).complexity;
        return nodeComplexity > complexityThr ? nodeComplexity - complexityThr : 0;
    }

    for (let target of targets) {
        const h = hMetrics(target);
        queue.push({cost: 0, node: target, path: [target]});
    }

    while (queue.length > 0 && paths.length < k) {
        const current = queue.shift();
        const {cost, node, path} = current;
        const currentH = hMetrics(node);

        if (cost > maxLen) continue;
        
        if (cost > 0)
            paths.push({path, length: cost});
        if (paths.length > k)
            paths.shift();

        for (let neigh of graph_reverse.get(node) || []) {
            const h = hMetrics(neigh);
            if (h > currentH || path.includes(neigh) || backgroundCids.has(neigh)) continue;
            const newCost = cost + 1;
            const newPath = [...path, neigh];
            queue.push({cost: newCost, node: neigh, path: newPath});
        }
    }

    paths.forEach(path => { path.path.reverse() });

    return paths;
}

function getProcessedLinks() {
    const links = [];
    const processed = new Set();
    for (let edge of directedEdges) {
        if (processed.has(edge)) continue;
        const [a, b] = edge.split('-').map(Number);

        const reverse = `${b}-${a}`;
        let type = null;
        if (directedEdges.has(reverse)) {
            type = 'bi';
            processed.add(reverse);
        } else {
            type = 'directed';
        }

        const secondary = secondaryNodes.has(a) || secondaryNodes.has(b);
        links.push({source: a, target: b, type, secondary});
        processed.add(edge);
    }
    return links;
}

function handleSubmit() {
    secondaryNodes = new Set();
    directedEdges = new Set();

    const sources = Array.from(sourceNodes);
    const targets = Array.from(targetNodes);
    if (targets.length === 0) {
        if (sources.length !== 0) {
            alert('Please add at least one target substance.');
            return;
        }
        returnToBlank();
        return;
    }

    const sourcesSelected = sources.length > 0;

    let k = parseInt(document.getElementById('k-slider').value);
    if (k === 11) k = Infinity;
    const n = parseInt(document.getElementById('n-slider').value);

    showLoading('Computing paths...', true);

    setTimeout(() => {
        const allPaths = sourcesSelected ? findKShortestPathsWithSources(sources, targets, k, n) : findKShortestPathsNoSources(targets, k, n);
        selectedCIDs.clear();

        allPaths.forEach(({path}) => path.forEach(node => selectedCIDs.add(node)));
        allPaths.forEach(({path}) => {
            for (let i = 0; i < path.length - 1; i++) {
                const edgeStr = `${path[i]}-${path[i+1]}`;
                const reactionIDs = edgeToReactionID.get(edgeStr);
                for(const rid of reactionIDs) {
                    const reaction = RIDToReaction.get(rid);
                    for(const reagent of reaction.reagents) {
                        const cid = reagent.cid;
                        directedEdges.add(`${cid}-${path[i+1]}`);
                        if(!selectedCIDs.has(cid)) {
                            if(sourcesSelected)
                                secondaryNodes.add(cid);
                            selectedCIDs.add(cid);
                        }
                    }
                }
            }
        });
        
        updateGraph();
        hideLoading();
    }, 0);
}

function updateGraph(cid = null) {
    const selectedArray = Array.from(selectedCIDs);
    if (selectedArray.length === 0) {
        document.getElementById('main').innerHTML = originalMainContent;
        return;
    }

    if(cid !== null) {
        selectedCIDs.forEach(selected_cid => {
            cidToEdges.get(selected_cid)
                .filter(edge => 
                    selectedCIDs.has(edge.first) && selectedCIDs.has(edge.second) &&
                    (edge.first === cid || edge.second === cid) &&
                    !secondaryNodes.has(edge.first) && !secondaryNodes.has(edge.second))
                .forEach(edge => directedEdges.add(`${edge.first}-${edge.second}`));
        });
    }

    const nodes = selectedArray.map(node => {
        const comp = cidToCompound.get(node);
        return {
            id: node,
            name: comp ? comp.cmpdname : 'Unknown',
            organic: comp ? comp.organic : false,
            secondary: secondaryNodes.has(node)
        };
    });

    const links = getProcessedLinks();

    refreshResults();
    renderGraph(nodes, links);
}

function renderGraph(nodes, links) {
    const main = d3.select("#main");
    main.html("");

    const width = main.node().clientWidth;
    const height = main.node().clientHeight;

    const svg = main.append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .style("max-width", "100%")
        .style("height", "auto");

    const defs = svg.append("defs");

    defs.append("marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 -3 10 6")
        .attr("refX", "10")
        .attr("refY", "0")
        .attr("markerWidth", "6")
        .attr("markerHeight", "6")
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-3L10,0L0,3")
        .attr("fill", "#b8a8d9");

    defs.append("marker")
        .attr("id", "arrow-start")
        .attr("viewBox", "0 -3 10 6")
        .attr("refX", "10")
        .attr("refY", "0")
        .attr("markerWidth", "6")
        .attr("markerHeight", "6")
        .attr("orient", "auto-start-reverse")
        .append("path")
        .attr("d", "M0,-3L10,0L0,3")
        .attr("fill", "#b8a8d9");

    const g = svg.append("g");

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = g.append("g")
        .attr("stroke", "#b8a8d9")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke-width", 2)
        .attr("marker-end", d => "url(#arrow)")
        .attr("marker-start", d => d.type === 'bi' ? "url(#arrow-start)" : null)
        .attr("stroke-opacity", d => d.secondary ? 0.3 : 0.7)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            d3.select(this).attr("stroke-width", 4).attr("stroke", "#9d89c7");
        })
        .on("mouseout", function(event, d) {
            d3.select(this).attr("stroke-width", 2).attr("stroke", "#b8a8d9");
        })
        .on("click", function(event, d) {
            showPopup('edge', d);
        });
    
    const secondaryOpacity = 0.3;
    const defaultOpacity = 1.0;

    const nodeGroups = g.append("g")
        .selectAll("g")
        .data(nodes)
        .join("g")
        .attr("opacity", d => secondaryNodes.has(d.id) ? secondaryOpacity : defaultOpacity)
        .style("cursor", "pointer")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
        )
        .on("mouseover", function(event, d) {
            d3
            .select(this)
            .attr('opacity', defaultOpacity)
            .select("circle, rect, polygon")
            .attr("r", 8).attr("width", 16)
            .attr("height", 16)
            .attr("points", "0,-8 8,8 -8,8");
            hoverTimeout = setTimeout(() => {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`<img src="data/structures/${d.id}.svg" alt="${d.name}"><div class="tooltip-name">${d.name}</div>`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }, 500);
        })
        .on("mousemove", function(event, d) {
            tooltip.style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(event, d) {
            d3
            .select(this)
            .attr('opacity', d => secondaryNodes.has(d.id) ? secondaryOpacity : defaultOpacity)
            .select("circle, rect, polygon")
            .attr("r", 5)
            .attr("width", 10)
            .attr("height", 10)
            .attr("points", "0,-5 5,5 -5,5");
            clearTimeout(hoverTimeout);
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        })
        .on("click", function(event, d) {
            showPopup('node', d);
        });

    nodeGroups.each(function(d) {
        const sel = d3.select(this);
        const color = d.organic ? "#7bc67b" : "#6ba3d6";
        if (sourceNodes.has(d.id)) {
            sel.append("rect")
                .attr("x", -5)
                .attr("y", -5)
                .attr("width", 10)
                .attr("height", 10)
                .attr("fill", color);
        } else if (targetNodes.has(d.id)) {
            sel.append("polygon")
                .attr("points", "0,-5 5,5 -5,5")
                .attr("fill", color);
        } else {
            sel.append("circle")
                .attr("r", 5)
                .attr("fill", color);
        }
    });

    const labels = nodeGroups.append("text")
        .text(d => d.name)
        .attr("font-size", 10)
        .attr("text-anchor", "middle")
        .attr("dy", 20)
        .attr("fill", "#4a4a6a")
        .attr("font-weight", 500);

    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    simulation.on("tick", () => {
        link
            .attr("x1", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                const normX = dx / dr;
                const sourcePadding = d.type === 'bi' ? 8 : 0;
                return d.source.x + (normX * sourcePadding);
            })
            .attr("y1", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                const normY = dy / dr;
                const sourcePadding = d.type === 'bi' ? 8 : 0;
                return d.source.y + (normY * sourcePadding);
            })
            .attr("x2", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                const normX = dx / dr;
                const targetPadding = (d.type === 'directed' || d.type === 'bi') ? 8 : 0;
                return d.target.x - (normX * targetPadding);
            })
            .attr("y2", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                const normY = dy / dr;
                const targetPadding = (d.type === 'directed' || d.type === 'bi') ? 8 : 0;
                return d.target.y - (normY * targetPadding);
            });

        nodeGroups
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    const zoom = d3.zoom()
        .scaleExtent([0.5, 4])
        .on("zoom", ({transform}) => {
            g.attr("transform", transform);
        });

    svg.call(zoom);
}


function makeDraggable(element) {
    const header = element.querySelector('.popup-header');
    header.addEventListener('mousedown', function(e) {
        e.preventDefault();
        let shiftX = e.clientX - element.getBoundingClientRect().left;
        let shiftY = e.clientY - element.getBoundingClientRect().top;

        function moveAt(pageX, pageY) {
            element.style.left = pageX - shiftX + 'px';
            element.style.top = pageY - shiftY + 'px';
        }

        function onMouseMove(e) {
            moveAt(e.pageX, e.pageY);
        }

        document.addEventListener('mousemove', onMouseMove);

        document.addEventListener('mouseup', function() {
            document.removeEventListener('mousemove', onMouseMove);
        }, {once: true});
    });
}