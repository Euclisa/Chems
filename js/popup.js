function showPopup(type, data) {
    // Create new popup container dynamically
    const popup = document.createElement('div');
    popup.classList.add('popup', `popup-${type}`);

    popup.innerHTML = `
        <div class="popup-header">
            <span id="popup-title">Title</span>
            <span class="popup-close" onclick="this.closest('.popup').remove()">×</span>
        </div>
        <div class="popup-content" id="popup-content">
        Content
        </div>
    `;

    document.body.appendChild(popup);

    const title = popup.querySelector('#popup-title');
    const content = popup.querySelector('#popup-content');

    if (type === 'node') {
        const compound = cidToCompound.get(data.id);
        title.textContent = compound ? compound.cmpdname : 'Node Info';
        
        let html = '<div class="compound-structure">';
        html += `<img src="data/structures/${data.id}.svg" alt="${compound.cmpdname}" onerror="this.style.display='none'">`;
        html += '</div>';
        
        html += '<div class="compound-info">';
        html += '<div class="compound-info-row">';
        html += '<span class="compound-info-label">CID:</span>';
        html += `<span class="compound-info-value"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/${data.id}" target="_blank">${data.id}</a></span>`;
        html += '</div>';
        
        if (compound && compound.wiki) {
            html += '<div class="compound-info-row">';
            html += '<span class="compound-info-label">Wikipedia:</span>';
            html += `<span class="compound-info-value"><a href="${compound.wiki}" target="_blank">${compound.wiki.split('/').at(-1).replace('_', ' ')}</a></span>`;
            html += '</div>';
        }
        
        html += '<div class="compound-info-row">';
        html += '<span class="compound-info-label">Type:</span>';
        html += `<span class="compound-info-value">${compound.organic ? 'Organic' : 'Inorganic'}</span>`;
        html += '</div>';

        html += '</div>';
        
        const description = cidToDescription.get(data.id);
        if (description) {
            html += '<div class="compound-info">';
            html += '<span class="compound-info-label description">Description:</span>';
            description_html = '<p>' + description.split('\n\n').join('</p><p>') + '</p>';
            html += `<span class="compound-info-value">${description_html}</span>`;
            html += '</div>';
            html += '</div>';
        }
        
        content.innerHTML = html;
    } else if (type === 'edge') {
        const sourceId = typeof data.source === 'object' ? data.source.id : data.source;
        const targetId = typeof data.target === 'object' ? data.target.id : data.target;
        const sourceComp = cidToCompound.get(sourceId);
        const targetComp = cidToCompound.get(targetId);
        const sourceName = sourceComp ? sourceComp.cmpdname : sourceId;
        const targetName = targetComp ? targetComp.cmpdname : targetId;
        
        title.textContent = 'Reactions';
        
        const edgeStr = `${sourceId}-${targetId}`;
        const reverseEdgeStr = `${targetId}-${sourceId}`;
        const forwardReactionIDs = edgeToReactionID.get(edgeStr) || [];
        const reverseReactionIDs = data.type === 'bi' ? (edgeToReactionID.get(reverseEdgeStr) || []) : [];
        
        let html = '';

        const generateReactionParticipants = (participants, balanced) => {
            let reactants_htmls = [];
            for (const entry of participants) {
                const cid = entry.cid;
                const compound = cidToCompound.get(cid);
                const name = compound.cmpdname;
                let curr_html = ""
                curr_html += `<div class="reaction-participant">`;
                curr_html += `<img class="reaction-svg-image" src="data/structures/${cid}.svg" alt="${name}" data-cid="${cid}">`;
                curr_html += `<span class="reaction-participant-name" title="${name}">${name}</span>`;
                curr_html += '</div>';
                
                if (balanced) {
                    const coeff = entry.coeff;
                    curr_html = `<div class="reaction-participant-balanced"><span class="reaction-participant-coeff">${coeff}</span>${curr_html}</div>`
                }
                reactants_htmls.push(curr_html)
            }
            return reactants_htmls.join('<span class="reaction-sep reaction-sep-plus">+</span>')
        };

        const generateReactionItem = (rid) => {
            const getConfidenceClass = (confidence) => {
                const classPrefix = 'reaction-confidence-'
                if (confidence < 0.5)
                    return classPrefix + 'low';
                else if (confidence < 0.7)
                    return classPrefix + 'medium';
                else
                    return classPrefix + 'high';
            };

            item_html = "";
            const reaction = RIDToReaction.get(rid);
            if (reaction) {
                const balanced = reaction.balanced;
                console.log(balanced)
                item_html += '<div class="reaction-item">';
                item_html += `<div class="reaction-equation">`;
                item_html += generateReactionParticipants(reaction.reagents, balanced);
                item_html += '<span class="reaction-sep reaction-sep-arrow">→</span>';
                item_html += generateReactionParticipants(reaction.products, balanced);
                item_html += '</div>';

                const description = ridToDescription.get(rid);
                if (description) {
                    item_html += '<div class="reaction-description">';
                    item_html += '<strong>Description:</strong> ';
                    item_html += description;
                    item_html += '</div>';
                }

                const confidence = reaction.confidence;
                if (confidence) {
                    const confidenceClass = getConfidenceClass(confidence);
                    item_html += `<span class="reaction-confidence ${confidenceClass}" title="This reaction is generated automatically and this is its validation score.">${confidence.toFixed(2)}</span>`;
                } else if (reaction.source == 'ord') {
                    item_html += `<img src="data/assets/ord.svg" class="reaction-ord-label" title="Sourced from Open Reaction Database">`
                }

                item_html += '</div>';
            }

            return item_html;
        };

        const generateReactionList = (rids) => {
            let reactionsHtml = "";
            if (rids.length > 0) {
                rids.forEach(rid => {
                    reactionsHtml += generateReactionItem(rid);
                });
            } else {
                reactionsHtml += '<div class="reaction-item">No reactions found</div>';
            }

            return reactionsHtml;
        };
        
        if (data.type === 'bi') {
            html += '<div class="popup-split">';
            
            html += '<div class="popup-direction">';
            html += `<h4> ${sourceName} → ${targetName}</h4>`;
            html += generateReactionList(forwardReactionIDs);
            html += '</div>';
            
            html += '<div class="popup-direction">';
            html += `<h4> ${targetName} → ${sourceName}</h4>`;
            html += generateReactionList(reverseReactionIDs);
            html += '</div>';
            
            html += '</div>';
        } else {
            html += '<div class="popup-direction">';
            html += `<h4> ${sourceName} → ${targetName}</h4>`;
            html += generateReactionList(forwardReactionIDs);
            html += '</div>';
        }
        
        content.innerHTML = html;
    }

    popup.style.display = 'block';
    popup.style.position = 'absolute';
    popup.style.visibility = 'visible';
    popup.style.opacity = '1';

    const rect = popup.getBoundingClientRect();
    popup.style.left = (window.innerWidth / 2 - rect.width / 2) + 'px';
    popup.style.top = (window.innerHeight / 2 - rect.height / 2) + 'px';

    makeDraggable(popup);

    document.querySelectorAll('.reaction-svg-image').forEach(el => {
        el.onclick = () => {
            const cid = Number(el.dataset.cid);
            showCompoundInfoPopup(cid);
        };
    });
}


function showCompoundInfoPopup(cid) {
    const name = cidToCompound.get(cid).cmpdname;
    showPopup('node', {id: cid, name: name});
}