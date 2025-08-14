// Initialize Preline UI components and main characters data loading
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
    
    // Load main characters data if we're on the today page
    if (window.location.pathname.includes('today.html') || 
        window.location.pathname.endsWith('/today') ||
        document.getElementById('main-characters-section')) {
        loadMainCharacters();
        
        // Refresh data every 2 minutes
        setInterval(loadMainCharacters, 120000);
    }
});

// Main Characters Data Loading
async function loadMainCharacters() {
    try {
        console.log('Loading main characters data...');
        
        // Fetch data from backend
        const response = await fetch('./today.json');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Update the UI with fresh data
        renderMainCharacters(data.main_characters || []);
        updateTimestamps(data.metadata || {});
        
        console.log(`Loaded ${data.main_characters?.length || 0} main characters`);
        
    } catch (error) {
        console.error('Failed to load main characters data:', error);
        
        // If loading fails, keep template data but show error indicator
        updateTimestamps({
            last_metrics_update: new Date().toISOString(),
            error: true
        });
    }
}

function renderMainCharacters(characters) {
    const container = document.querySelector('#main-characters-table tbody');
    
    if (!container) {
        console.warn('Main characters table not found');
        return;
    }
    
    // If no characters, keep template data
    if (!characters || characters.length === 0) {
        console.log('No main characters data, keeping template');
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Render each character
    characters.forEach((character, index) => {
        const row = createMainCharacterRow(character, index);
        container.appendChild(row);
    });
    
    // Re-initialize Preline accordion components
    setTimeout(() => {
        if (window.HSAccordion) {
            window.HSAccordion.autoInit();
        }
    }, 100);
}

function createMainCharacterRow(character, index) {
    const row = document.createElement('tr');
    row.className = 'hs-accordion';
    row.id = `hs-table-accordion-char-${index}`;
    
    // Determine controversy color
    const controversyColor = getControversyColor(character.controversy);
    const changeColor = getChangeColor(character.change_type);
    
    row.innerHTML = `
        <td colspan="5" class="p-0">
            <button class="hs-accordion-toggle hs-accordion-active:text-blue-600 w-full text-center text-gray-800 hover:text-gray-500 disabled:opacity-50 disabled:pointer-events-none dark:hs-accordion-active:text-blue-500 dark:text-neutral-200 dark:hover:text-neutral-400" 
                    aria-expanded="false" 
                    aria-controls="hs-table-collapse-char-${index}" 
                    aria-label="Toggle ${character.user.handle} details">
                <div class="py-4 px-5">
                    <!-- Row 1: User + Metrics -->
                    <div class="grid grid-cols-[2fr_1fr_1fr_1fr_auto] gap-4 items-center mb-3">
                        <div class="font-medium text-left flex items-center gap-3">
                            <span class="inline-flex items-center justify-center size-11 text-sm font-semibold rounded-full border border-gray-800 text-gray-800 dark:border-neutral-200 dark:text-white">
                                ${character.user.initials}
                            </span>
                            @${character.user.handle}
                        </div>
                        <div class="text-sm text-gray-600 dark:text-neutral-400 text-center">${character.ratio}</div>
                        <div class="text-sm ${controversyColor} font-semibold text-center">${character.controversy}/10</div>
                        <div class="status-indicator ${changeColor} text-center">${character.change}</div>
                        <div class="justify-self-end">
                            <svg class="hs-accordion-active:hidden block size-3.5" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M5 12h14"></path>
                                <path d="M12 5v14"></path>
                            </svg>
                            <svg class="hs-accordion-active:block hidden size-3.5" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M5 12h14"></path>
                            </svg>
                        </div>
                    </div>
                    <!-- Row 2: Post -->
                    <div class="text-sm text-gray-600 dark:text-neutral-400 italic mb-2 text-left">
                        "${character.post.text}"
                    </div>
                    <!-- Row 3: Post Metrics -->
                    <div class="flex items-center gap-6 text-xs text-gray-500 dark:text-neutral-500">
                        <span>${character.engagement.formatted.reposts} RT</span>
                        <span>${character.engagement.formatted.quotes} QT</span>
                        <span>${character.engagement.formatted.replies} Replies</span>
                        <span>${character.engagement.formatted.likes} Likes</span>
                    </div>
                </div>
            </button>
            <div id="hs-table-collapse-char-${index}" class="hs-accordion-content hidden w-full overflow-hidden transition-[height] duration-300" role="region" aria-labelledby="hs-table-accordion-char-${index}">
                <div class="pb-4 px-5">
                    <div class="space-y-3 text-sm text-gray-600 dark:text-neutral-400 mt-3">
                        ${createRepliesHtml(character.sample_replies)}
                    </div>
                </div>
            </div>
        </td>
    `;
    
    return row;
}

function createRepliesHtml(replies) {
    if (!replies || replies.length === 0) {
        return '<div class="text-center text-gray-500">No replies captured</div>';
    }
    
    return replies.map(reply => `
        <div class="flex items-start gap-3">
            <div class="flex flex-col items-center">
                <div class="w-2 h-2 bg-gray-400 dark:bg-neutral-500 rounded-full"></div>
                <div class="w-px h-6 bg-gray-300 dark:bg-neutral-600"></div>
            </div>
            <span>"${reply}"</span>
        </div>
    `).join('');
}

function getControversyColor(score) {
    if (score >= 8) return 'text-red-600 dark:text-red-400';
    if (score >= 5) return 'text-orange-600 dark:text-orange-400';
    return 'text-gray-600 dark:text-neutral-400';
}

function getChangeColor(changeType) {
    switch (changeType) {
        case 'positive': return 'positive';
        case 'negative': return 'negative';
        case 'new': return 'new';
        default: return 'neutral';
    }
}

function updateTimestamps(metadata) {
    // Update "Updated X min ago" timestamp
    const timestampElement = document.querySelector('[data-timestamp]') || 
                            document.querySelector('.text-sm.text-gray-500:last-child');
    
    if (timestampElement && metadata.last_metrics_update) {
        try {
            const lastUpdate = new Date(metadata.last_metrics_update);
            const now = new Date();
            const diffMinutes = Math.floor((now - lastUpdate) / (1000 * 60));
            
            let timeAgo;
            if (diffMinutes < 1) {
                timeAgo = 'just now';
            } else if (diffMinutes < 60) {
                timeAgo = `${diffMinutes} min ago`;
            } else {
                const diffHours = Math.floor(diffMinutes / 60);
                timeAgo = `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
            }
            
            timestampElement.textContent = metadata.error ? 'Update failed' : `Updated ${timeAgo}`;
            
        } catch (error) {
            console.error('Error updating timestamp:', error);
        }
    }
    
    // Update collection info if available
    if (metadata.total_posts_analyzed) {
        const statsElement = document.querySelector('[data-stats]');
        if (statsElement) {
            statsElement.textContent = `${metadata.total_posts_analyzed.toLocaleString()} posts analyzed`;
        }
    }
}