// ipython_notebook_goodies
// ========================
// Make a table of contents for your notebook. Uses headings (e.g. H1, H2, etc.) to build TOC, 
// and provides anchors (added where needed).
// 
// Author: see https://github.com/kmahelona/ipython_notebook_goodies
//
// Changes (by jpmeuret@free.fr) :
// * removed articifial roman numbering
// * added support for notebook title
//
// Known bugs:
// * as jupyter notebook seems to force <Hi> ids to their innerHTML,
//   you can't hide the NB title in the generated TOC
// 
// **Usage:** 
// 
// 1. Add a *markdown* cell at the top of your notebook with the following code inside :
//    Note: the last div sections is mandatory
//    Note: but above it, you can add any HTML code, even with <h> elements
//          (with class="tocIgnore" if you don't want them to appear in the generated TOC)
// 
// <!-- Auto table of contents -->
// <h1 id='tocIgnoreNbTitle'>... title of notebook ...</h1>
// <p> ... some text as you like </p>
// <p> ... but may be any HTML you like </p>
// <div style="overflow-y: auto">
//   <h2 id='tocTitle'>Table of contents</h2>
//   <div id="toc"></div>
// </div>
// 
// 2. Add a *code* cell anywhere in the notebook with the following:
//    (provided that this file is named ipython_notebook_toc.js
//     and located in the folder of the notebook)
// 
// %%javascript
// var maxlevel = 3; // or 1, or 2, or 4, or ... for keeping only H1, H2, H3, H...
// $.getScript('ipython_notebook_toc.js', function() {createTOC(maxlevel);})
// 
// 3. Re-run it to update the TOC

// Builds a <ul> Table of Contents from all <h> in DOM (ex: <h3> => level 3 nested <ul>)
function createTOC(maxlevel) {
    let toc = "";
    let level = 0;
    let levels = {};
    $('#toc').html('');

    $(":header").each(function(i) {
        
        //$('#toc').append('<br>' + this.innerHTML + ' : ' + this.className);
        //return;
        
        // Ignore any <h> with 'tocIgnore' class
	    if (this.className == 'tocIgnore')
            return;
        
	    let titleText = this.innerHTML;
	    let openLevel = this.tagName[1];

	    if (levels[openLevel]){
            levels[openLevel] += 1;
	    } else{
            levels[openLevel] = 1;
	    }

	    if (openLevel > level) {
            toc += (new Array(openLevel - level + 1)).join('<ul class="toc">');
	    } else if (openLevel < level) {
            toc += (new Array(level - openLevel + 1)).join("</ul>");
            for (i=level;i>openLevel;i--)
                levels[i]=0;
	    }

	    level = parseInt(openLevel);
        if (level <= maxlevel) {
    
    	    if (this.id=='')
                this.id = this.innerHTML.replace(/ /g,"-");
    	    let anchor = this.id;
            
    	    toc += '<li><a href="#' + encodeURIComponent(anchor) + '">'
                      + titleText
                   + '</a></li>';
        }
	});

    if (level)
        toc += (new Array(level + 1)).join("</ul>");

    $('#toc').append(toc);

};

// Executes the createToc function
//setTimeout(function(){createTOC(3);},100);

// Rebuild to TOC every minute
//setInterval(function(){createTOC();},60000);
