/* add data-tilt attribute to all image elements */
var images = document.getElementsByTagName('img');
for (var i = 0; images[i]; i++) {
     images[i].setAttribute('data-tilt', '');
     images[i].setAttribute('data-tilt-axis', 'x');
     images[i].setAttribute('data-tilt-glare', 'true');
     images[i].setAttribute('data-tilt-maxGlare', '0.5');
}
