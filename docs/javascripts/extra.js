/* add data-tilt attribute to all image elements */
var images = document.getElementsByTagName('img');
for (var i = 0; images[i]; i++) {
     images[i].setAttribute('data-tilt', '');
     images[i].setAttribute('data-tilt-perspective', 1800);
}
