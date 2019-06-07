function loadTestData(index) {
    console.log("index = " + index);
    var table = $("#test-data-table tbody")[0];
    console.log(table);
    // 设置src
    var cell = table.rows[index].cells[1]; // This is a DOM "TD" element
    var $cell = $(cell); // Now it's a jQuery object.
    $("#src").val($cell.text());
    // console.log(cell);
    // console.log($cell);
    // console.log("Cell text = " + $cell.text());
    // 设置mt
    cell = table.rows[index].cells[2];
    var $cell = $(cell);
    $("#mt").val($cell.text());
    // 设置pe
    cell = table.rows[index].cells[3];
    var $cell = $(cell);
    $("#pe").val($cell.text());
    // 设置hter
    cell = table.rows[index].cells[4];
    var $cell = $(cell);
    $("#hter").val($cell.text());
}

function resetGold() {
    // console.log("Changed!");
    $("#pe").val("");
    $("#hter").val("");
}

$(document).ready(function() {
    $("#src").on("input", function(e) {
        var input = $(this);
        var val = input.val();
        if (input.data("lastval") != val) {
            input.data("lastval", val);
            console.log(val);
            resetGold();
        }
    });
    $("#mt").on("input", function(e) {
        var input = $(this);
        var val = input.val();
        if (input.data("lastval") != val) {
            input.data("lastval", val);
            console.log(val);
            resetGold();
        }
    });
});
