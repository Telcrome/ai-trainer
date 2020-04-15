import { Component, OnInit } from '@angular/core';
import { Log } from '../log';
import { Input } from '@angular/core';

@Component({
  selector: 'app-cell',
  templateUrl: './cell.component.html',
  styleUrls: ['./cell.component.scss']
})
export class CellComponent implements OnInit {

  @Input() log: Log;

  constructor() {
  }

  ngOnInit(): void {
    console.log(this.log);
  }

}
