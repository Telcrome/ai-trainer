import { Component, OnInit } from '@angular/core';
import { LogService } from '../log.service';

@Component({
  selector: 'app-log',
  templateUrl: './log.component.html',
  styleUrls: ['./log.component.scss']
})
export class LogComponent implements OnInit {

  constructor(private logs: LogService) { }
  title = this.logs.welcome;
  logList = this.logs.getLogs();
  model = {
    left: true,
    middle: false,
    right: false
  };
  ngOnInit(): void {
  }

}
